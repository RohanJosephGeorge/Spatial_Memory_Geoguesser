"""Multi-step spatial reasoning over SAM masks via Muse Wonder.

Builds a strict geolocation prompt from SAM 3.1 segmentation output plus the
user's regional hint, calls Meta Model API with the Muse Wonder model, and
validates the strictly formatted JSON result. Async throughout (aiohttp);
transient 429/503/504 failures retry with the shared tenacity policy.

Model id note: Meta documents the Model API id as "muse-wonder-1.0"
(https://dev.meta.ai/docs/models). The model stays a constructor parameter
so a deployment-specific id can be passed instead.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import Callable, Sequence

import aiohttp
from PIL import Image

from sam_segmentation import (
    DEFAULT_BASE_URL,
    RETRYABLE_STATUS_CODES,
    SamApiError,
    SamRetryableError,
    default_sam_retrying,
    encode_png_bytes,
    format_visual_breakdown,
    get_meta_api_key,
    sniff_image_mime,
    summarize_masks,
)

DEFAULT_REASONING_MODEL = "muse-wonder-1.0"
REASONING_EFFORT = "max"
MIN_MASKS_FOR_FULL_INVENTORY = 3
REQUIRED_KEYS = (
    "latitude",
    "longitude",
    "city",
    "country",
    "reasoning_steps",
    "confidence_score",
    "precision_level",
)
MAX_REASONING_TOKENS = 250

# Authoritative strict rules for the geolocation prompt.
# build_spatial_prompt() also accepts strict_rules per call to override.
DEFAULT_STRICT_RULES = (
    "Exhaustive Visual Synthesis: Analyze every micro-detail directly from "
    "the image and provided masks: sun position/shadow angle, driving side, "
    "license plate aspect ratios, road paint colors, utility pole "
    "structures, foliage/soil types, architectural rooflines, and regional "
    "hints.",
    "Best-Effort Coordinates (No Easy Quits): DO NOT output null "
    "coordinates unless the image is completely pitch black or synthetic. "
    "If exact street-level accuracy cannot be determined, make your best "
    "probabilistic deduction based on regional geography, climate, and "
    "architecture.",
    "Graduated Precision Scale: Map precision_level dynamically to one of: "
    "'Public Landmark' (within 50m), 'Street Level' (within 200m), "
    "'Neighborhood Radius' (within 1km), 'City Level' (within 10km), or "
    "'State / Region Radius' (within 50km+).",
    "Confidence Calibration: Reflect your certainty in the confidence "
    "score (e.g., 0.35 for regional guess, 0.95 for exact landmark), but "
    "still provide valid coordinates for the regional center or most "
    "likely intersection.",
)
PRECISION_LEVELS = (
    "Public Landmark",
    "Street Level",
    "Neighborhood Radius",
    "City Level",
    "State / Region Radius",
)
HALT_PRECISION = "Unknown"
HALT_PHRASES = ("Synthetic image detected", "Pitch black image")
GEOLOCATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "latitude": {"type": ["number", "null"], "minimum": -90, "maximum": 90},
        "longitude": {
            "type": ["number", "null"],
            "minimum": -180,
            "maximum": 180,
        },
        "city": {"type": "string"},
        "country": {"type": "string"},
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "precision_level": {
            "type": "string",
            "enum": [*PRECISION_LEVELS, HALT_PRECISION],
        },
    },
    "required": list(REQUIRED_KEYS),
    "additionalProperties": False,
}


class SpatialReasoningError(Exception):
    """Any failure reasoning about a location or validating its result."""


def estimate_tokens(text: str) -> int:
    """Rough token count (whitespace heuristic) for conciseness checks."""
    return len(text.split())


FALLBACK_INSTRUCTION = (
    "FALLBACK (sparse segmentation): the mask inventory above is empty or "
    "minimal. Analyze the attached raw image directly for geolocation "
    "evidence and do not output null coordinates solely because "
    "segmentation masks are missing."
)
ATTACHED_IMAGE_NOTE = (
    "NOTE: the full-resolution raw image is attached alongside this "
    "inventory; use it to corroborate or extend the mask evidence."
)
OCR_INSTRUCTION = (
    "Use the Local Signage, Text & Script masks to locate text regions, "
    "then actively extract and decode text micro-details from the attached "
    "raw image: shop headers, street signs, highway numbers, vehicle "
    "license plate state prefixes (e.g. 'KL' for Kerala, 'MH' for "
    "Maharashtra), domain extensions (e.g. .in, .co.uk), and phone area "
    "codes. Recognize regional scripts (e.g. Malayalam, Tamil, Hindi, "
    "Cyrillic, Arabic, Hangul) and explicitly identify the language and "
    "geographic region associated with each script."
)
TEXT_ANCHOR_INSTRUCTION = (
    "If legible text, shop names, or district names are detected, "
    "prioritize them as Tier-1 primary anchor points. Use the text to "
    "narrow down the specific district, town, or transit corridor before "
    "falling back to landscape cues."
)
TRIANGULATION_HEURISTICS = (
    "Solar Geometry & Shadow Analysis: Evaluate shadow angle and light "
    "direction relative to visible structures and trees to estimate the "
    "approximate hemisphere (Northern vs. Southern) and latitude alignment.",
    "Infrastructure Fingerprinting: Classify utility pole architecture "
    "(concrete rectangular RCC posts, wooden cross-arms, steel lattice "
    "towers, cylindrical steel); check driving side (left vs. right-hand "
    "traffic), curb paint patterns (yellow/black, red/white, plain "
    "concrete), and asphalt vs. dirt road composition.",
    "Vegetation & Biome Indexing: Match plant species (rubber plantation "
    "vs. coconut palm vs. pine forest) and soil color (red laterite, black "
    "soil, sandy coastal) to specific climatic sub-zones.",
)


def build_spatial_prompt(
    sam_masks,
    regional_hint: str,
    *,
    strict_rules: Sequence[str] | None = None,
    max_reasoning_tokens: int = MAX_REASONING_TOKENS,
    has_image: bool = False,
) -> str:
    """Format SAM masks + hint into a strict geolocation prompt.

    sam_masks accepts summarize_masks() output or a full segment_image()
    result dict (summarized automatically; raster bytes never enter the
    prompt). Every mask also appears in a VISUAL BREAKDOWN section with
    its category, normalized box, and spatial position. When has_image is
    True, an empty or minimal inventory adds a fallback instruction to
    analyze the attached raw image directly instead of halting, and a TEXT
    DECODING section directs OCR of signage, codes, and regional scripts
    with Tier-1 text anchoring. A STRUCTURAL TRIANGULATION section always
    follows with solar, infrastructure, and biome deduction heuristics.
    """
    if isinstance(sam_masks, dict):
        sam_masks = summarize_masks(sam_masks)
    rules = (
        list(strict_rules) if strict_rules is not None else list(DEFAULT_STRICT_RULES)
    )
    hint = (regional_hint or "").strip() or "(no hint provided)"
    rules_text = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1))
    summaries = list(sam_masks)
    masks_json = json.dumps(summaries, indent=2)
    breakdown = format_visual_breakdown(summaries)
    triangulation = "\n".join(f"- {item}" for item in TRIANGULATION_HEURISTICS)
    visual_note = ""
    ocr_section = ""
    if has_image:
        if len(summaries) < MIN_MASKS_FOR_FULL_INVENTORY:
            visual_note = f"\n{FALLBACK_INSTRUCTION}\n"
        else:
            visual_note = f"\n{ATTACHED_IMAGE_NOTE}\n"
        ocr_section = (
            f"TEXT DECODING (OCR):\n{OCR_INSTRUCTION}\n"
            f"{TEXT_ANCHOR_INSTRUCTION}\n\n"
        )
    return f"""You are an elite, world-class reverse geolocation expert and spatial reasoning engine. Your goal is to pinpoint the location of any image using all available visual and contextual evidence, making educated, highly reasoned deductions even when features are subtle.

Reason in multiple hypotheses: (1) generate 3 distinct candidate locations (Candidate A, B, C) from the visual evidence; (2) score every candidate against all extracted features (architecture, flora, soil, infrastructure, text) in an evidence matrix and eliminate candidates that violate key constraints; (3) assign a confidence weighting to the surviving winner and output its exact geographic center or landmark coordinates.

SEGMENTATION MASKS (concept, object id, box, size, confidence):
{masks_json}

VISUAL BREAKDOWN (label, category, normalized box, position):
{breakdown}
{visual_note}
{ocr_section}STRUCTURAL TRIANGULATION:
{triangulation}

REGIONAL HINT: {hint}

STRICT DEDUCTION RULES:
{rules_text}

REQUIRED OUTPUT: a single JSON object with exactly these keys:
- latitude: number between -90 and 90, or null when halting
- longitude: number between -180 and 180, or null when halting
- city: string ("" when the exact city is unknown or when halting)
- country: string ("" when the exact country is unknown or when halting)
- reasoning_steps: non-empty array of concise strings, under {max_reasoning_tokens} tokens in total, narrating the 3 candidates, the evidence-matrix elimination, and the winning selection
- confidence_score: number between 0 and 1 (exactly 0.0 when halting)
- precision_level: one of 'Public Landmark', 'Street Level', 'Neighborhood Radius', 'City Level', 'State / Region Radius' ('Unknown' when halting)

HALT FORMAT: halt only when the image is completely pitch black or synthetic. Then output latitude null, longitude null, city "", country "", confidence_score 0.0, precision_level "Unknown", and put the state phrase (e.g. 'Synthetic image detected', 'Pitch black image') as the first reasoning step. Never output 0.0, 0.0 as a placeholder."""


def extract_output_text(response: dict) -> str:
    """Pull concatenated assistant text from a Responses API payload."""
    texts: list[str] = []
    output = response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    if not texts and isinstance(response, dict):
        for key in ("output_text", "text"):
            if isinstance(response.get(key), str):
                texts.append(response[key])
    combined = "".join(texts).strip()
    if not combined:
        raise SpatialReasoningError("Model response contained no text.")
    return combined


def _load_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise SpatialReasoningError(
                "Model did not return valid JSON."
            ) from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise SpatialReasoningError(
                f"Model did not return valid JSON: {exc}"
            ) from exc
    if not isinstance(parsed, dict):
        raise SpatialReasoningError("Model result must be a JSON object.")
    return parsed


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_reasoning_result(text: str) -> dict:
    """Parse and strictly validate the model's JSON geolocation result.

    Halted results use the null convention: null coordinates, empty
    city/country, 0.0 confidence, and "Unknown" precision. Non-halted
    results keep their coordinates even when city/country are empty
    strings (coarse regional guesses); only the string type is enforced.
    """
    parsed = _load_json_object(text)
    missing = [key for key in REQUIRED_KEYS if key not in parsed]
    if missing:
        raise SpatialReasoningError(
            f"Model result is missing keys: {', '.join(missing)}."
        )
    extra = [key for key in parsed if key not in REQUIRED_KEYS]
    if extra:
        raise SpatialReasoningError(
            f"Model result has forbidden extra keys: {', '.join(extra)}."
        )
    lat, lon = parsed["latitude"], parsed["longitude"]
    halted = lat is None or lon is None
    if halted and not (lat is None and lon is None):
        raise SpatialReasoningError(
            "latitude and longitude must both be null or both be numbers."
        )
    if not halted:
        if not _is_number(lat) or not -90 <= lat <= 90:
            raise SpatialReasoningError(
                "latitude must be a number from -90 to 90."
            )
        if not _is_number(lon) or not -180 <= lon <= 180:
            raise SpatialReasoningError(
                "longitude must be a number from -180 to 180."
            )
    for key in ("city", "country"):
        value = parsed[key]
        if not isinstance(value, str):
            raise SpatialReasoningError(f"{key} must be a string.")
        if halted and value != "":
            raise SpatialReasoningError(
                f"{key} must be empty when coordinates are null."
            )
    steps = parsed["reasoning_steps"]
    if (
        not isinstance(steps, list)
        or not steps
        or any(not isinstance(step, str) or not step.strip() for step in steps)
    ):
        raise SpatialReasoningError(
            "reasoning_steps must be a non-empty array of non-empty strings."
        )
    confidence = parsed["confidence_score"]
    if not _is_number(confidence) or not 0 <= confidence <= 1:
        raise SpatialReasoningError(
            "confidence_score must be a number from 0 to 1."
        )
    if halted and confidence != 0:
        raise SpatialReasoningError(
            "confidence_score must be 0.0 when coordinates are null."
        )
    precision = parsed["precision_level"]
    if halted and precision != HALT_PRECISION:
        raise SpatialReasoningError(
            f'precision_level must be "{HALT_PRECISION}" when halted.'
        )
    if not halted and precision not in PRECISION_LEVELS:
        raise SpatialReasoningError(
            f"precision_level must be one of {', '.join(PRECISION_LEVELS)}."
        )
    return parsed


class LocationUnknownError(SpatialReasoningError):
    """The model returned Null Island (0.0, 0.0): location unknown."""


def check_null_island(parsed: dict) -> dict:
    """Intercept Null Island results; return anything else unchanged."""
    if parsed.get("latitude") == 0 and parsed.get("longitude") == 0:
        raise LocationUnknownError(
            "Location Unknown: the model returned coordinates 0.0, 0.0."
        )
    return parsed


def encode_image_data_url(image: Image.Image | bytes) -> str:
    """Encode a PIL image or raw image bytes as a base64 data URL."""
    if isinstance(image, Image.Image):
        data = encode_png_bytes(image)
        mime = "image/png"
    elif isinstance(image, (bytes, bytearray)):
        data = bytes(image)
        try:
            mime = sniff_image_mime(data)
        except ValueError as exc:
            raise SpatialReasoningError(
                f"Cannot attach image to reasoning request: {exc}"
            ) from exc
    else:
        raise SpatialReasoningError(
            "Reasoning image must be a PIL image or raw image bytes, "
            f"got {type(image).__name__}."
        )
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def build_reasoning_payload(
    prompt: str,
    *,
    model: str = DEFAULT_REASONING_MODEL,
    image_data_url: str | None = None,
) -> dict:
    """Build the Meta Responses API payload for Muse Wonder.

    The prompt carries the SAM mask inventory; image_data_url attaches
    the raw image as a second multimodal input part when provided.
    Reasoning always runs at maximum effort.
    """
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    if image_data_url is not None:
        content.append({"type": "input_image", "image_url": image_data_url})
    return {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": content,
            }
        ],
        "stream": False,
        "reasoning": {"effort": REASONING_EFFORT},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "Geolocation",
                "schema": GEOLOCATION_JSON_SCHEMA,
            }
        },
    }


class SpatialReasoner:
    """Async Muse Wonder client for mask-based geolocation."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_REASONING_MODEL,
        timeout: float = 120.0,
        session: aiohttp.ClientSession | None = None,
        retrying=None,
        on_retry: Callable[[int, float], None] | None = None,
    ) -> None:
        self._api_key = get_meta_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._injected_session = session
        self._owned_session: aiohttp.ClientSession | None = None
        self._retrying_override = retrying
        self._on_retry = on_retry

    async def _session(self) -> aiohttp.ClientSession:
        if self._injected_session is not None:
            return self._injected_session
        if self._owned_session is None or self._owned_session.closed:
            self._owned_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        return self._owned_session

    async def aclose(self) -> None:
        if self._owned_session is not None and not self._owned_session.closed:
            await self._owned_session.close()
        self._owned_session = None

    async def __aenter__(self) -> "SpatialReasoner":
        await self._session()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def _new_retrying(self):
        if self._retrying_override is not None:
            return self._retrying_override
        return default_sam_retrying(self._on_retry)

    async def _fetch_json(self, payload: dict):
        session = await self._session()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with session.post(
                f"{self._base_url}/responses",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    try:
                        body = (await response.text())[:500]
                    except Exception:
                        body = "<unreadable body>"
                    message = (
                        f"Reasoning request failed "
                        f"(HTTP {response.status}): {body}"
                    )
                    if response.status in RETRYABLE_STATUS_CODES:
                        raise SamRetryableError(message)
                    raise SpatialReasoningError(message)
                try:
                    return await response.json()
                except Exception as exc:
                    raise SpatialReasoningError(
                        f"Reasoning response was not JSON: {exc}"
                    ) from exc
        except (SamApiError, SpatialReasoningError):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise SpatialReasoningError(f"Reasoning request failed: {exc}") from exc

    async def reason(
        self,
        sam_masks,
        regional_hint: str,
        *,
        strict_rules: Sequence[str] | None = None,
        image: Image.Image | bytes | None = None,
    ) -> dict:
        """Run multi-step spatial reasoning; return the validated result.

        image attaches the raw photo as a second multimodal input part
        next to the SAM mask inventory. Reasoning always proceeds, even
        with zero masks: an empty inventory plus an attached image adds
        a fallback instruction to analyze the raw image directly.
        """
        prompt = build_spatial_prompt(
            sam_masks,
            regional_hint,
            strict_rules=strict_rules,
            has_image=image is not None,
        )
        payload = build_reasoning_payload(
            prompt,
            model=self._model,
            image_data_url=(
                encode_image_data_url(image) if image is not None else None
            ),
        )
        async for attempt in self._new_retrying():
            with attempt:
                data = await self._fetch_json(payload)
        parsed = parse_reasoning_result(extract_output_text(data))
        return check_null_island(parsed)
