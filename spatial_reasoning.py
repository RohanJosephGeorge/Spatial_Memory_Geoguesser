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
import json
import re
from collections.abc import Callable, Sequence

import aiohttp

from sam_segmentation import (
    DEFAULT_BASE_URL,
    RETRYABLE_STATUS_CODES,
    SamApiError,
    SamRetryableError,
    default_sam_retrying,
    format_visual_breakdown,
    get_meta_api_key,
    summarize_masks,
)

DEFAULT_REASONING_MODEL = "muse-wonder-1.0"
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
    "Privacy Override: You may receive images where small background faces "
    "or license plates have been intentionally blurred out for safety. Do "
    "not let these blurred regions trigger a Privacy Override. Ignore the "
    "blurred spots and focus on the surrounding architecture. Only halt "
    "analysis and output null if you detect a clear, unblurred human face.",
    "Synthetic Check: If the image appears to be from a video game, 3D "
    "render, or AI generator, output null coordinates and state 'Synthetic "
    "image detected'.",
    "Indoor/Featureless Check: If SAM 3.1 segmentation reveals only indoor "
    "furniture or lacks identifiable architectural or botanical clues, DO "
    "NOT guess. Output null coordinates and state 'Not enough distinct "
    "features'.",
    "Text Override: The user's regional hint is secondary to visual "
    "evidence. If visual clues heavily contradict the hint, ignore the hint.",
    "Privacy Radius: Do not output exact coordinates for private "
    "residences. Snap the final coordinates to the nearest public "
    "intersection. Set precision_level to either 'Public Landmark', "
    "'Street Level', or 'Neighborhood Radius'.",
)
PRECISION_LEVELS = ("Public Landmark", "Street Level", "Neighborhood Radius")
HALT_PRECISION = "Unknown"
HALT_PHRASES = ("Synthetic image detected", "Not enough distinct features")
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


def build_spatial_prompt(
    sam_masks,
    regional_hint: str,
    *,
    strict_rules: Sequence[str] | None = None,
    max_reasoning_tokens: int = MAX_REASONING_TOKENS,
) -> str:
    """Format SAM masks + hint into a strict geolocation prompt.

    sam_masks accepts summarize_masks() output or a full segment_image()
    result dict (summarized automatically; raster bytes never enter the
    prompt). Every mask also appears in a VISUAL BREAKDOWN section with
    its category, normalized box, and spatial position.
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
    return f"""You are a geospatial reasoning engine. Geolocate the photo from its SAM 3.1 segmentation masks and a regional hint.

Reason in multiple steps: (1) inventory the visible evidence per concept; (2) combine independent clues into candidate regions; (3) eliminate candidates inconsistent with any clue; (4) commit to one fix with a calibrated confidence score.

SEGMENTATION MASKS (concept, object id, box, size, confidence):
{masks_json}

VISUAL BREAKDOWN (label, category, normalized box, position):
{breakdown}

REGIONAL HINT: {hint}

STRICT RULES:
{rules_text}

REQUIRED OUTPUT: a single JSON object with exactly these keys:
- latitude: number between -90 and 90, or null when halting
- longitude: number between -180 and 180, or null when halting
- city: non-empty string ("" when halting)
- country: non-empty string ("" when halting)
- reasoning_steps: non-empty array of concise strings, under {max_reasoning_tokens} tokens in total
- confidence_score: number between 0 and 1 (exactly 0.0 when halting)
- precision_level: one of 'Public Landmark', 'Street Level', 'Neighborhood Radius' ('Unknown' when halting)

HALT FORMAT: when a strict rule halts analysis, output latitude null, longitude null, city "", country "", confidence_score 0.0, precision_level "Unknown", and put the state phrase (e.g. 'Synthetic image detected', 'Not enough distinct features') as the first reasoning step. Never output 0.0, 0.0 as a placeholder."""


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
    city/country, 0.0 confidence, and "Unknown" precision.
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
        if not halted and not value.strip():
            raise SpatialReasoningError(f"{key} must be a non-empty string.")
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
    ) -> dict:
        """Run multi-step spatial reasoning; return the validated result."""
        prompt = build_spatial_prompt(
            sam_masks, regional_hint, strict_rules=strict_rules
        )
        payload = {
            "model": self._model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "stream": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Geolocation",
                    "schema": GEOLOCATION_JSON_SCHEMA,
                }
            },
        }
        async for attempt in self._new_retrying():
            with attempt:
                data = await self._fetch_json(payload)
        parsed = parse_reasoning_result(extract_output_text(data))
        return check_null_island(parsed)
