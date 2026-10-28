"""Async SAM 3.1 segmentation through Meta Model API.

Sends the full-resolution upload (never resized) to the documented
OpenAI-compatible Responses endpoint with one concept phrase per request,
streams the output-text lane, and decodes segmentation masks with the
official meta_sam_parser package.

Contract sources (verified 2026-10-27):
- POST {base_url}/responses, model="sam-3.1", input_text + input_image
  (public URL or base64 data URL), metadata.mask_encoding, stream to
  response.completed. https://dev.meta.ai/docs/sam/segmenting
- Output-text grammar: <Nf> lines, object id + box + mask tokens, H,W
  height-first, ~ lossless / ! one_bit base85 payloads.
  https://dev.meta.ai/docs/sam/reading-segmentation
- Python parsing/decoding: meta_sam_parser (parse_responses_stream,
  image_segmentation_format, decode_mask_to_raster).
  https://dev.meta.ai/docs/sam/client-libraries
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import Callable, Sequence

import aiohttp
import streamlit as st
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from PIL import Image, ImageDraw

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "sam-3.1"
CONCEPT_CATEGORIES = {
    "Regional Architecture & Construction": (
        "roof materials and styles including terracotta clay tiles, "
        "sloped tin sheet roofs, flat concrete slabs, and thatch",
        "wall finishes including laterite stone brick, unpainted concrete, "
        "brightly painted plaster, and mud plaster",
        "structural features including verandas, wooden pillars, compound "
        "walls, decorative gates, and window grills",
    ),
    "Flora, Agriculture & Soil": (
        "trees including coconut palms, areca nut palms, rubber trees, "
        "banyan and peepal trees, bamboo, pine, and banana plants",
        "crops and ground cover including paddy fields, tea and coffee "
        "bushes, spice foliage, and potted plants",
        "soil and ground including red laterite soil, black alluvial soil, "
        "sand, and clay",
    ),
    "Infrastructure & Utilities": (
        "electric and utility poles including concrete rectangular posts, "
        "wooden poles, lattice steel towers, and transformer mountings",
        "overhead wiring configurations and cable clutter density",
        "roads and edges including paved asphalt, red dirt tracks, "
        "interlock paver blocks, drainage canals, and gutters",
        "curbing and painted safety markings including yellow-black and "
        "white-black curbs",
    ),
    "Local Signage, Text & Script": (
        "shop front hoardings, banners, and store headers",
        "regional language scripts including Malayalam, Tamil, Hindi, "
        "Kannada, Telugu, and Bengali text",
        "milestones, highway shields, direction boards, and house nameplates",
    ),
    "Local Transport & Cultural Artifacts": (
        "auto-rickshaws, state transport buses including KSRTC, and trucks "
        "with body art",
        "vehicle license plates and registration color schemes including "
        "yellow and white plates",
    ),
}
FOCUS_CONCEPTS = tuple(
    phrase for phrases in CONCEPT_CATEGORIES.values() for phrase in phrases
)
CONCEPT_CATEGORY = {
    phrase: category
    for category, phrases in CONCEPT_CATEGORIES.items()
    for phrase in phrases
}
GENERAL_CATEGORY = "General"
RETRYABLE_STATUS_CODES = frozenset({429, 503, 504})
SAM_MAX_ATTEMPTS = 3  # Initial try plus up to 2 retries.
SAM_RETRY_WAIT = wait_exponential(multiplier=2, max=10)  # 2s, then 4s.
RETRY_STATUS_TEXT = "API congested, retrying..."


class SamApiError(Exception):
    """Any failure calling Meta Model API or decoding its masks."""


def get_meta_api_key(explicit: str | None = None) -> str:
    """Resolve the Meta key: explicit value or st.secrets['META_API_KEY']."""
    if explicit is None:
        try:
            explicit = st.secrets["META_API_KEY"]
        except KeyError as exc:
            raise SamApiError(
                "Missing Meta Model API key: add META_API_KEY to "
                ".streamlit/secrets.toml."
            ) from exc
    if not explicit:
        raise SamApiError(
            "Empty Meta Model API key: set META_API_KEY in "
            ".streamlit/secrets.toml."
        )
    return explicit


class SamRetryableError(SamApiError):
    """Transient Meta API failure (HTTP 429/503/504); safe to retry."""


def make_retry_before_sleep(on_retry: Callable[[int, float], None] | None):
    """Build a tenacity hook reporting (attempt_number, wait_seconds)."""

    def _hook(state) -> None:
        if on_retry is not None:
            on_retry(state.attempt_number, state.next_action.sleep)

    return _hook


def default_sam_retrying(
    on_retry: Callable[[int, float], None] | None = None,
) -> AsyncRetrying:
    """Retry 429/503/504 up to 2 times with 2s, 4s exponential backoff."""
    return AsyncRetrying(
        stop=stop_after_attempt(SAM_MAX_ATTEMPTS),
        wait=SAM_RETRY_WAIT,
        retry=retry_if_exception_type(SamRetryableError),
        reraise=True,
        before_sleep=make_retry_before_sleep(on_retry),
    )


def sniff_image_mime(data: bytes) -> str:
    """Return the MIME type for supported raw image bytes."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise ValueError("Unrecognized image bytes; expected JPEG, PNG, WebP, or GIF.")


def encode_png_bytes(image: Image.Image) -> bytes:
    """Losslessly encode a PIL image; dimensions are never changed."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def select_latest_masks(records: Sequence) -> list:
    """Keep the highest-revision mask record per identity."""
    latest = {}
    for record in records:
        if isinstance(record, SegmentationMaskRecord):
            previous = latest.get(record.identity)
            if previous is None or record.revision > previous.revision:
                latest[record.identity] = record
    return list(latest.values())


def _frame_index(record) -> int | None:
    frame = getattr(record, "frame", None)
    if isinstance(frame, int):
        return frame
    index = getattr(frame, "frame_index", None)
    return index if isinstance(index, int) else None


def _bounds_dict(record) -> dict | None:
    bounds = getattr(record, "bounds", None)
    if bounds is None:
        return None
    return {
        "left": bounds.left,
        "top": bounds.top,
        "right": bounds.right,
        "bottom": bounds.bottom,
    }


CATEGORY_COLORS = {
    "Regional Architecture & Construction": (80, 160, 255),
    "Flora, Agriculture & Soil": (80, 220, 120),
    "Infrastructure & Utilities": (255, 200, 80),
    "Local Signage, Text & Script": (255, 80, 80),
    "Local Transport & Cultural Artifacts": (200, 120, 255),
}
CONCEPT_COLORS = {
    phrase: CATEGORY_COLORS[category]
    for category, phrases in CONCEPT_CATEGORIES.items()
    for phrase in phrases
}
CONCEPT_COLORS.update(
    {
        "text and signage": (255, 80, 80),
        "architectural structure and roofs": (80, 160, 255),
        "infrastructure, utility poles, and road markings": (255, 200, 80),
        "distinct flora and trees": (80, 220, 120),
        "text": (255, 80, 80),
        "architectural features": (80, 160, 255),
        "vegetation": (80, 220, 120),
    }
)
DEFAULT_BOX_COLOR = (200, 200, 200)


def category_for_concept(concept: str | None) -> str:
    """Return the extraction category for a concept phrase."""
    return CONCEPT_CATEGORY.get(concept, GENERAL_CATEGORY)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_box(
    box: dict | None, width: int | float | None, height: int | float | None
) -> dict | None:
    """Normalize a pixel box to 0-1 top/left/width/height coordinates.

    Returns None when the box is missing or incomplete, or when the
    image dimensions are missing or non-positive.
    """
    if not isinstance(box, dict):
        return None
    try:
        left, top, right, bottom = (
            box["left"],
            box["top"],
            box["right"],
            box["bottom"],
        )
    except KeyError:
        return None
    if not all(_is_finite_number(v) for v in (left, top, right, bottom)):
        return None
    if (
        not _is_finite_number(width)
        or not _is_finite_number(height)
        or width <= 0
        or height <= 0
    ):
        return None

    def _clamp(value: float) -> float:
        return round(min(1.0, max(0.0, value)), 3)

    return {
        "top": _clamp(top / height),
        "left": _clamp(left / width),
        "width": _clamp((right - left) / width),
        "height": _clamp((bottom - top) / height),
    }


FOREGROUND_MIN_AREA = 0.20
MIDGROUND_MIN_AREA = 0.05


def describe_position(normalized: dict | None) -> str:
    """Describe a normalized box as vertical, depth, and lateral zone.

    Vertical and lateral split the frame into thirds; depth follows the
    relative box area (large/close is foreground, tiny/distant is
    background). Returns "unknown" without a usable box.
    """
    if not isinstance(normalized, dict):
        return "unknown"
    try:
        top, left, width, height = (
            normalized["top"],
            normalized["left"],
            normalized["width"],
            normalized["height"],
        )
    except KeyError:
        return "unknown"
    if not all(_is_finite_number(v) for v in (top, left, width, height)):
        return "unknown"
    center_x, center_y = left + width / 2, top + height / 2
    vertical = "upper" if center_y < 1 / 3 else "middle" if center_y < 2 / 3 else "lower"
    lateral = "left" if center_x < 1 / 3 else "center" if center_x < 2 / 3 else "right"
    area = width * height
    depth = (
        "foreground"
        if area >= FOREGROUND_MIN_AREA
        else "midground"
        if area >= MIDGROUND_MIN_AREA
        else "background"
    )
    return f"{vertical} {depth} {lateral}"


def format_visual_breakdown(sam_masks) -> str:
    """Format every mask as markdown grouped by extraction category.

    Accepts summarize_masks() output or a full segment_image() result dict.
    Each line carries the concept label, category, normalized box, and
    spatial position; raster bytes never appear.
    """
    if isinstance(sam_masks, dict):
        sam_masks = summarize_masks(sam_masks)
    summaries = list(sam_masks or [])
    lines = [f"## Visual breakdown ({len(summaries)} masks)"]
    if not summaries:
        lines.append("No segmentation masks detected.")
        return "\n".join(lines)
    grouped: dict[str, list[str]] = {}
    for mask in summaries:
        concept = mask.get("concept") or "unknown concept"
        category = category_for_concept(mask.get("concept"))
        normalized = normalize_box(
            mask.get("box"), mask.get("width"), mask.get("height")
        )
        if normalized is None:
            geometry = "box unknown"
            position = "unknown"
        else:
            geometry = (
                f"top={normalized['top']}, left={normalized['left']}, "
                f"width={normalized['width']}, height={normalized['height']}"
            )
            position = describe_position(normalized)
        confidence = mask.get("confidence")
        confidence_text = (
            f"{confidence:.2f}"
            if _is_finite_number(confidence)
            else "n/a"
        )
        grouped.setdefault(category, []).append(
            f"- {concept} #{mask.get('object_id', '?')} "
            f"(conf {confidence_text}) @ {geometry} \u2014 {position}"
        )
    ordered = [c for c in CONCEPT_CATEGORIES if c in grouped]
    ordered.extend(c for c in grouped if c not in ordered)
    for category in ordered:
        entries = grouped[category]
        lines.append(f"### {category} ({len(entries)})")
        lines.extend(entries)
    return "\n".join(lines)


def summarize_masks(result: dict | None) -> list[dict]:
    """Build a JSON-safe summary of segmentation masks (no raster bytes)."""
    summaries = []
    for mask in (result or {}).get("masks", []):
        info = mask.get("mask", {})
        summaries.append(
            {
                "concept": mask.get("concept"),
                "object_id": mask.get("object_id"),
                "confidence": mask.get("confidence"),
                "box": mask.get("box"),
                "width": info.get("width"),
                "height": info.get("height"),
                "pixel_count": info.get("pixel_count"),
            }
        )
    return summaries


def draw_mask_boxes(
    image: Image.Image, masks: Sequence[dict], width: int = 3
) -> Image.Image:
    """Return a copy with each mask's box drawn in its concept color."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for mask in masks:
        box = mask.get("box") or {}
        try:
            coords = [box["left"], box["top"], box["right"], box["bottom"]]
        except KeyError:
            continue
        color = CONCEPT_COLORS.get(mask.get("concept"), DEFAULT_BOX_COLOR)
        draw.rectangle(coords, outline=color, width=width)
        draw.text(
            (coords[0] + 2, coords[1] + 2),
            f"{mask.get('concept', '?')} #{mask.get('object_id', '?')}",
            fill=color,
        )
    return annotated


class SamSegmenter:
    """Async client for SAM 3.1 concept segmentation.

    One API request is issued per concept phrase (the documented contract
    takes a single coherent phrase per request); independent concepts run
    concurrently. Images are always sent at full resolution.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        mask_encoding: str = "one_bit",
        timeout: float = 120.0,
        max_concurrency: int = 3,
        session: aiohttp.ClientSession | None = None,
        retrying: AsyncRetrying | None = None,
        on_retry: Callable[[int, float], None] | None = None,
    ) -> None:
        """Create a client.

        retrying overrides the whole tenacity policy (test seam); on_retry
        is called as on_retry(attempt_number, wait_seconds) before each
        backoff sleep. Streamlit callers should show RETRY_STATUS_TEXT
        ("API congested, retrying...") from that callback.
        """
        key = get_meta_api_key(api_key)
        if mask_encoding not in ("one_bit", "lossless"):
            raise ValueError("mask_encoding must be 'one_bit' or 'lossless'.")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._mask_encoding = mask_encoding
        self._timeout = timeout
        self._max_concurrency = max(1, max_concurrency)
        self._injected_session = session
        self._owned_session: aiohttp.ClientSession | None = None
        self._retrying_override = retrying
        self._on_retry = on_retry

    def _new_retrying(self) -> AsyncRetrying:
        """Fresh retry policy per request (safe under concurrency)."""
        if self._retrying_override is not None:
            return self._retrying_override
        return default_sam_retrying(self._on_retry)

    async def _session(self) -> aiohttp.ClientSession:
        if self._injected_session is not None:
            return self._injected_session
        if self._owned_session is None or self._owned_session.closed:
            self._owned_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        return self._owned_session

    async def aclose(self) -> None:
        """Close the owned session, if any (injected sessions are untouched)."""
        if self._owned_session is not None and not self._owned_session.closed:
            await self._owned_session.close()
        self._owned_session = None

    async def __aenter__(self) -> "SamSegmenter":
        await self._session()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def _sse_events(self, response):
        """Yield parsed SSE event dicts from an open streaming response."""
        async for chunk in response.content:
            for line in chunk.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data and data != "[DONE]":
                    yield json.loads(data)

    async def _fetch_result(self, payload: dict, concept: str):
        """POST once and parse the streamed result (single attempt)."""
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
                        f"SAM request failed (HTTP {response.status}): {body}"
                    )
                    if response.status in RETRYABLE_STATUS_CODES:
                        raise SamRetryableError(message)
                    raise SamApiError(message)
                try:
                    parsed = parse_responses_stream(
                        self._sse_events(response),
                        image_segmentation_format(),
                    )
                    async with parsed:
                        return await parsed.final_result()
                except SamApiError:
                    raise
                except Exception as exc:
                    raise SamApiError(
                        f"SAM stream failed for concept {concept!r}: {exc}"
                    ) from exc
        except SamApiError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise SamApiError(f"SAM request failed: {exc}") from exc

    async def segment_concept(
        self, image_data_url: str, concept: str
    ) -> list[dict]:
        """Segment one concept phrase; return decoded mask dicts."""
        payload = {
            "model": self._model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": concept},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            "stream": True,
            "metadata": {"mask_encoding": self._mask_encoding},
        }
        async for attempt in self._new_retrying():
            with attempt:
                result = await self._fetch_result(payload, concept)
        if not isinstance(result.outcome, CompletedOutcome) or result.diagnostics:
            raise SamApiError(
                f"SAM request for {concept!r} did not complete: "
                f"{result.outcome} {result.diagnostics}"
            )
        masks = []
        for record in select_latest_masks(result.records):
            try:
                raster = decode_mask_to_raster(record.mask)
            except Exception as exc:
                raise SamApiError(
                    f"Could not decode mask for concept {concept!r}: {exc}"
                ) from exc
            masks.append(
                {
                    "object_id": record.object_id,
                    "identity": record.identity,
                    "revision": record.revision,
                    "frame": _frame_index(record),
                    "confidence": record.confidence,
                    "box": _bounds_dict(record),
                    "mask": {
                        "width": record.mask.width,
                        "height": record.mask.height,
                        "encoding": record.mask.encoding,
                        "pixel_count": sum(raster),
                        "raster": raster,
                    },
                }
            )
        return masks

    async def segment_image(
        self,
        image: Image.Image | bytes,
        concepts: str | Sequence[str] = FOCUS_CONCEPTS,
    ) -> dict:
        """Segment the full-resolution image for each concept, concurrently.

        Returns {"model", "concepts", "mask_count", "masks"} with one mask
        dict per decoded mask, each tagged with its concept phrase. Rasters
        are row-major bytes of 0/1 with exactly width*height entries.
        """
        if isinstance(concepts, str):
            concepts = (concepts,)
        concepts = tuple(concepts)
        if not concepts:
            raise ValueError("At least one concept phrase is required.")
        if isinstance(image, Image.Image):
            data = await asyncio.to_thread(encode_png_bytes, image)
            mime = "image/png"
        else:
            data = bytes(image)
            mime = sniff_image_mime(data)
        data_url = (
            f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def one(concept: str):
            async with semaphore:
                return concept, await self.segment_concept(data_url, concept)

        masks: list[dict] = []
        for concept, found in await asyncio.gather(
            *(one(concept) for concept in concepts)
        ):
            for mask in found:
                masks.append({"concept": concept, **mask})
        return {
            "model": self._model,
            "concepts": list(concepts),
            "mask_count": len(masks),
            "masks": masks,
        }
