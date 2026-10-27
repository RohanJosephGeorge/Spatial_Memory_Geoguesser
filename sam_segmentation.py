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
from PIL import Image

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "sam-3.1"
FOCUS_CONCEPTS = ("text", "architectural features", "vegetation")
RETRYABLE_STATUS_CODES = frozenset({429, 503, 504})
SAM_MAX_ATTEMPTS = 3  # Initial try plus up to 2 retries.
SAM_RETRY_WAIT = wait_exponential(multiplier=2, max=10)  # 2s, then 4s.
RETRY_STATUS_TEXT = "API congested, retrying..."


class SamApiError(Exception):
    """Any failure calling Meta Model API or decoding its masks."""


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
        if api_key is None:
            try:
                api_key = st.secrets["META_API_KEY"]
            except KeyError as exc:
                raise SamApiError(
                    "Missing Meta Model API key: add META_API_KEY to "
                    ".streamlit/secrets.toml."
                ) from exc
        if not api_key:
            raise SamApiError(
                "Empty Meta Model API key: set META_API_KEY in "
                ".streamlit/secrets.toml."
            )
        key = api_key
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
