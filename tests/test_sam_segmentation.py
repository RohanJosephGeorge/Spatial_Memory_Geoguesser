"""Tests for the async SAM 3.1 client: request fan-out, mask parsing,
full-resolution uploads, and error mapping.

Run: python -m unittest discover -s tests -v
Requires aiohttp, meta-sam-parser, numpy, Pillow. No network: HTTP is
fully mocked. Fixture mask payloads are generated with the official
parser's own encoder so they are valid by construction.
"""

import base64
import io
import json
import unittest
import unittest.mock
from unittest.mock import patch

import numpy as np
from meta_sam_parser import SegmentationMaskRecord
from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_none,
)

from sam_segmentation import (
    FOCUS_CONCEPTS,
    RETRY_STATUS_TEXT,
    SAM_RETRY_WAIT,
    SamApiError,
    SamRetryableError,
    SamSegmenter,
    draw_mask_boxes,
    select_latest_masks,
    sniff_image_mime,
    summarize_masks,
)


def fixture_line():
    raster = [1, 0, 0, 1]
    payload = _encode_raster(raster, 2, 2)
    if payload[:1] not in "!~":
        payload = "!" + payload
    return (
        "<0f>0<|box;x1=1;y1=2;x2=10;y2=12;w=64;h=48|>"
        f"<|mask;x=0;y=0;data=2,2,{payload}|>"
    )


def sse_bytes(events):
    lines = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return lines.encode("utf-8")


def success_events():
    line = fixture_line()
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    return [
        {"type": "response.output_text.delta", **lane, "delta": line},
        {"type": "response.output_text.done", **lane, "text": line},
        {"type": "response.completed"},
    ]


class FakeContent:
    def __init__(self, data: bytes):
        self._data = data

    def __aiter__(self):
        async def gen():
            for line in self._data.splitlines(keepends=True):
                yield line

        return gen()


class FakeResponse:
    def __init__(self, status=200, events=(), body=""):
        self.status = status
        self._events = events
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def text(self):
        return self._body

    @property
    def content(self):
        return FakeContent(sse_bytes(self._events))


class FakeSession:
    def __init__(self, response):
        self.calls = []
        self._responses = (
            response if isinstance(response, list) else [response]
        )

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def fast_retrying():
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_none(),
        retry=retry_if_exception_type(SamRetryableError),
        reraise=True,
    )


def error_response(status):
    return FakeResponse(status=status, body=f'{{"error": {{"code": {status}}}}}')


def success_response():
    return FakeResponse(events=success_events())


def gradient_png(width=32, height=24):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    rgb = np.stack([x, x, x], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class FanOutTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_request_per_focus_concept(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="test-key", session=session)
        result = await seg.segment_image(gradient_png())
        self.assertEqual(len(session.calls), len(FOCUS_CONCEPTS))
        self.assertEqual(result["concepts"], list(FOCUS_CONCEPTS))
        self.assertEqual(result["model"], "sam-3.1")
        seen = set()
        for call in session.calls:
            self.assertEqual(call["url"], "https://api.meta.ai/v1/responses")
            self.assertEqual(
                call["headers"]["Authorization"], "Bearer test-key"
            )
            body = call["json"]
            self.assertEqual(body["model"], "sam-3.1")
            self.assertTrue(body["stream"])
            self.assertEqual(body["metadata"], {"mask_encoding": "one_bit"})
            parts = body["input"][0]["content"]
            texts = [p["text"] for p in parts if p["type"] == "input_text"]
            images = [p["image_url"] for p in parts if p["type"] == "input_image"]
            self.assertEqual(len(texts), 1)
            self.assertEqual(len(images), 1)
            self.assertTrue(images[0].startswith("data:image/png;base64,"))
            seen.add(texts[0])
        self.assertEqual(seen, set(FOCUS_CONCEPTS))

    async def test_single_string_concept(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        result = await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(result["concepts"], ["text"])

    async def test_empty_concepts_rejected(self):
        seg = SamSegmenter(api_key="k", session=FakeSession(FakeResponse()))
        with self.assertRaises(ValueError):
            await seg.segment_image(gradient_png(), concepts=[])

    def test_missing_secret_rejected(self):
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {}
            with self.assertRaises(SamApiError) as ctx:
                SamSegmenter(api_key=None)
        self.assertIn("META_API_KEY", str(ctx.exception))

    def test_empty_secret_rejected(self):
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {"META_API_KEY": ""}
            with self.assertRaises(SamApiError):
                SamSegmenter(api_key=None)

    async def test_key_read_from_streamlit_secrets(self):
        session = FakeSession(FakeResponse(events=success_events()))
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {"META_API_KEY": "secret-from-store"}
            seg = SamSegmenter(api_key=None, session=session)
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"],
            "Bearer secret-from-store",
        )


class MaskParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_masks_decoded_with_raster(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        result = await seg.segment_image(gradient_png(), concepts=["text"])
        self.assertEqual(result["mask_count"], 1)
        mask = result["masks"][0]
        self.assertEqual(mask["concept"], "text")
        self.assertEqual(mask["object_id"], "0")
        # The official parser reports exclusive right/bottom edges.
        self.assertEqual(
            mask["box"], {"left": 1, "top": 2, "right": 11, "bottom": 13}
        )
        self.assertEqual(mask["mask"]["width"], 2)
        self.assertEqual(mask["mask"]["height"], 2)
        self.assertEqual(len(mask["mask"]["raster"]), 4)
        self.assertEqual(mask["mask"]["pixel_count"], 2)
        self.assertEqual(bytes(mask["mask"]["raster"]), bytes([1, 0, 0, 1]))

    def test_latest_revision_wins_per_identity(self):
        def record(identity, revision):
            mock = unittest.mock.create_autospec(
                SegmentationMaskRecord, instance=True
            )
            mock.identity = identity
            mock.revision = revision
            return mock

        old = record("a", 1)
        new = record("a", 3)
        other = record("b", 1)
        chosen = select_latest_masks([old, other, new])
        self.assertEqual(len(chosen), 2)
        self.assertIn(new, chosen)
        self.assertIn(other, chosen)
        self.assertNotIn(old, chosen)


class FidelityTests(unittest.IsolatedAsyncioTestCase):
    async def test_pil_sent_full_resolution_lossless(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        arr = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
        await seg.segment_image(
            Image.fromarray(arr, mode="RGB"), concepts="text"
        )
        body = session.calls[0]["json"]
        url = body["input"][0]["content"][1]["image_url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(sent.size, (32, 24))
        np.testing.assert_array_equal(np.asarray(sent.convert("RGB")), arr)

    async def test_raw_bytes_sent_unmodified(self):
        data = gradient_png()
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        await seg.segment_image(data, concepts="text")
        url = session.calls[0]["json"]["input"][0]["content"][1]["image_url"]
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), data)

    def test_unknown_bytes_rejected(self):
        with self.assertRaises(ValueError):
            sniff_image_mime(b"definitely not an image")


class ErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_error_maps_to_sam_api_error(self):
        session = FakeSession(
            FakeResponse(status=401, body='{"error": {"code": "bad_key"}}')
        )
        seg = SamSegmenter(api_key="bad", session=session)
        with self.assertRaises(SamApiError) as ctx:
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(len(session.calls), 1)


class RetryPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_congestion_statuses_retry_then_succeed(self):
        for status in (429, 503, 504):
            with self.subTest(status=status):
                session = FakeSession(
                    [error_response(status), success_response()]
                )
                seg = SamSegmenter(
                    api_key="k", session=session, retrying=fast_retrying()
                )
                result = await seg.segment_image(
                    gradient_png(), concepts="text"
                )
                self.assertEqual(len(session.calls), 2)
                self.assertEqual(result["mask_count"], 1)

    async def test_exhaustion_reraises_after_three_attempts(self):
        session = FakeSession(error_response(503))
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamRetryableError) as ctx:
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertIn("503", str(ctx.exception))
        self.assertEqual(len(session.calls), 3)

    async def test_client_errors_never_retry(self):
        for status in (400, 401, 404):
            with self.subTest(status=status):
                session = FakeSession(error_response(status))
                seg = SamSegmenter(
                    api_key="k", session=session, retrying=fast_retrying()
                )
                with self.assertRaises(SamApiError) as ctx:
                    await seg.segment_image(gradient_png(), concepts="text")
                self.assertNotIsInstance(ctx.exception, SamRetryableError)
                self.assertEqual(len(session.calls), 1)

    def test_backoff_schedule_is_two_then_four_seconds(self):
        for attempt, expected in ((1, 2), (2, 4)):
            state = RetryCallState(Retrying(), None, (), {})
            state.attempt_number = attempt
            self.assertEqual(SAM_RETRY_WAIT(state), expected)

    def test_retryable_error_is_sam_api_error(self):
        self.assertTrue(issubclass(SamRetryableError, SamApiError))

    def test_retry_status_text_is_exact(self):
        self.assertEqual(RETRY_STATUS_TEXT, "API congested, retrying...")

    async def test_hook_reports_real_backoff_waits(self):
        seen = []
        session = FakeSession(
            [error_response(503), error_response(429), success_response()]
        )
        seg = SamSegmenter(
            api_key="k",
            session=session,
            on_retry=lambda attempt, wait: seen.append((attempt, wait)),
        )
        result = await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(result["mask_count"], 1)
        self.assertEqual(seen, [(1, 2), (2, 4)])


class MaskDisplayTests(unittest.TestCase):
    def _mask(self, concept="text", box="default", raster=b"\x01\x00\x00\x01"):
        if box == "default":
            box = {"left": 2, "top": 3, "right": 8, "bottom": 9}
        return {
            "concept": concept,
            "object_id": "0",
            "confidence": 0.9,
            "box": box,
            "mask": {"width": 2, "height": 2, "pixel_count": 2, "raster": raster},
        }

    def test_summarize_strips_raster_keeps_fields(self):
        summary = summarize_masks({"masks": [self._mask()]})
        self.assertEqual(len(summary), 1)
        entry = summary[0]
        self.assertNotIn("raster", entry)
        self.assertEqual(entry["concept"], "text")
        self.assertEqual(entry["pixel_count"], 2)
        self.assertEqual(entry["box"]["left"], 2)
        json.dumps(summary)

    def test_summarize_empty_result(self):
        self.assertEqual(summarize_masks(None), [])
        self.assertEqual(summarize_masks({}), [])

    def test_draw_boxes_marks_concept_color(self):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        out = draw_mask_boxes(img, [self._mask()])
        self.assertEqual(out.size, (20, 20))
        self.assertEqual(out.getpixel((2, 3)), (255, 80, 80))
        self.assertEqual(img.getpixel((2, 3)), (255, 255, 255))

    def test_draw_boxes_skips_missing_box(self):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        out = draw_mask_boxes(img, [self._mask(box=None)])
        self.assertEqual(list(out.getdata()), list(img.getdata()))

    def test_draw_boxes_unknown_concept_uses_default(self):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        out = draw_mask_boxes(img, [self._mask(concept="car")])
        self.assertEqual(out.getpixel((2, 3)), (200, 200, 200))

    async def test_failed_stream_maps_to_sam_api_error(self):
        session = FakeSession(
            FakeResponse(events=[{"type": "response.failed"}])
        )
        seg = SamSegmenter(api_key="k", session=session)
        with self.assertRaises(SamApiError):
            await seg.segment_image(gradient_png(), concepts="text")


if __name__ == "__main__":
    unittest.main()
