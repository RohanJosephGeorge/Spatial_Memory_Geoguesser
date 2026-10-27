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

from sam_segmentation import (
    FOCUS_CONCEPTS,
    SamApiError,
    SamSegmenter,
    select_latest_masks,
    sniff_image_mime,
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
        self._response = response

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._response


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
        self.assertEqual(len(session.calls), 3)
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

    async def test_failed_stream_maps_to_sam_api_error(self):
        session = FakeSession(
            FakeResponse(events=[{"type": "response.failed"}])
        )
        seg = SamSegmenter(api_key="k", session=session)
        with self.assertRaises(SamApiError):
            await seg.segment_image(gradient_png(), concepts="text")


if __name__ == "__main__":
    unittest.main()
