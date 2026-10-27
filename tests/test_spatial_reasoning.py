"""Tests for the Muse Wonder spatial reasoning prompt and client.

Run: python -m unittest discover -s tests -v
Requires aiohttp and tenacity. No network: HTTP is fully mocked.
"""

import json
import unittest
from unittest.mock import patch

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_none,
)

from sam_segmentation import SamRetryableError
from spatial_reasoning import (
    REQUIRED_KEYS,
    LocationUnknownError,
    SpatialReasoner,
    SpatialReasoningError,
    build_spatial_prompt,
    check_null_island,
    estimate_tokens,
    extract_output_text,
    parse_reasoning_result,
)


def halt_payload():
    return {
        "latitude": None,
        "longitude": None,
        "city": "",
        "country": "",
        "reasoning_steps": ["Synthetic image detected"],
        "confidence_score": 0.0,
        "precision_level": "Unknown",
    }


def sample_summary():
    return [
        {
            "concept": "text",
            "object_id": "0",
            "confidence": 0.92,
            "box": {"left": 10, "top": 20, "right": 200, "bottom": 60},
            "width": 190,
            "height": 40,
            "pixel_count": 5123,
        },
        {
            "concept": "vegetation",
            "object_id": "1",
            "confidence": 0.87,
            "box": {"left": 0, "top": 300, "right": 640, "bottom": 480},
            "width": 640,
            "height": 180,
            "pixel_count": 88120,
        },
    ]


def good_payload():
    return {
        "latitude": 48.8584,
        "longitude": 2.2945,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower", "French signage"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }


def responses_json(text):
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    }


class FakeJsonResponse:
    def __init__(self, status=200, payload=None, body=""):
        self.status = status
        self._payload = payload
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    async def text(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.calls = []
        self._responses = (
            responses if isinstance(responses, list) else [responses]
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


class PromptTests(unittest.TestCase):
    def test_prompt_carries_masks_hint_keys_and_limits(self):
        prompt = build_spatial_prompt(sample_summary(), "riverside cafes")
        self.assertIn("riverside cafes", prompt)
        self.assertIn("architectural", prompt.lower())
        self.assertIn('"concept": "text"', prompt)
        self.assertIn('"pixel_count": 5123', prompt)
        for key in REQUIRED_KEYS:
            self.assertIn(key, prompt)
        self.assertIn("250", prompt)
        self.assertIn("JSON", prompt)
        self.assertIn("multi", prompt.lower())

    def test_prompt_accepts_full_result_dict(self):
        prompt = build_spatial_prompt(
            {"masks": [{"concept": "text", "mask": {"raster": b"\x01"}}]},
            "",
        )
        self.assertIn("(no hint provided)", prompt)
        self.assertIn('"concept": "text"', prompt)
        self.assertNotIn("raster", prompt)

    def test_custom_rules_replace_defaults(self):
        prompt = build_spatial_prompt([], "x", strict_rules=["Only reply {}."])
        self.assertIn("Only reply {}.", prompt)
        self.assertNotIn("markdown fences", prompt)

    def test_estimate_tokens_counts_words(self):
        self.assertEqual(estimate_tokens("one two  three"), 3)
        self.assertEqual(estimate_tokens(""), 0)

    def test_prompt_contains_all_strict_rules(self):
        prompt = build_spatial_prompt([], "x")
        for phrase in (
            "Privacy Override",
            "Synthetic Check",
            "Synthetic image detected",
            "Indoor/Featureless Check",
            "Not enough distinct features",
            "Text Override",
            "Privacy Radius",
            "Public Landmark",
            "Street Level",
            "Neighborhood Radius",
        ):
            self.assertIn(phrase, prompt)


class ParseTests(unittest.TestCase):
    def test_valid_payload_passes_through(self):
        self.assertEqual(parse_reasoning_result(json.dumps(good_payload())),
                         good_payload())

    def test_surrounding_text_tolerated(self):
        text = "Here you go:\n" + json.dumps(good_payload()) + "\nDone."
        self.assertEqual(parse_reasoning_result(text)["city"], "Paris")

    def test_garbage_rejected(self):
        with self.assertRaises(SpatialReasoningError):
            parse_reasoning_result("not json at all")

    def test_missing_key_rejected(self):
        payload = good_payload()
        del payload["city"]
        with self.assertRaises(SpatialReasoningError) as ctx:
            parse_reasoning_result(json.dumps(payload))
        self.assertIn("city", str(ctx.exception))

    def test_extra_key_rejected(self):
        payload = good_payload()
        payload["bonus"] = 1
        with self.assertRaises(SpatialReasoningError) as ctx:
            parse_reasoning_result(json.dumps(payload))
        self.assertIn("bonus", str(ctx.exception))

    def test_bad_ranges_rejected(self):
        for key, value in (("latitude", 91), ("longitude", -181),
                           ("confidence_score", 1.5)):
            with self.subTest(key=key):
                payload = good_payload()
                payload[key] = value
                with self.assertRaises(SpatialReasoningError):
                    parse_reasoning_result(json.dumps(payload))

    def test_empty_steps_rejected(self):
        payload = good_payload()
        payload["reasoning_steps"] = []
        with self.assertRaises(SpatialReasoningError):
            parse_reasoning_result(json.dumps(payload))

    def test_null_halt_accepted(self):
        self.assertEqual(
            parse_reasoning_result(json.dumps(halt_payload())), halt_payload()
        )

    def test_mixed_null_coordinates_rejected(self):
        payload = halt_payload()
        payload["latitude"] = 48.8
        with self.assertRaises(SpatialReasoningError):
            parse_reasoning_result(json.dumps(payload))

    def test_broken_halt_convention_rejected(self):
        cases = [
            ("city", "Paris"),
            ("country", "France"),
            ("confidence_score", 0.5),
            ("precision_level", "Street Level"),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                payload = halt_payload()
                payload[key] = value
                with self.assertRaises(SpatialReasoningError):
                    parse_reasoning_result(json.dumps(payload))

    def test_precision_levels_enforced(self):
        for level in ("Public Landmark", "Street Level", "Neighborhood Radius"):
            with self.subTest(level=level):
                payload = good_payload()
                payload["precision_level"] = level
                self.assertEqual(
                    parse_reasoning_result(json.dumps(payload))[
                        "precision_level"
                    ],
                    level,
                )
        payload = good_payload()
        payload["precision_level"] = "Country"
        with self.assertRaises(SpatialReasoningError):
            parse_reasoning_result(json.dumps(payload))

    def test_null_island_intercepted(self):
        self.assertTrue(issubclass(LocationUnknownError, SpatialReasoningError))
        with self.assertRaises(LocationUnknownError) as ctx:
            check_null_island({"latitude": 0.0, "longitude": 0.0})
        self.assertIn("Location Unknown", str(ctx.exception))
        self.assertEqual(
            check_null_island({"latitude": 48.8, "longitude": 2.3}),
            {"latitude": 48.8, "longitude": 2.3},
        )
        self.assertEqual(
            check_null_island({"latitude": None, "longitude": None}),
            {"latitude": None, "longitude": None},
        )

    def test_extract_output_text(self):
        data = responses_json(json.dumps(good_payload()))
        self.assertEqual(json.loads(extract_output_text(data))["city"], "Paris")

    def test_extract_output_text_missing(self):
        with self.assertRaises(SpatialReasoningError):
            extract_output_text({"output": []})


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_reason_returns_validated_dict(self):
        session = FakeSession(
            FakeJsonResponse(payload=responses_json(json.dumps(good_payload())))
        )
        reasoner = SpatialReasoner(api_key="k", session=session)
        result = await reasoner.reason(sample_summary(), "riverside cafes")
        self.assertEqual(result["city"], "Paris")
        body = session.calls[0]["json"]
        self.assertEqual(body["model"], "muse-wonder-1.0")
        self.assertFalse(body["stream"])
        sent = body["input"][0]["content"][0]["text"]
        self.assertIn("riverside cafes", sent)
        self.assertIn("reasoning_steps", sent)

    async def test_model_override_honored(self):
        session = FakeSession(
            FakeJsonResponse(payload=responses_json(json.dumps(good_payload())))
        )
        reasoner = SpatialReasoner(
            api_key="k", session=session, model="meta/muse-wonder"
        )
        await reasoner.reason([], "")
        self.assertEqual(session.calls[0]["json"]["model"], "meta/muse-wonder")

    async def test_retryable_status_retries_then_succeeds(self):
        session = FakeSession(
            [
                FakeJsonResponse(status=503, body="overloaded"),
                FakeJsonResponse(
                    payload=responses_json(json.dumps(good_payload()))
                ),
            ]
        )
        reasoner = SpatialReasoner(
            api_key="k", session=session, retrying=fast_retrying()
        )
        result = await reasoner.reason([], "")
        self.assertEqual(result["country"], "France")
        self.assertEqual(len(session.calls), 2)

    async def test_client_error_never_retries(self):
        session = FakeSession(FakeJsonResponse(status=404, body="no model"))
        reasoner = SpatialReasoner(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SpatialReasoningError):
            await reasoner.reason([], "")
        self.assertEqual(len(session.calls), 1)

    async def test_invalid_model_json_maps_to_error(self):
        session = FakeSession(
            FakeJsonResponse(payload=responses_json("oops"))
        )
        reasoner = SpatialReasoner(api_key="k", session=session)
        with self.assertRaises(SpatialReasoningError):
            await reasoner.reason([], "")

    async def test_structured_json_schema_requested(self):
        session = FakeSession(
            FakeJsonResponse(payload=responses_json(json.dumps(good_payload())))
        )
        reasoner = SpatialReasoner(api_key="k", session=session)
        await reasoner.reason([], "")
        fmt = session.calls[0]["json"]["text"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["schema"]["type"], "object")
        self.assertEqual(
            sorted(fmt["schema"]["required"]), sorted(REQUIRED_KEYS)
        )
        self.assertFalse(fmt["schema"]["additionalProperties"])

    async def test_null_island_end_to_end(self):
        payload = good_payload()
        payload["latitude"] = 0.0
        payload["longitude"] = 0.0
        payload["precision_level"] = "Neighborhood Radius"
        session = FakeSession(
            FakeJsonResponse(payload=responses_json(json.dumps(payload)))
        )
        reasoner = SpatialReasoner(api_key="k", session=session)
        with self.assertRaises(LocationUnknownError):
            await reasoner.reason([], "")

    async def test_halt_result_end_to_end(self):
        session = FakeSession(
            FakeJsonResponse(payload=responses_json(json.dumps(halt_payload())))
        )
        reasoner = SpatialReasoner(api_key="k", session=session)
        result = await reasoner.reason([], "")
        self.assertIsNone(result["latitude"])
        self.assertEqual(result["precision_level"], "Unknown")

    def test_missing_secret_rejected(self):
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {}
            with self.assertRaises(Exception) as ctx:
                SpatialReasoner(api_key=None)
        self.assertIn("META_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
