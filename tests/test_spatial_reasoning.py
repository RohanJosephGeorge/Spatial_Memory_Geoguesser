"""Tests for the Muse Wonder spatial reasoning prompt and client.

Run: python -m unittest discover -s tests -v
Requires aiohttp and tenacity. No network: HTTP is fully mocked.
"""

import io
import json
import unittest
from unittest.mock import patch

from PIL import Image
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_none,
)

from backend import build_frontend_result, round_wonder_coordinates
from sam_segmentation import SamRetryableError
from spatial_reasoning import (
    ARCHITECTURAL_TAXONOMY,
    COASTAL_INDICATORS,
    CULTURAL_TAXONOMY,
    ENVIRONMENTAL_INFERENCE,
    INFRASTRUCTURE_TAXONOMY,
    PRECISION_LEVELS,
    REQUIRED_KEYS,
    REASONING_EFFORT,
    TEXT_ANCHOR_INSTRUCTION,
    TRANSIT_TAXONOMY,
    TRIANGULATION_HEURISTICS,
    LocationUnknownError,
    SpatialReasoner,
    SpatialReasoningError,
    build_reasoning_payload,
    build_spatial_prompt,
    check_null_island,
    encode_image_data_url,
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
            "Exhaustive Visual Synthesis",
            "Best-Effort Coordinates",
            "Graduated Precision Scale",
            "Confidence Calibration",
            "Public Landmark",
            "Street Level",
            "Neighborhood Radius",
            "City Level",
            "State / Region Radius",
        ):
            self.assertIn(phrase, prompt)


class AggressivePromptTests(unittest.TestCase):
    def test_elite_expert_identity(self):
        prompt = build_spatial_prompt([], "x")
        self.assertIn("reverse geolocation expert", prompt)
        self.assertIn("STRICT DEDUCTION RULES", prompt)

    def test_no_easy_quits_and_narrowed_halt(self):
        prompt = build_spatial_prompt([], "x")
        self.assertIn("pitch black", prompt)
        self.assertIn("Synthetic image detected", prompt)
        self.assertNotIn("Not enough distinct features", prompt)
        self.assertNotIn("DO NOT guess", prompt)

    def test_graduated_scale_has_five_levels(self):
        self.assertEqual(
            PRECISION_LEVELS,
            (
                "Public Landmark",
                "Street Level",
                "Neighborhood Radius",
                "City Level",
                "State / Region Radius",
            ),
        )


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

    def test_coarse_levels_with_empty_city_keep_coordinates(self):
        for level in ("City Level", "State / Region Radius"):
            with self.subTest(level=level):
                payload = good_payload()
                payload["city"] = ""
                payload["precision_level"] = level
                payload["confidence_score"] = 0.35
                parsed = parse_reasoning_result(json.dumps(payload))
                self.assertEqual(parsed["latitude"], 48.8584)
                self.assertEqual(parsed["longitude"], 2.2945)
                self.assertEqual(parsed["city"], "")
                self.assertEqual(parsed["precision_level"], level)

    def test_empty_country_with_valid_coordinates_passes(self):
        payload = good_payload()
        payload["country"] = ""
        payload["precision_level"] = "State / Region Radius"
        parsed = parse_reasoning_result(json.dumps(payload))
        self.assertEqual(parsed["longitude"], 2.2945)

    def test_non_string_city_still_rejected(self):
        payload = good_payload()
        payload["city"] = None
        with self.assertRaises(SpatialReasoningError):
            parse_reasoning_result(json.dumps(payload))

    def test_parsed_coarse_result_flows_to_frontend_shape(self):
        payload = good_payload()
        payload["city"] = ""
        payload["precision_level"] = "State / Region Radius"
        payload["confidence_score"] = 0.4
        parsed = parse_reasoning_result(json.dumps(payload))
        result = build_frontend_result(
            round_wonder_coordinates(parsed), "m.jpg", ""
        )
        self.assertEqual(
            result["coordinates"], {"latitude": 48.858, "longitude": 2.295}
        )
        self.assertEqual(result["city"], "")
        self.assertEqual(result["precision"], "State / Region Radius")

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
        for level in (
            "Public Landmark",
            "Street Level",
            "Neighborhood Radius",
            "City Level",
            "State / Region Radius",
        ):
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


def tiny_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 200, 90)).save(buf, format="PNG")
    return buf.getvalue()


class DualInputPayloadTests(unittest.TestCase):
    def test_reasoning_effort_is_max(self):
        self.assertEqual(REASONING_EFFORT, "max")
        payload = build_reasoning_payload("prompt")
        self.assertEqual(payload["reasoning"], {"effort": "max"})

    def test_text_only_payload_keeps_single_text_part(self):
        payload = build_reasoning_payload("prompt", model="custom-model")
        self.assertEqual(payload["model"], "custom-model")
        self.assertFalse(payload["stream"])
        content = payload["input"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[0]["text"], "prompt")

    def test_image_payload_pairs_text_with_raw_image(self):
        url = "data:image/png;base64,AAA"
        payload = build_reasoning_payload("prompt", image_data_url=url)
        content = payload["input"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], url)
        self.assertEqual(payload["reasoning"], {"effort": "max"})

    def test_encode_pil_image_to_png_data_url(self):
        url = encode_image_data_url(Image.new("RGB", (4, 3), (1, 2, 3)))
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_encode_png_bytes_to_data_url(self):
        url = encode_image_data_url(tiny_png_bytes())
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_encode_rejects_garbage_and_wrong_types(self):
        with self.assertRaises(SpatialReasoningError):
            encode_image_data_url(b"not an image")
        with self.assertRaises(SpatialReasoningError):
            encode_image_data_url("data:image/png;base64,AAA")

    def test_empty_masks_with_image_falls_back_to_raw_image(self):
        prompt = build_spatial_prompt([], "hint", has_image=True)
        self.assertIn("0 masks", prompt)
        self.assertIn("attached raw image directly", prompt)
        self.assertIn("REQUIRED OUTPUT", prompt)

    def test_minimal_masks_with_image_falls_back_to_raw_image(self):
        prompt = build_spatial_prompt(sample_summary(), "hint", has_image=True)
        self.assertIn("attached raw image directly", prompt)

    def test_rich_inventory_with_image_notes_attachment(self):
        masks = sample_summary() + [dict(sample_summary()[0], object_id="2")]
        prompt = build_spatial_prompt(masks, "hint", has_image=True)
        self.assertNotIn("attached raw image directly", prompt)
        self.assertIn("attached alongside", prompt)

    def test_no_image_mentioned_without_attached_image(self):
        prompt = build_spatial_prompt([], "hint")
        self.assertNotIn("attached raw image", prompt)
        self.assertNotIn("attached alongside", prompt)


class OcrPromptTests(unittest.TestCase):
    def test_anchor_instruction_is_verbatim(self):
        self.assertEqual(
            TEXT_ANCHOR_INSTRUCTION,
            "If legible text, shop names, or district names are detected, "
            "prioritize them as Tier-1 primary anchor points. Use the text "
            "to narrow down the specific district, town, or transit "
            "corridor before falling back to landscape cues.",
        )

    def test_image_prompt_directs_text_and_code_decoding(self):
        prompt = build_spatial_prompt([], "hint", has_image=True)
        for phrase in (
            "TEXT DECODING",
            "shop headers",
            "highway numbers",
            "license plate",
            "KL",
            "Kerala",
            "MH",
            "Maharashtra",
            ".in",
            "area codes",
            "Malayalam",
            "Tamil",
            "Hindi",
            "Cyrillic",
            "Arabic",
            "Hangul",
            "language",
            "geographic region",
            "Tier-1 primary anchor points",
            "transit corridor",
        ):
            self.assertIn(phrase, prompt)

    def test_no_ocr_section_without_attached_image(self):
        prompt = build_spatial_prompt([], "hint")
        self.assertNotIn("TEXT DECODING", prompt)
        self.assertNotIn("Tier-1", prompt)


class MultiHypothesisPromptTests(unittest.TestCase):
    def test_protocol_generates_three_candidates(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "3 distinct candidate",
            "Candidate A",
            "evidence matrix",
            "eliminate",
            "confidence weighting",
            "geographic center",
        ):
            self.assertIn(phrase, prompt)

    def test_reasoning_steps_spec_requires_hypothesis_trace(self):
        prompt = build_spatial_prompt([], "hint")
        self.assertIn("candidates", prompt)
        self.assertIn("elimination", prompt)

    def test_handler_passes_multi_candidate_response_through(self):
        response = {
            "latitude": 9.4981,
            "longitude": 76.3388,
            "city": "Alappuzha",
            "country": "India",
            "reasoning_steps": [
                "Candidate A: Alappuzha, Kerala; Candidate B: Mangalore, "
                "Karnataka; Candidate C: Goa.",
                "Evidence matrix: laterite soil and coconut palms fit A; "
                "driving side eliminates C.",
                "Winner: Candidate A with 0.62 confidence weighting.",
                "Centroid fixed on the Alappuzha district center.",
            ],
            "confidence_score": 0.62,
            "precision_level": "City Level",
        }
        parsed = parse_reasoning_result(json.dumps(response))
        self.assertEqual(parsed["latitude"], 9.4981)
        self.assertEqual(parsed["longitude"], 76.3388)
        self.assertEqual(parsed["city"], "Alappuzha")
        self.assertEqual(len(parsed["reasoning_steps"]), 4)


class CoastalPromptTests(unittest.TestCase):
    def test_four_indicators_defined(self):
        self.assertEqual(len(COASTAL_INDICATORS), 4)

    def test_sand_and_soil_indicators(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "COASTAL",
            "MARINE",
            "white sand",
            "golden",
            "volcanic",
            "Kollam",
            "Varkala",
            "estuarine",
        ):
            self.assertIn(phrase, prompt)

    def test_coastal_topography_indicators(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "laterite red cliffs",
            "tetrapods",
            "breakwaters",
            "spinifex",
            "casuarina",
        ):
            self.assertIn(phrase, prompt)

    def test_marine_craft_and_water_optics_indicators(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "catamarans",
            "Chinese fishing nets",
            "trawlers",
            "painting schemes",
            "water clarity",
            "wave break",
            "tidal mudflats",
            "backwater",
            "mangrove",
        ):
            self.assertIn(phrase, prompt)


class TransitTaxonomyPromptTests(unittest.TestCase):
    def test_two_taxonomy_entries_defined(self):
        self.assertEqual(len(TRANSIT_TAXONOMY), 2)

    def test_grid_mapping_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "TRANSIT",
            "GRID",
            "RCC",
            "spun concrete",
            "H-beams",
            "tubular",
            "lattice",
            "insulator",
            "cobra-head",
            "sodium vapor",
            "clutter",
        ):
            self.assertIn(phrase, prompt)

    def test_roadway_delineation_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "Left-Hand Drive",
            "Right-Hand Drive",
            "parked vehicle",
            "dashed white",
            "double yellow",
            "culverts",
            "yellow cap",
            "reflector bollard",
        ):
            self.assertIn(phrase, prompt)


class CulturalPromptTests(unittest.TestCase):
    def test_three_taxonomy_entries_defined(self):
        self.assertEqual(len(CULTURAL_TAXONOMY), 3)

    def test_vernacular_architecture_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "CULTURAL",
            "Mangalore",
            "slate tiles",
            "laundry lines",
            "Sintex",
            "stainless steel",
            "satellite dish",
            "piercing grills",
            "broken-glass",
        ):
            self.assertIn(phrase, prompt)

    def test_linguistic_commercial_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "Devanagari",
            "Gujarati",
            "Thai",
            "bilingual",
            "Airtel",
            "Jio",
            "shutter styles",
        ):
            self.assertIn(phrase, prompt)

    def test_vehicle_transport_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "rickshaw",
            "liveries",
            "aspect ratios",
            "green EV",
            "red temp",
            "registration prefixes",
        ):
            self.assertIn(phrase, prompt)


class EnvironmentalPromptTests(unittest.TestCase):
    def test_three_inference_rules_defined(self):
        self.assertEqual(len(ENVIRONMENTAL_INFERENCE), 3)

    def test_solar_geometry_inference(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "ENVIRONMENTAL",
            "ASTRONOMICAL",
            "shadow direction",
            "sun angle",
            "hemisphere",
            "latitude bands",
            "smog",
            "cumulus",
            "golden hour",
        ):
            self.assertIn(phrase, prompt)

    def test_soil_and_geology_inference(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "laterite",
            "chernozem",
            "Deccan",
            "basalt",
            "karst",
            "mesas",
            "deltas",
            "terraced",
        ):
            self.assertIn(phrase, prompt)

    def test_botanical_bio_marker_inference(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "areca",
            "acacia",
            "eucalyptus",
            "silver oak",
            "sugarcane",
            "wheat fields",
            "scrubland",
        ):
            self.assertIn(phrase, prompt)


class ArchitecturalPromptTests(unittest.TestCase):
    def test_two_taxonomy_entries_defined(self):
        self.assertEqual(len(ARCHITECTURAL_TAXONOMY), 2)

    def test_roof_and_wall_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "ARCHITECTURAL CLASSIFICATION",
            "Mangalore",
            "corrugated",
            "drying lines",
            "barrel tiles",
            "laterite stone",
            "whitewashed",
            "teal",
        ):
            self.assertIn(phrase, prompt)

    def test_domestic_and_commercial_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "Thinnai",
            "carved pillars",
            "lintels",
            "security grills",
            "roller shutters",
            "awnings",
            "Sintex",
            "solar water heater",
        ):
            self.assertIn(phrase, prompt)


class InfrastructureTaxonomyPromptTests(unittest.TestCase):
    def test_two_taxonomy_entries_defined(self):
        self.assertEqual(len(INFRASTRUCTURE_TAXONOMY), 2)

    def test_electrical_distribution_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "MICRO-INFRASTRUCTURE",
            "reinforced concrete",
            "spun concrete",
            "cross-arm",
            "tubular",
            "lattice",
            "insulator",
            "single-phase",
            "three-phase",
        ):
            self.assertIn(phrase, prompt)

    def test_roadway_boundary_taxonomy(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "High-Density Urban",
            "Peri-Urban",
            "Deep Rural",
            "red dirt shoulders",
            "drainage canals",
            "yellow/white",
            "green/white",
            "black/white",
            "guardrails",
            "boundary pillars",
        ):
            self.assertIn(phrase, prompt)


class TriangulationPromptTests(unittest.TestCase):
    def test_three_heuristics_defined(self):
        self.assertEqual(len(TRIANGULATION_HEURISTICS), 3)

    def test_solar_geometry_heuristics(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "STRUCTURAL TRIANGULATION",
            "Solar Geometry",
            "shadow angle",
            "light direction",
            "hemisphere",
            "latitude",
        ):
            self.assertIn(phrase, prompt)

    def test_infrastructure_fingerprinting_heuristics(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "Infrastructure Fingerprinting",
            "RCC",
            "lattice",
            "driving side",
            "curb",
            "asphalt",
        ):
            self.assertIn(phrase, prompt)

    def test_vegetation_biome_indexing_heuristics(self):
        prompt = build_spatial_prompt([], "hint")
        for phrase in (
            "Vegetation",
            "Biome",
            "rubber plantation",
            "coconut palm",
            "pine",
            "laterite",
            "climatic sub-zones",
        ):
            self.assertIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
