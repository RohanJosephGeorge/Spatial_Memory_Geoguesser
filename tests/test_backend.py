"""Tests for the FastAPI backend's privacy rounding and payload mapping.

Run: python -m unittest discover -s tests -v
Requires Pillow only: the rounding, payload, and decode helpers are
pure and importable without fastapi/uvicorn installed.
"""

import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from backend import (
    build_frontend_result,
    decode_upload,
    round_wonder_coordinates,
)


def wonder_payload(latitude=48.8584, longitude=2.2945):
    return {
        "latitude": latitude,
        "longitude": longitude,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower", "French signage"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }


def halt_payload():
    return {
        "latitude": None,
        "longitude": None,
        "city": "",
        "country": "",
        "reasoning_steps": ["Not enough distinct features"],
        "confidence_score": 0.0,
        "precision_level": "Unknown",
    }


class RoundWonderCoordinatesTests(unittest.TestCase):
    def test_rounds_to_three_decimals(self):
        rounded = round_wonder_coordinates(wonder_payload())
        self.assertEqual(rounded["latitude"], 48.858)
        self.assertEqual(rounded["longitude"], 2.295)
        self.assertIsInstance(rounded["latitude"], float)
        self.assertIsInstance(rounded["longitude"], float)

    def test_rounds_negative_coordinates(self):
        rounded = round_wonder_coordinates(wonder_payload(-33.8568, -151.2153))
        self.assertEqual(rounded["latitude"], -33.857)
        self.assertEqual(rounded["longitude"], -151.215)

    def test_short_and_whole_values_become_floats(self):
        rounded = round_wonder_coordinates(wonder_payload(48.8, 48))
        self.assertEqual(rounded["latitude"], 48.8)
        self.assertEqual(rounded["longitude"], 48.0)
        self.assertIsInstance(rounded["longitude"], float)

    def test_tiny_values_round_to_zero_float(self):
        rounded = round_wonder_coordinates(wonder_payload(0.00049, -0.00049))
        self.assertEqual(rounded["latitude"], 0.0)
        self.assertEqual(rounded["longitude"], -0.0)
        self.assertIsInstance(rounded["latitude"], float)

    def test_null_halt_coordinates_pass_through(self):
        self.assertEqual(round_wonder_coordinates(halt_payload()), halt_payload())

    def test_custom_decimals_honored(self):
        rounded = round_wonder_coordinates(wonder_payload(), decimals=2)
        self.assertEqual(rounded["latitude"], 48.86)
        self.assertEqual(rounded["longitude"], 2.29)

    def test_input_never_mutated_and_other_keys_kept(self):
        original = wonder_payload()
        snapshot = dict(original)
        rounded = round_wonder_coordinates(original)
        self.assertEqual(original, snapshot)
        self.assertIsNot(rounded, original)
        for key in (
            "city",
            "country",
            "reasoning_steps",
            "confidence_score",
            "precision_level",
        ):
            self.assertEqual(rounded[key], snapshot[key])

    def test_missing_coordinate_key_fails_closed(self):
        for key in ("latitude", "longitude"):
            with self.subTest(key=key):
                payload = wonder_payload()
                del payload[key]
                with self.assertRaises(ValueError) as ctx:
                    round_wonder_coordinates(payload)
                self.assertIn(key, str(ctx.exception))

    def test_non_numeric_coordinates_fail_closed(self):
        for bad in (True, "48.85", [48.85], {"lat": 48.85}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    round_wonder_coordinates(wonder_payload(bad, 2.29))
                with self.assertRaises(ValueError):
                    round_wonder_coordinates(wonder_payload(48.85, bad))


class BuildFrontendResultTests(unittest.TestCase):
    def test_maps_rounded_wonder_result(self):
        result = build_frontend_result(
            round_wonder_coordinates(wonder_payload()),
            "memory.jpg",
            "riverside cafes",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "backend")
        self.assertEqual(result["filename"], "memory.jpg")
        self.assertEqual(result["regional_hint"], "riverside cafes")
        self.assertEqual(result["city"], "Paris")
        self.assertEqual(result["country"], "France")
        self.assertEqual(result["precision"], "Public Landmark")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(
            result["coordinates"], {"latitude": 48.858, "longitude": 2.295}
        )
        self.assertEqual(
            result["reasoning_steps"],
            ["Iron lattice tower", "French signage"],
        )

    def test_blank_hint_gets_default(self):
        result = build_frontend_result(
            round_wonder_coordinates(wonder_payload()), "m.jpg", "   "
        )
        self.assertEqual(result["regional_hint"], "(no hint provided)")

    def test_halt_coordinates_stay_null(self):
        result = build_frontend_result(halt_payload(), "m.jpg", "")
        self.assertEqual(
            result["coordinates"], {"latitude": None, "longitude": None}
        )


class DecodeUploadTests(unittest.TestCase):
    def test_valid_jpeg_base64_round_trip(self):
        img = Image.new("RGB", (16, 12), (10, 200, 90))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        decoded = decode_upload(
            base64.b64encode(buf.getvalue()).decode("ascii")
        )
        self.assertEqual(decoded.size, (16, 12))
        self.assertEqual(decoded.mode, "RGB")

    def test_invalid_base64_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload("!!! not base64 !!!")

    def test_empty_payload_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload("")

    def test_oversize_payload_rejected(self):
        with patch("backend.MAX_UPLOAD_BYTES", 10):
            with self.assertRaises(ValueError) as ctx:
                decode_upload(base64.b64encode(b"x" * 11).decode("ascii"))
        self.assertIn("10 MB", str(ctx.exception))

    def test_non_image_bytes_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload(base64.b64encode(b"x" * 64).decode("ascii"))


if __name__ == "__main__":
    unittest.main()
