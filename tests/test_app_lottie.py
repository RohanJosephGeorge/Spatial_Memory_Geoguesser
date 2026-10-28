"""Tests for the Analyze Memory Lottie loading animation.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas (no browser, no network: HTTP is mocked).
Each test uses a distinct URL because load_lottie_animation() caches
per URL.
"""

import json
import unittest
import urllib.error
from unittest.mock import patch

import app
from app import LOTTIE_RADAR_URL, load_lottie_animation


def lottie_bytes():
    return json.dumps(
        {
            "v": "5.5.9",
            "fr": 60,
            "ip": 0,
            "op": 241,
            "w": 766,
            "h": 576,
            "layers": [],
        }
    ).encode("utf-8")


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._payload


class LoadLottieAnimationTests(unittest.TestCase):
    def test_valid_lottie_json_returned_as_dict(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(lottie_bytes()),
        ) as mock_open:
            animation = load_lottie_animation("https://example.com/a.json")
        self.assertIsInstance(animation, dict)
        self.assertEqual(animation["v"], "5.5.9")
        self.assertEqual(animation["fr"], 60)
        mock_open.assert_called_once_with(
            "https://example.com/a.json", timeout=app.LOTTIE_FETCH_TIMEOUT
        )

    def test_network_failure_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/b.json")
            )

    def test_invalid_json_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(b"<html>not json</html>"),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/c.json")
            )

    def test_non_object_json_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(b"[1, 2, 3]"),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/d.json")
            )

    def test_default_url_points_at_radar_json(self):
        self.assertTrue(LOTTIE_RADAR_URL.startswith("https://"))
        self.assertTrue(LOTTIE_RADAR_URL.endswith("lottie_radar.json"))

    def test_lottie_import_degrades_gracefully(self):
        # Guarded import: st_lottie is either the component or None,
        # never an ImportError at module load.
        self.assertTrue(
            app.st_lottie is None or callable(app.st_lottie)
        )


if __name__ == "__main__":
    unittest.main()
