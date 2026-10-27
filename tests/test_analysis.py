"""Tests for analysis backends: mock payload contents, graceful
backend failure, and the full execute_analysis pipeline.

Run: python -m unittest discover -s tests -v
Requires numpy and Pillow (no OpenCV, no Streamlit, no network: the
failure test hits a closed loopback port, pipeline tests stub the
face screener).
"""

import io
import unittest

import numpy as np
from PIL import Image

from analysis import (
    AnalysisError,
    analyze_via_backend,
    execute_analysis,
    mock_analysis_result,
)


class FakeUpload(io.BytesIO):
    def __init__(self, data, name="photo.png", mime="image/png"):
        super().__init__(data)
        self.name = name
        self.type = mime
        self.size = len(data)


def png_upload(width=64, height=48, name="photo.png"):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    y = np.tile(np.arange(height, dtype=np.uint8)[:, None], (1, width))
    img = Image.fromarray(np.stack([x, y, x], axis=-1), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return FakeUpload(buf.getvalue(), name=name)


def clean_screener(image):
    return ("clean", "No faces detected.", image)


class MockResultTests(unittest.TestCase):
    def test_mock_payload_contents(self):
        result = mock_analysis_result("photo.jpg", "  ")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "mock")
        self.assertEqual(result["precision"], "Public Landmark")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(
            result["coordinates"], {"latitude": 48.8584, "longitude": 2.2945}
        )
        self.assertIsInstance(result["reasoning_steps"], list)
        self.assertGreater(len(result["reasoning_steps"]), 0)
        self.assertTrue(
            all(isinstance(step, str) for step in result["reasoning_steps"])
        )

    def test_mock_echoes_filename_and_hint(self):
        result = mock_analysis_result("memory.png", "alpine village")
        self.assertEqual(result["filename"], "memory.png")
        self.assertEqual(result["regional_hint"], "alpine village")
        self.assertTrue(
            any("alpine village" in step for step in result["reasoning_steps"])
        )

    def test_mock_has_no_network_side_effects(self):
        first = mock_analysis_result("a.jpg", "x")
        second = mock_analysis_result("a.jpg", "x")
        self.assertEqual(first, second)
        self.assertIsNot(first["coordinates"], second["coordinates"])


class BackendCallTests(unittest.TestCase):
    def test_unreachable_backend_raises_analysis_error(self):
        with self.assertRaises(AnalysisError) as ctx:
            analyze_via_backend(
                b"fake-jpeg",
                "a.jpg",
                "hint",
                backend_url="http://127.0.0.1:1",
                timeout=2.0,
            )
        self.assertIn("127.0.0.1:1", str(ctx.exception))


class ExecuteAnalysisTests(unittest.TestCase):
    def test_mock_mode_returns_ok_with_eiffel_tower(self):
        outcome = execute_analysis(
            png_upload(),
            "alpine village",
            developer_mode=True,
            screener=clean_screener,
            sleeper=lambda seconds: None,
        )
        self.assertEqual(outcome["kind"], "ok")
        self.assertTrue(outcome["mock"])
        self.assertEqual(
            outcome["result"]["coordinates"],
            {"latitude": 48.8584, "longitude": 2.2945},
        )
        self.assertEqual(outcome["result"]["filename"], "photo.png")

    def test_missing_file_halts(self):
        outcome = execute_analysis(
            None, "", screener=clean_screener, sleeper=lambda s: None
        )
        self.assertEqual(outcome["kind"], "halt")
        self.assertIn("No file", outcome["message"])

    def test_solid_image_halts_on_quality_gate(self):
        img = Image.new("RGB", (64, 64), (200, 30, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        outcome = execute_analysis(
            FakeUpload(buf.getvalue()),
            "",
            screener=clean_screener,
            sleeper=lambda s: None,
        )
        self.assertEqual(outcome["kind"], "halt")
        self.assertEqual(
            outcome["message"], "Image lacks sufficient visual data for analysis."
        )

    def test_rejected_face_halts(self):
        def reject(image):
            return ("rejected", "Privacy warning: too big.", None)

        outcome = execute_analysis(
            png_upload(), "", screener=reject, sleeper=lambda s: None
        )
        self.assertEqual(outcome["kind"], "halt")
        self.assertIn("Privacy warning", outcome["message"])

    def test_backend_failure_returns_error_outcome(self):
        def boom(*args, **kwargs):
            raise AnalysisError("Backend unavailable at X: down")

        outcome = execute_analysis(
            png_upload(),
            "",
            developer_mode=False,
            screener=clean_screener,
            backend=boom,
        )
        self.assertEqual(outcome["kind"], "error")
        self.assertIn("Backend unavailable", outcome["message"])

    def test_backend_success_forwards_screened_jpeg(self):
        seen = {}

        def fake_backend(jpeg_bytes, filename, hint):
            seen["bytes"] = jpeg_bytes
            seen["filename"] = filename
            return {"status": "ok", "echo": hint}

        outcome = execute_analysis(
            png_upload(name="memory.png"),
            "coast",
            developer_mode=False,
            screener=clean_screener,
            backend=fake_backend,
        )
        self.assertEqual(outcome["kind"], "ok")
        self.assertFalse(outcome["mock"])
        self.assertEqual(outcome["result"]["echo"], "coast")
        self.assertTrue(seen["bytes"].startswith(b"\xff\xd8\xff"))
        self.assertEqual(seen["filename"], "memory.png")


if __name__ == "__main__":
    unittest.main()
