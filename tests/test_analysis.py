"""Tests for analysis backends: mock payload contents and graceful
backend failure.

Run: python -m unittest discover -s tests -v
Requires stdlib only (no network: the failure test hits a closed
loopback port).
"""

import unittest

from analysis import (
    AnalysisError,
    analyze_via_backend,
    mock_analysis_result,
)


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


if __name__ == "__main__":
    unittest.main()
