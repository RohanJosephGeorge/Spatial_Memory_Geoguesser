"""Tests for the visual-data gate: solid/near-solid images halt, real
photos pass, and the halt message is exact.

Run: python -m unittest discover -s tests -v
Requires only numpy and Pillow (no OpenCV, no Streamlit, no network).
"""

import unittest

import numpy as np
from PIL import Image

from image_quality import (
    INSUFFICIENT_MESSAGE,
    has_sufficient_visual_data,
    image_information,
)


def gradient_image(width=64, height=48):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    y = np.tile(np.arange(height, dtype=np.uint8)[:, None], (1, width))
    return Image.fromarray(np.stack([x, y, x], axis=-1), mode="RGB")


class VisualDataGateTests(unittest.TestCase):
    def test_solid_color_halts(self):
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        ok, message, metrics = has_sufficient_visual_data(img)
        self.assertFalse(ok)
        self.assertEqual(metrics["variance"], 0.0)
        self.assertEqual(metrics["entropy"], 0.0)

    def test_nearly_solid_halts(self):
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        img.putpixel((0, 0), (129, 129, 129))
        ok, _, _ = has_sufficient_visual_data(img)
        self.assertFalse(ok)

    def test_low_entropy_halts_despite_high_variance(self):
        # Two flat halves (black/white): huge variance but only 1 bit of
        # entropy, so a strict entropy threshold still halts it.
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, 32:] = 255
        img = Image.fromarray(arr, mode="RGB")
        self.assertGreater(image_information(img)["variance"], 100.0)
        ok, _, _ = has_sufficient_visual_data(
            img, var_threshold=100.0, entropy_threshold=2.0
        )
        self.assertFalse(ok)

    def test_textured_photo_passes(self):
        ok, _, metrics = has_sufficient_visual_data(gradient_image())
        self.assertTrue(ok)
        self.assertGreater(metrics["variance"], 100.0)
        self.assertGreater(metrics["entropy"], 1.0)

    def test_halt_message_is_exact(self):
        img = Image.new("RGB", (8, 8), (0, 0, 255))
        _, message, _ = has_sufficient_visual_data(img)
        self.assertEqual(message, INSUFFICIENT_MESSAGE)
        self.assertEqual(message, "Image lacks sufficient visual data for analysis.")

    def test_metrics_report_variance_and_entropy(self):
        metrics = image_information(gradient_image())
        self.assertIn("variance", metrics)
        self.assertIn("entropy", metrics)
        self.assertGreaterEqual(metrics["variance"], 0.0)
        self.assertGreaterEqual(metrics["entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
