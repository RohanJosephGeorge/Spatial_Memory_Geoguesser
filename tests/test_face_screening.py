"""Focused tests for face-screening threshold and blur behavior.

Run: python -m unittest discover -s tests -v
Requires only numpy and Pillow (no OpenCV, no Streamlit).
"""

import io
import unittest

import numpy as np
from PIL import Image

from face_screening import (
    blur_face_regions,
    face_area_ratio,
    find_oversized_face,
    preprocess_uploaded_image,
    screen_image_faces,
)


def gradient_image(width=200, height=200):
    arr = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


class FaceThresholdTests(unittest.TestCase):
    def test_area_ratio_math(self):
        self.assertAlmostEqual(face_area_ratio((0, 0, 100, 100), 1000, 1000), 0.01)

    def test_exactly_five_percent_is_allowed(self):
        # 200x250 in 1000x1000 == 5.0%: only *more than* 5% rejects.
        self.assertIsNone(find_oversized_face([(0, 0, 200, 250)], 1000, 1000))

    def test_above_five_percent_is_rejected(self):
        box = (0, 0, 230, 230)  # 5.29% of 1000x1000
        self.assertEqual(find_oversized_face([box], 1000, 1000), box)

    def test_mixed_boxes_flagged_by_largest(self):
        small = (0, 0, 50, 50)
        big = (0, 0, 300, 300)
        self.assertEqual(find_oversized_face([small, big], 1000, 1000), big)


class ScreenImageTests(unittest.TestCase):
    def test_clean_when_no_faces(self):
        img = gradient_image()
        status, _, out = screen_image_faces(img, detector=lambda gray: [])
        self.assertEqual(status, "clean")
        self.assertIs(out, img)

    def test_rejects_large_face_and_passes_nothing(self):
        status, message, out = screen_image_faces(
            gradient_image(), detector=lambda gray: [(10, 10, 150, 150)]
        )
        self.assertEqual(status, "rejected")
        self.assertIsNone(out)
        self.assertIn("unblurred faces", message)

    def test_blurs_small_face_and_passes_blurred_image(self):
        img = gradient_image()
        status, _, out = screen_image_faces(
            img, detector=lambda gray: [(50, 50, 30, 30)]
        )
        self.assertEqual(status, "blurred")
        self.assertIsNotNone(out)
        before = np.asarray(img).astype(int)
        after = np.asarray(out).astype(int)
        self.assertTrue(np.abs(before - after).sum() > 0)

    def test_detector_failure_is_error_not_passthrough(self):
        def boom(gray):
            raise RuntimeError("no weights")

        status, _, out = screen_image_faces(gradient_image(), detector=boom)
        self.assertEqual(status, "error")
        self.assertIsNone(out)


class BlurLocalityTests(unittest.TestCase):
    def test_blur_changes_box_region_only(self):
        img = gradient_image()
        out = blur_face_regions(img, [(80, 80, 40, 40)])
        before = np.asarray(img).astype(int)
        after = np.asarray(out).astype(int)
        diff = np.abs(before - after).sum(axis=-1)
        self.assertGreater(diff[60:140, 60:140].sum(), 0)
        corner = np.concatenate(
            [diff[0:10, :].ravel(), diff[:, 0:10].ravel()]
        )
        self.assertEqual(corner.sum(), 0)

    def test_original_image_not_mutated(self):
        img = gradient_image()
        snapshot = np.asarray(img).copy()
        blur_face_regions(img, [(80, 80, 40, 40)])
        np.testing.assert_array_equal(np.asarray(img), snapshot)


class PreprocessUploadTests(unittest.TestCase):
    def test_garbage_bytes_fail_closed(self):
        status, _, out = preprocess_uploaded_image(io.BytesIO(b"not an image"))
        self.assertEqual(status, "error")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
