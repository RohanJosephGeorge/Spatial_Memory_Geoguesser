"""Focused tests for face-screening threshold and blur behavior.

Run: python -m unittest discover -s tests -v
Requires only numpy and Pillow (no OpenCV, no Streamlit).
"""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import face_screening
from face_screening import (
    blur_face_regions,
    detect_face_boxes,
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


class FakeCascade:
    def __init__(self, faces, detections, empty=False):
        self._faces = faces
        self._detections = detections
        self._empty = empty

    def empty(self):
        return self._empty

    def detectMultiScale(
        self, gray, scaleFactor=None, minNeighbors=None, minSize=None
    ):
        self._detections.append(
            {
                "gray": gray,
                "scaleFactor": scaleFactor,
                "minNeighbors": minNeighbors,
                "minSize": minSize,
            }
        )
        return self._faces


class FakeCv2Haar:
    """Minimal cv2 stub exposing only the Haar cascade entry points."""

    COLOR_BGR2GRAY = 6

    def __init__(self, faces, constructed, detections, empty=False):
        self._faces = faces
        self.constructed = constructed
        self.detections = detections
        self._empty = empty
        self.data = SimpleNamespace(haarcascades="/fake/haarcascades/")
        self.cvt_calls = []

    def CascadeClassifier(self, path):
        self.constructed.append(path)
        return FakeCascade(self._faces, self.detections, self._empty)

    def cvtColor(self, image, code):
        out = image[:, :, 0]
        self.cvt_calls.append({"code": code, "out": out})
        return out


class HaarCascadeTests(unittest.TestCase):
    def detect_with(self, faces, empty=False):
        fake = FakeCv2Haar(faces, [], [], empty)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(face_screening, "cv2", fake):
            return detect_face_boxes(bgr), fake

    def test_uses_bundled_offline_cascade(self):
        _, fake = self.detect_with(())
        self.assertEqual(
            fake.constructed,
            ["/fake/haarcascades/" + "haarcascade_frontalface_default.xml"],
        )
        self.assertFalse(hasattr(face_screening, "urllib"))
        self.assertFalse(hasattr(face_screening, "ensure_yunet_weights"))

    def test_strict_cascade_parameters(self):
        self.assertEqual(face_screening.CASCADE_SCALE_FACTOR, 1.1)
        self.assertEqual(face_screening.CASCADE_MIN_NEIGHBORS, 10)
        self.assertEqual(face_screening.CASCADE_MIN_SIZE, (50, 50))
        _, fake = self.detect_with(())
        self.assertEqual(len(fake.detections), 1)
        params = fake.detections[0]
        self.assertEqual(params["scaleFactor"], 1.1)
        self.assertEqual(params["minNeighbors"], 10)
        self.assertEqual(params["minSize"], (50, 50))

    def test_grayscale_before_detect(self):
        _, fake = self.detect_with(())
        self.assertEqual(len(fake.cvt_calls), 1)
        self.assertEqual(
            fake.cvt_calls[0]["code"], FakeCv2Haar.COLOR_BGR2GRAY
        )
        self.assertIs(
            fake.detections[0]["gray"], fake.cvt_calls[0]["out"]
        )

    def test_boxes_normalized_and_tiny_dropped(self):
        boxes, _ = self.detect_with(
            np.array([[10, 10, 100, 80], [5, 5, 20, 20]], dtype=np.int32)
        )
        self.assertEqual(boxes, [(10, 10, 100, 80)])

    def test_no_faces_returns_empty(self):
        boxes, _ = self.detect_with(())
        self.assertEqual(boxes, [])

    def test_missing_cascade_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self.detect_with((), empty=True)
        fake = FakeCv2Haar((), [], [], empty=True)
        with patch.object(face_screening, "cv2", fake):
            status, _, out = screen_image_faces(gradient_image())
        self.assertEqual(status, "error")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
