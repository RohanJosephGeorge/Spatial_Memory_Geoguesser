"""Tests for the upload sanitizer: EXIF stripping, RGB normalize, no
resize, and the 10MB cap.

Run: python -m unittest discover -s tests -v
Requires only numpy and Pillow (no OpenCV, no Streamlit).
"""

import io
import unittest

import numpy as np
from PIL import Image

from file_sanitizer import (
    MAX_UPLOAD_BYTES,
    encode_clean_jpeg,
    is_valid_image,
    sanitize_uploaded_image,
)


def gradient_image(width=64, height=48):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    y = np.tile(np.arange(height, dtype=np.uint8)[:, None], (1, width))
    return Image.fromarray(np.stack([x, y, x], axis=-1), mode="RGB")


def jpeg_bytes_with_exif(img):
    exif = Image.Exif()
    exif[271] = "TestMake"  # Make
    exif[274] = 6  # Orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


class ExifStrippingTests(unittest.TestCase):
    def test_exif_removed_but_pixels_kept(self):
        data = jpeg_bytes_with_exif(gradient_image())
        self.assertGreater(len(Image.open(io.BytesIO(data)).getexif()), 0)
        ok, _, out = sanitize_uploaded_image(io.BytesIO(data))
        self.assertTrue(ok)
        self.assertIsNotNone(out)
        self.assertEqual(len(out.getexif()), 0)
        self.assertEqual(out.info, {})
        reference = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        np.testing.assert_array_equal(np.asarray(out), reference)

    def test_dimensions_never_change(self):
        img = gradient_image(101, 77)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        ok, _, out = sanitize_uploaded_image(io.BytesIO(buf.getvalue()))
        self.assertTrue(ok)
        self.assertEqual(out.size, (101, 77))


class ModeNormalizationTests(unittest.TestCase):
    def test_rgba_flattens_over_white(self):
        rgba = gradient_image().convert("RGBA")
        rgba.putalpha(Image.new("L", rgba.size, 0))  # fully transparent
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")
        ok, _, out = sanitize_uploaded_image(io.BytesIO(buf.getvalue()))
        self.assertTrue(ok)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, rgba.size)
        self.assertEqual(out.getpixel((0, 0)), (255, 255, 255))

    def test_palette_image_becomes_rgb(self):
        pal = gradient_image().convert("P")
        buf = io.BytesIO()
        pal.save(buf, format="PNG")
        ok, _, out = sanitize_uploaded_image(io.BytesIO(buf.getvalue()))
        self.assertTrue(ok)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, pal.size)


class SizeCapTests(unittest.TestCase):
    def test_over_cap_rejected_without_decoding(self):
        big = io.BytesIO(b"0" * (MAX_UPLOAD_BYTES + 1))
        ok, message, out = sanitize_uploaded_image(big)
        self.assertFalse(ok)
        self.assertIsNone(out)
        self.assertIn("10 MB", message)

    def test_empty_rejected(self):
        ok, _, out = sanitize_uploaded_image(io.BytesIO(b""))
        self.assertFalse(ok)
        self.assertIsNone(out)

    def test_garbage_rejected(self):
        ok, message, out = sanitize_uploaded_image(io.BytesIO(b"not an image"))
        self.assertFalse(ok)
        self.assertIsNone(out)
        self.assertIn("decode", message)

    def test_stream_position_restored(self):
        f = io.BytesIO(jpeg_bytes_with_exif(gradient_image()))
        sanitize_uploaded_image(f)
        self.assertEqual(f.tell(), 0)


class ImageValidationTests(unittest.TestCase):
    def _upload(self, data, name="photo.png", mime="image/png", size=None):
        f = io.BytesIO(data)
        f.name = name
        f.type = mime
        f.size = len(data) if size is None else size
        return f

    def _png_bytes(self):
        buf = io.BytesIO()
        gradient_image().save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_png_passes(self):
        valid, _ = is_valid_image(self._upload(self._png_bytes()))
        self.assertTrue(valid)

    def test_none_rejected(self):
        valid, message = is_valid_image(None)
        self.assertFalse(valid)
        self.assertIn("No file", message)

    def test_bad_extension_rejected(self):
        valid, message = is_valid_image(
            self._upload(self._png_bytes(), name="photo.gif", mime="image/gif")
        )
        self.assertFalse(valid)
        self.assertIn("extension", message)

    def test_bad_mime_rejected(self):
        valid, message = is_valid_image(
            self._upload(self._png_bytes(), mime="application/octet-stream")
        )
        self.assertFalse(valid)
        self.assertIn("MIME", message)

    def test_oversize_rejected(self):
        valid, message = is_valid_image(
            self._upload(self._png_bytes(), size=MAX_UPLOAD_BYTES + 1)
        )
        self.assertFalse(valid)
        self.assertIn("10 MB", message)

    def test_wrong_magic_bytes_rejected(self):
        valid, message = is_valid_image(self._upload(b"GIF89a" + b"0" * 20))
        self.assertFalse(valid)
        self.assertIn("contents", message)


class JpegEncodingTests(unittest.TestCase):
    def test_clean_jpeg_round_trip(self):
        data = encode_clean_jpeg(gradient_image())
        self.assertTrue(data.startswith(b"\xff\xd8\xff"))
        reopened = Image.open(io.BytesIO(data))
        reopened.load()
        self.assertEqual(reopened.size, (64, 48))
        self.assertEqual(len(reopened.getexif()), 0)


if __name__ == "__main__":
    unittest.main()
