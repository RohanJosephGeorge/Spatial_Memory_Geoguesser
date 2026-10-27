"""Upload sanitizer: metadata-free, standard RGB images at full resolution.

Every upload is decoded, flattened to RGB, and rebuilt pixel-by-pixel so no
EXIF, ICC profile, or other metadata survives. Dimensions are never changed.
Files over the byte cap are rejected before decoding to bound memory use.
"""

from __future__ import annotations

import io
import os
from typing import BinaryIO

from PIL import Image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
JPEG_QUALITY = 95
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def is_valid_image(uploaded_file) -> tuple[bool, str]:
    """Strictly validate that an uploaded file is a supported image type.

    Checks filename/extension, Streamlit-reported MIME type, non-empty size
    with an upper limit, and image magic bytes.
    """
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = getattr(uploaded_file, "name", "") or ""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Invalid file extension '{ext or '(none)'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    mime_type = getattr(uploaded_file, "type", "") or ""
    if mime_type not in ALLOWED_MIME_TYPES:
        return False, (
            f"Invalid MIME type '{mime_type or '(unknown)'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}."
        )

    size = getattr(uploaded_file, "size", None)
    if size is None:
        return False, "Could not determine file size."
    if size == 0:
        return False, "File is empty."
    if size > MAX_UPLOAD_BYTES:
        return False, "File exceeds 10 MB limit."

    try:
        uploaded_file.seek(0)
        header = uploaded_file.read(12)
    except Exception:
        return False, "Could not read file contents."
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    if len(header) < 4:
        return False, "File is too small to be a valid image."

    is_jpeg = header.startswith(b"\xff\xd8\xff")
    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = len(header) >= 12 and header[0:4] == b"RIFF" and header[8:12] == b"WEBP"
    if not (is_jpeg or is_png or is_webp):
        return False, "File contents do not match a JPEG, PNG, or WebP image."

    return True, "Valid image."


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to RGB, compositing transparency over white."""
    if img.mode == "RGB":
        return img
    if img.mode == "P":
        return _flatten_to_rgb(img.convert("RGBA"))
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGB"), mask=img.getchannel("A"))
        return background
    return img.convert("RGB")


def sanitize_uploaded_image(
    uploaded_file: BinaryIO, max_bytes: int = MAX_UPLOAD_BYTES
) -> tuple[bool, str, Image.Image | None]:
    """Strip metadata and normalize an upload to a standard RGB image.

    Never resizes: output dimensions always equal the decoded image.
    Returns (ok, message, image); image is None unless ok.
    """
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
    except Exception:
        return (False, "Could not read the uploaded file.", None)
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    if not data:
        return (False, "File is empty.", None)
    if len(data) > max_bytes:
        size_mb = len(data) / 1024 / 1024
        cap_mb = max_bytes / 1024 / 1024
        return (
            False,
            f"File is {size_mb:.1f} MB, over the {cap_mb:.0f} MB limit.",
            None,
        )

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            rgb = _flatten_to_rgb(img)
            # Rebuild from raw pixels: drops EXIF, ICC, and all other metadata.
            clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    except Exception:
        return (False, "Could not decode the uploaded image.", None)

    return (
        True,
        f"Sanitized to metadata-free RGB ({clean.width}x{clean.height}).",
        clean,
    )


def encode_clean_jpeg(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    """Encode an RGB image as a standard JPEG with no metadata attached."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
