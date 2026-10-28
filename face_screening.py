"""Face screening for uploads.

Uses OpenCV's bundled Haar cascade (frontal-face weights ship with
opencv-python, so detection is fully offline with no downloads).
Strict cascade parameters suppress background noise, and boxes under
MIN_FACE_PIXELS are dropped as pixel artifacts. Any single remaining
face covering more than 5% of the image area rejects the image;
smaller detected faces are heavily blurred with Pillow before the
image is passed on. Only clean or successfully blurred images move
forward.
"""

from __future__ import annotations

from typing import BinaryIO, Callable

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

FACE_AREA_THRESHOLD = 0.05
CASCADE_SCALE_FACTOR = 1.1
CASCADE_MIN_NEIGHBORS = 10
CASCADE_MIN_SIZE = (50, 50)
MIN_FACE_PIXELS = 50
BLUR_PADDING_RATIO = 0.15

Box = tuple[int, int, int, int]
Detector = Callable[[np.ndarray], list[Box]]


def face_area_ratio(box: Box, image_width: int, image_height: int) -> float:
    """Fraction of total image area covered by one bounding box."""
    _, _, w, h = box
    return (w * h) / (image_width * image_height)


def find_oversized_face(
    boxes: list[Box], image_width: int, image_height: int
) -> Box | None:
    """Return the first face above the area threshold, else None."""
    for box in boxes:
        if face_area_ratio(box, image_width, image_height) > FACE_AREA_THRESHOLD:
            return box
    return None


def detect_face_boxes(bgr: np.ndarray) -> list[Box]:
    """Detect faces with the offline Haar cascade (grayscale input).

    Fully local: the cascade XML ships with OpenCV, so no weights are
    ever downloaded. Boxes under MIN_FACE_PIXELS are dropped as
    pixel artifacts.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; face detection unavailable.")
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError(
            "Haar cascade weights missing from this OpenCV install; "
            "face detection unavailable."
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=CASCADE_SCALE_FACTOR,
        minNeighbors=CASCADE_MIN_NEIGHBORS,
        minSize=CASCADE_MIN_SIZE,
    )
    if len(faces) == 0:
        return []
    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    return [
        box
        for box in boxes
        if box[2] >= MIN_FACE_PIXELS and box[3] >= MIN_FACE_PIXELS
    ]


def to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to BGR for the detector."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; face detection unavailable.")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _padded_box(box: Box, image_width: int, image_height: int) -> Box:
    x, y, w, h = box
    pad_w = int(w * BLUR_PADDING_RATIO)
    pad_h = int(h * BLUR_PADDING_RATIO)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(image_width, x + w + pad_w)
    y1 = min(image_height, y + h + pad_h)
    return (x0, y0, x1 - x0, y1 - y0)


def blur_face_regions(image: Image.Image, boxes: list[Box]) -> Image.Image:
    """Return a copy with a heavy Gaussian blur applied inside each box."""
    width, height = image.size
    blurred = image.copy()
    for box in boxes:
        x, y, w, h = _padded_box(box, width, height)
        if w <= 0 or h <= 0:
            continue
        region = blurred.crop((x, y, x + w, y + h))
        radius = max(10, max(w, h) // 2)
        blurred.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x, y))
    return blurred


def screen_image_faces(
    image: Image.Image, detector: Detector | None = None
) -> tuple[str, str, Image.Image | None]:
    """Screen one image: reject large faces, blur small ones.

    Returns (status, message, safe_image) with status in
    {"clean", "blurred", "rejected", "error"}. Only "clean" and "blurred"
    carry an image forward; callers must not use the original otherwise.
    """
    width, height = image.size
    try:
        if detector is not None:
            boxes = detector(np.array(image.convert("L")))
        else:
            boxes = detect_face_boxes(to_bgr(image))
    except Exception as exc:
        return ("error", f"Face detector failed: {exc}", None)

    if not boxes:
        return ("clean", "No faces detected.", image)

    oversized = find_oversized_face(boxes, width, height)
    if oversized is not None:
        pct = face_area_ratio(oversized, width, height) * 100
        return (
            "rejected",
            f"Privacy warning: a detected face covers {pct:.1f}% of the image "
            f"(limit {FACE_AREA_THRESHOLD * 100:.0f}%). "
            "No images containing unblurred faces are processed; "
            "please blur or crop out faces and re-upload.",
            None,
        )

    return (
        "blurred",
        f"{len(boxes)} small face(s) detected and blurred before processing.",
        blur_face_regions(image, boxes),
    )


def preprocess_uploaded_image(
    uploaded_file: BinaryIO, detector: Detector | None = None
) -> tuple[str, str, Image.Image | None]:
    """Open an uploaded file and screen it for faces (fail closed)."""
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")
    except Exception:
        return ("error", "Could not decode the uploaded image.", None)
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return screen_image_faces(image, detector=detector)
