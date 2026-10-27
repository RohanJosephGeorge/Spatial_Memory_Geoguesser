"""Visual-data gate: halt uploads with too little information to analyze.

Pillow-only check over grayscale variance (ImageStat) and histogram entropy
(stdlib math). Solid, nearly solid, or completely flat/blurry images fall
below the thresholds and must not proceed. No API calls, no network.
"""

from __future__ import annotations

import math

from PIL import Image, ImageStat

VAR_THRESHOLD = 100.0
ENTROPY_THRESHOLD = 1.0
INSUFFICIENT_MESSAGE = "Image lacks sufficient visual data for analysis."


def image_information(image: Image.Image) -> dict[str, float]:
    """Return grayscale variance and histogram entropy for an image."""
    gray = image.convert("L")
    variance = float(ImageStat.Stat(gray).var[0])
    histogram = gray.histogram()
    total = sum(histogram)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in histogram
        if count
    )
    return {"variance": variance, "entropy": float(entropy)}


def has_sufficient_visual_data(
    image: Image.Image,
    var_threshold: float = VAR_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> tuple[bool, str, dict[str, float]]:
    """Decide whether an image carries enough visual data for analysis.

    Returns (ok, message, metrics). When ok is False the caller must halt:
    show INSUFFICIENT_MESSAGE and make no further processing or API calls.
    """
    metrics = image_information(image)
    if (
        metrics["variance"] < var_threshold
        or metrics["entropy"] < entropy_threshold
    ):
        return (False, INSUFFICIENT_MESSAGE, metrics)
    return (
        True,
        f"Sufficient visual data "
        f"(variance={metrics['variance']:.1f}, "
        f"entropy={metrics['entropy']:.2f} bits).",
        metrics,
    )
