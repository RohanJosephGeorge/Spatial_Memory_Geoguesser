"""FastAPI backend for SpatialGeoGuesser: /analyze endpoint.

Receives the privacy-screened upload from the Streamlit frontend (see
analysis.analyze_via_backend), runs the visual-data gate, SAM 3.1
segmentation, and Muse Wonder geolocation, then returns the result.

Mathematical privacy: Wonder's exact coordinates are intercepted by
round_wonder_coordinates() and rounded to PRIVACY_DECIMALS (3) decimal
places -- roughly a 100 m grid cell -- before anything is sent back to
the frontend. Three decimals span ~111 m of latitude at the equator
(longitude cells shrink with cos(latitude)), so the frontend can place
an approximate marker but never recovers the model's exact fix.

Face screening is intentionally not repeated here: the frontend only
uploads images that already passed screen_image_faces() ("clean" or
"blurred"). The Meta API key is read from the META_API_KEY environment
variable and passed explicitly, so this server never touches
Streamlit secrets.

Run: uvicorn backend:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import binascii
import io
import os

from file_sanitizer import MAX_UPLOAD_BYTES
from image_quality import has_sufficient_visual_data
from PIL import Image
from sam_segmentation import SamApiError, SamSegmenter, summarize_masks
from spatial_reasoning import (
    LocationUnknownError,
    SpatialReasoner,
    SpatialReasoningError,
)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - server extras not installed
    FastAPI = None
    HTTPException = None
    BaseModel = object

PRIVACY_DECIMALS = 3
COORDINATE_KEYS = ("latitude", "longitude")


def round_wonder_coordinates(
    wonder_result: dict, decimals: int = PRIVACY_DECIMALS
) -> dict:
    """Return a copy of a Muse Wonder result with rounded coordinates.

    Intercepts the latitude/longitude floats and rounds each to exactly
    ``decimals`` places (3 by default, ~100 m). Null coordinates (the
    halted-analysis convention) pass through untouched. The input dict
    is never mutated; every other key is preserved as-is.

    Raises ValueError (fail closed) when a coordinate key is missing or
    holds something other than a real number or None, so an unrounded
    payload can never be mistaken for a privacy-safe one.
    """
    rounded = dict(wonder_result)
    for key in COORDINATE_KEYS:
        if key not in rounded:
            raise ValueError(f"Muse Wonder result is missing {key!r}.")
        value = rounded[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{key} must be a number or null, "
                f"got {type(value).__name__}."
            )
        rounded[key] = float(round(value, decimals))
    return rounded


def build_frontend_result(
    wonder_result: dict, filename: str, regional_hint: str
) -> dict:
    """Map a (rounded) Wonder result onto the frontend payload shape.

    Mirrors mock_analysis_result() keys so the Streamlit UI renders
    backend and mock results identically. Callers must pass Wonder
    output through round_wonder_coordinates() first.
    """
    hint = (regional_hint or "").strip() or "(no hint provided)"
    return {
        "status": "ok",
        "mode": "backend",
        "filename": filename,
        "regional_hint": hint,
        "city": wonder_result["city"],
        "country": wonder_result["country"],
        "reasoning_steps": wonder_result["reasoning_steps"],
        "precision": wonder_result["precision_level"],
        "confidence": wonder_result["confidence_score"],
        "coordinates": {
            "latitude": wonder_result["latitude"],
            "longitude": wonder_result["longitude"],
        },
    }


def decode_upload(image_jpeg_base64: str) -> Image.Image:
    """Decode the frontend's base64 JPEG into an RGB image (fail closed)."""
    try:
        data = base64.b64decode(image_jpeg_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_jpeg_base64 is not valid base64.") from exc
    if not data:
        raise ValueError("Decoded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Decoded image exceeds the 10 MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError("Could not decode the uploaded image.") from exc


def get_backend_api_key() -> str:
    """Read META_API_KEY from the environment (server-side config)."""
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Meta Model API key: set META_API_KEY "
            "in the backend environment."
        )
    return key


if FastAPI is not None:

    app = FastAPI(title="SpatialGeoGuesser backend")

    class AnalyzeRequest(BaseModel):
        filename: str
        regional_hint: str = ""
        image_jpeg_base64: str

    @app.post("/analyze")
    async def analyze(payload: AnalyzeRequest) -> dict:
        """Screen, segment, geolocate, then return privacy-rounded coords."""
        try:
            api_key = get_backend_api_key()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        try:
            image = decode_upload(payload.image_jpeg_base64)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        data_ok, data_message, _ = has_sufficient_visual_data(image)
        if not data_ok:
            raise HTTPException(status_code=422, detail=data_message)
        try:
            async with SamSegmenter(api_key=api_key) as segmenter:
                sam_result = await segmenter.segment_image(image)
            async with SpatialReasoner(api_key=api_key) as reasoner:
                wonder_result = await reasoner.reason(
                    summarize_masks(sam_result), payload.regional_hint
                )
        except LocationUnknownError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except SpatialReasoningError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except SamApiError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        rounded = round_wonder_coordinates(wonder_result)
        return build_frontend_result(
            rounded, payload.filename, payload.regional_hint
        )

else:  # pragma: no cover - importable without server extras installed
    app = None


if __name__ == "__main__":  # pragma: no cover - manual server launch
    if app is None:
        raise RuntimeError(
            "FastAPI is not installed. Run: pip install fastapi uvicorn"
        )
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
