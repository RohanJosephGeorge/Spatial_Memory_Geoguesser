"""Analysis backends: instant mock data or the real FastAPI service.

Mock mode never touches the network; the caller adds any simulated delay.
Real mode POSTs the privacy-screened image to the configured backend.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

from face_screening import screen_image_faces
from file_sanitizer import (
    encode_clean_jpeg,
    is_valid_image,
    sanitize_uploaded_image,
)
from image_quality import has_sufficient_visual_data

BACKEND_URL = os.environ.get("SPATIAL_BACKEND_URL", "http://localhost:8000")
ANALYZE_PATH = "/analyze"
EIFFEL_TOWER = {"latitude": 48.8584, "longitude": 2.2945}


class AnalysisError(Exception):
    """Raised when the real backend call fails for any reason."""


def mock_analysis_result(filename: str, regional_hint: str) -> dict:
    """Hardcoded dummy JSON: no backend call, no network."""
    hint = regional_hint.strip() or "(no hint provided)"
    return {
        "status": "ok",
        "mode": "mock",
        "filename": filename,
        "regional_hint": hint,
        "reasoning_steps": [
            "Detected wrought-iron lattice tower structure in frame.",
            "Matched silhouette against public landmark embeddings.",
            f"Cross-checked regional hint: {hint}.",
            "Constellation of nearby rooftops consistent with Paris, FR.",
        ],
        "precision": "Public Landmark",
        "confidence": 0.95,
        "coordinates": dict(EIFFEL_TOWER),
    }


def analyze_via_backend(
    image_jpeg_bytes: bytes,
    filename: str,
    regional_hint: str,
    backend_url: str = BACKEND_URL,
    timeout: float = 30.0,
) -> dict:
    """POST the screened image to the FastAPI backend; return parsed JSON."""
    payload = json.dumps(
        {
            "filename": filename,
            "regional_hint": regional_hint,
            "image_jpeg_base64": base64.b64encode(image_jpeg_bytes).decode("ascii"),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        backend_url.rstrip("/") + ANALYZE_PATH,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise AnalysisError(f"Backend unavailable at {backend_url}: {exc}")


def execute_analysis(
    uploaded_file,
    regional_hint: str,
    developer_mode: bool = True,
    screener=screen_image_faces,
    backend=analyze_via_backend,
    sleeper=time.sleep,
) -> dict:
    """Run validate -> sanitize -> quality -> screen -> analyze.

    Returns a JSON-serializable outcome dict: {"kind": "ok", "mock": bool,
    "result": dict} on success, or {"kind": "halt"|"error", "message": str}.
    """
    valid, message = is_valid_image(uploaded_file)
    if not valid:
        return {"kind": "halt", "message": f"Cannot analyze yet: {message}"}
    clean_ok, clean_message, rgb_image = sanitize_uploaded_image(uploaded_file)
    if not clean_ok:
        return {"kind": "halt", "message": f"Cannot analyze yet: {clean_message}"}
    data_ok, data_message, _ = has_sufficient_visual_data(rgb_image)
    if not data_ok:
        return {"kind": "halt", "message": data_message}
    status, face_message, safe_image = screener(rgb_image)
    if status in ("rejected", "error"):
        return {"kind": "halt", "message": f"Cannot analyze: {face_message}"}
    if developer_mode:
        sleeper(2)
        return {
            "kind": "ok",
            "mock": True,
            "result": mock_analysis_result(uploaded_file.name, regional_hint),
        }
    try:
        result = backend(
            encode_clean_jpeg(safe_image), uploaded_file.name, regional_hint
        )
    except AnalysisError as exc:
        return {"kind": "error", "message": str(exc)}
    return {"kind": "ok", "mock": False, "result": result}
