"""Analysis backends: instant mock data or the real FastAPI service.

Mock mode never touches the network; the caller adds any simulated delay.
Real mode POSTs the privacy-screened image to the configured backend.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

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
