"""Basic SpatialGeoGuesser upload app."""

import asyncio
import json
import urllib.request

import pandas as pd
import streamlit as st

from analysis import execute_analysis
from backend import build_frontend_result, round_wonder_coordinates
from face_screening import screen_image_faces
from file_sanitizer import is_valid_image, sanitize_uploaded_image
from image_quality import has_sufficient_visual_data
from sam_segmentation import (
    RETRY_STATUS_TEXT,
    SamApiError,
    SamSegmenter,
    draw_mask_boxes,
    summarize_masks,
)
from spatial_reasoning import (
    LocationUnknownError,
    SpatialReasoner,
    SpatialReasoningError,
)

try:
    import streamlit_shadcn_ui as ui
except ImportError:  # pragma: no cover
    ui = None

try:
    from streamlit_option_menu import option_menu
except ImportError:  # pragma: no cover
    option_menu = None

try:
    from streamlit_lottie import st_lottie
except ImportError:  # pragma: no cover
    st_lottie = None

try:
    import streamlit_antd_components as sac
except Exception:  # pragma: no cover - missing OR broken install degrades
    sac = None

try:
    from streamlit_extras.metric_cards import style_metric_cards
except ImportError:  # pragma: no cover
    style_metric_cards = None

try:
    import folium
except ImportError:  # pragma: no cover
    folium = None

try:
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover
    st_folium = None

NEON_ACCENT = "#00FFAA"
METRIC_CARD_BACKGROUND = "#1E1E1E"

LOTTIE_RADAR_URL = (
    "https://raw.githubusercontent.com/"
    "soheltanvir0699/radar-scan-notification-logger/main/"
    "app/src/main/assets/lottie_radar.json"
)
LOTTIE_SEARCH_EMPTY_URL = (
    "https://raw.githubusercontent.com/ramani888/biblestudy/master/"
    "frontend/src/assets/animations/empty.json"
)
LOTTIE_FETCH_TIMEOUT = 10.0


@st.cache_data(show_spinner=False, ttl=3600)
def load_lottie_animation(url: str = LOTTIE_RADAR_URL) -> dict | None:
    """Fetch a Lottie JSON animation; None when unavailable.

    Any failure (network, timeout, invalid JSON, non-object payload)
    returns None so callers fall back to the plain spinner.
    """
    try:
        with urllib.request.urlopen(url, timeout=LOTTIE_FETCH_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def show_centered_lottie(
    slot, animation: dict, key: str = "analyzing_lottie"
) -> None:
    """Render a Lottie animation centered inside a placeholder slot."""
    with slot.container():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st_lottie(animation, height=240, key=key)


def inject_heading_styles() -> None:
    st.markdown(
        """
        <style>
        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_map(result: dict) -> None:
    """Render a map for result coordinates when present."""
    coords = (result or {}).get("coordinates") or {}
    lat, lon = coords.get("latitude"), coords.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=13)


PRECISION_RADII_METERS = {
    "Public Landmark": 50,
    "Street Level": 100,
    "Neighborhood Radius": 500,
    "City Level": 10000,
    "State / Region Radius": 50000,
}
DEFAULT_RADIUS_METERS = 50000
FOLIUM_DARK_TILES = "CartoDB dark_matter"
CIRCLE_FILL_OPACITY = 0.3
WONDER_MAP_HEIGHT = 400


def precision_to_radius(precision_level: str) -> int:
    """Map a Wonder precision_level to a map circle radius in meters.

    Unknown levels fail toward the widest, most private radius.
    """
    return PRECISION_RADII_METERS.get(precision_level, DEFAULT_RADIUS_METERS)


def build_wonder_folium_map(result: dict):
    """Build a dark-theme folium map with a precision circle.

    Returns None when folium is unavailable or coordinates are missing.
    """
    if folium is None:
        return None
    result = result or {}
    lat, lon = result.get("latitude"), result.get("longitude")
    if not (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
    ):
        return None
    radius = precision_to_radius(result.get("precision_level"))
    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=14 if radius >= 500 else 15,
        tiles=FOLIUM_DARK_TILES,
    )
    folium.Circle(
        location=[lat, lon],
        radius=radius,
        color=NEON_ACCENT,
        weight=3,
        opacity=1.0,
        fill=True,
        fill_color=NEON_ACCENT,
        fill_opacity=CIRCLE_FILL_OPACITY,
    ).add_to(fmap)
    return fmap


def render_wonder_folium_map(result: dict) -> None:
    """Render the Wonder precision map, falling back to st.map if needed."""
    fmap = build_wonder_folium_map(result)
    if fmap is None or st_folium is None:
        render_result_map(
            {
                "coordinates": {
                    "latitude": (result or {}).get("latitude"),
                    "longitude": (result or {}).get("longitude"),
                }
            }
        )
        return
    st_folium(fmap, height=WONDER_MAP_HEIGHT, use_container_width=True)


def render_analysis_outcome(outcome: dict | None) -> None:
    """Render a stored analysis outcome (result + map, or halt/error)."""
    if not outcome:
        return
    kind = outcome.get("kind")
    if kind == "ok":
        if outcome.get("mock"):
            st.success("Mock analysis complete (no backend call).")
        else:
            st.success("Analysis complete.")
        st.json(outcome.get("result", {}))
        render_result_map(outcome.get("result", {}))
    elif kind == "halt":
        st.warning(outcome.get("message", "Analysis halted."))
    else:
        st.error(outcome.get("message", "Analysis failed."))


async def _run_wonder_reasoning(sam_summary, regional_hint, on_retry):
    async with SpatialReasoner(on_retry=on_retry) as reasoner:
        return await reasoner.reason(sam_summary, regional_hint)


async def _run_live_pipeline(safe_image, regional_hint, on_retry=None):
    """Run SAM 3.1 segmentation then Muse Wonder reasoning directly."""
    async with SamSegmenter(on_retry=on_retry) as segmenter:
        sam_result = await segmenter.segment_image(safe_image)
    async with SpatialReasoner(on_retry=on_retry) as reasoner:
        return await reasoner.reason(
            summarize_masks(sam_result), regional_hint, image=safe_image
        )


def _run_live_pipeline_sync(safe_image, regional_hint, on_retry=None):
    """Sync wrapper for the direct SAM+Wonder pipeline (test seam)."""
    return asyncio.run(_run_live_pipeline(safe_image, regional_hint, on_retry))


def execute_live_analysis_direct(
    uploaded_file, regional_hint: str, screener=screen_image_faces
) -> dict:
    """Validate gates then run SAM+Wonder directly (no localhost backend).

    Returns the same outcome shape as analysis.execute_analysis:
    {"kind": "ok", "mock": False, "result": dict} on success, or
    {"kind": "halt"|"error", "message": str}.
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
    try:
        wonder_result = _run_live_pipeline_sync(safe_image, regional_hint)
    except (SamApiError, SpatialReasoningError) as exc:
        return {"kind": "error", "message": str(exc)}
    try:
        rounded = round_wonder_coordinates(wonder_result)
    except ValueError as exc:
        return {"kind": "error", "message": f"Invalid location result: {exc}"}
    result = build_frontend_result(rounded, uploaded_file.name, regional_hint)
    result["mode"] = "live"
    return {"kind": "ok", "mock": False, "result": result}


LOW_CONFIDENCE_THRESHOLD = 0.2


def is_uncertain_result(result: dict) -> bool:
    """Null coordinates or confidence below 20%: location not determined."""
    result = result or {}
    if result.get("latitude") is None or result.get("longitude") is None:
        return True
    confidence = result.get("confidence_score", 0)
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        return True
    return confidence < LOW_CONFIDENCE_THRESHOLD


def render_uncertain_result(result: dict) -> None:
    """Empty-search animation plus the AI's reasoning for the failure."""
    steps = list((result or {}).get("reasoning_steps") or ["Analysis halted."])
    if st_lottie is not None:
        animation = load_lottie_animation(LOTTIE_SEARCH_EMPTY_URL)
        if animation is not None:
            show_centered_lottie(st.empty(), animation, key="uncertain_lottie")
    st.info(
        "**Why the location couldn't be determined:**\n"
        + "\n".join(f"- {step}" for step in steps)
    )


def render_wonder_result(result: dict) -> None:
    """Render a Wonder result: reasoning as nodes, scores as metric cards."""
    steps_list = list(result.get("reasoning_steps") or [])
    if sac is not None and steps_list:
        sac.steps(
            items=steps_list,
            index=len(steps_list) - 1,
            direction="vertical",
            dot=True,
            color=NEON_ACCENT,
        )
    else:
        for step in steps_list:
            st.markdown(f"- {step}")
    confidence_col, precision_col = st.columns(2)
    with confidence_col:
        st.metric(
            "Confidence Score",
            f"{result.get('confidence_score', 0):.0%}",
        )
    with precision_col:
        st.metric(
            "Precision Level",
            result.get("precision_level", "Unknown"),
        )
    if style_metric_cards is not None:
        style_metric_cards(
            background_color=METRIC_CARD_BACKGROUND,
            border_color=NEON_ACCENT,
            border_left_color=NEON_ACCENT,
            border_size_px=1,
            border_radius_px=8,
        )


def render_wonder_section(
    sam_summary, regional_hint: str, is_processing: bool = False
) -> None:
    """Muse Wonder geolocation over SAM masks plus the regional hint."""
    if not sam_summary:
        return
    st.space("small")
    st.subheader("Muse Wonder geolocation")
    st.caption(
        "Multi-step spatial reasoning over the SAM masks "
        "(requires META_API_KEY in secrets)."
    )
    if st.button(
        "Geolocate with Wonder",
        key="wonder_btn",
        disabled=is_processing,
        help="Sends SAM masks and the hint to Muse Wonder.",
    ):
        retry_status = st.empty()

        def _show_retry(attempt: int, wait_seconds: float) -> None:
            retry_status.info(RETRY_STATUS_TEXT)

        try:
            with st.spinner("Reasoning about location..."):
                wonder_result = asyncio.run(
                    _run_wonder_reasoning(
                        sam_summary, regional_hint, _show_retry
                    )
                )
        except LocationUnknownError as exc:
            retry_status.empty()
            st.session_state["wonder_result"] = {"location_unknown": str(exc)}
        except (SamApiError, SpatialReasoningError) as exc:
            retry_status.empty()
            st.session_state["wonder_result"] = {"error": str(exc)}
        else:
            retry_status.empty()
            st.session_state["wonder_result"] = {"result": wonder_result}
    stored = st.session_state.get("wonder_result")
    if not stored:
        return
    if "location_unknown" in stored:
        st.error(f"Location Unknown: {stored['location_unknown']}")
    elif "error" in stored:
        st.error(stored["error"])
    else:
        result = stored["result"]
        if is_uncertain_result(result):
            if result.get("latitude") is None:
                steps = result.get("reasoning_steps") or ["Analysis halted."]
                st.warning(f"Analysis halted: {steps[0]}")
            render_uncertain_result(result)
        else:
            st.success(
                f"Located: {result['city']}, {result['country']} "
                f"({result['confidence_score']:.0%} confidence)."
            )
            result_col, map_col = st.columns([3, 2])
            with result_col:
                render_wonder_result(result)
            with map_col:
                render_wonder_folium_map(result)


def render_home(developer_mode: bool = False) -> None:
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("analysis_pending", False)
    st.session_state.setdefault("analysis_outcome", None)
    st.session_state.setdefault("last_upload_key", None)
    st.session_state.setdefault("screened_image", None)
    st.session_state.setdefault("sam_result", None)
    st.session_state.setdefault("wonder_result", None)
    st.session_state.setdefault("developer_mode", developer_mode)
    if st.session_state.developer_mode:
        developer_mode = True
    else:
        developer_mode = False
    is_processing = st.session_state["is_processing"]

    st.title("SpatialGeoGuesser")
    st.caption("Upload a memory photo and add an optional regional hint.")
    st.space("small")

    if ui is None:
        st.error(
            "Missing dependency `streamlit-shadcn-ui`. "
            "Install it with: "
            "`pip install git+https://github.com/ObservedObserver/streamlit-shadcn-ui.git`"
        )
        return

    ui.card(
        title="Upload memory",
        description="JPEG, PNG, or WebP up to 10 MB.",
        key="upload_card",
    )

    st.space("small")
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="JPEG, PNG, or WebP up to 10 MB.",
            disabled=is_processing,
        )
        st.space("small")
        regional_hint = st.text_input(
            "Regional hint",
            placeholder="e.g. coastal town, temple architecture, alpine village",
        )
        st.space("small")
        analyze_clicked = ui.button(
            "Analyze Memory", key="analyze_btn", disabled=is_processing
        )

    if analyze_clicked and not is_processing:
        st.session_state["is_processing"] = True
        st.session_state["analysis_pending"] = True
        st.rerun()

    if uploaded_file is not None:
        valid, message = is_valid_image(uploaded_file)
        if not valid:
            st.error(message)
        else:
            clean_ok, clean_message, rgb_image = sanitize_uploaded_image(
                uploaded_file
            )
            if not clean_ok:
                st.error(clean_message)
            else:
                data_ok, data_message, _ = has_sufficient_visual_data(rgb_image)
                if not data_ok:
                    st.warning(data_message)
                else:
                    status, face_message, safe_image = screen_image_faces(
                        rgb_image
                    )
                    if status == "rejected":
                        st.error(face_message)
                    elif status == "error":
                        st.error(face_message)
                    else:
                        if status == "blurred":
                            st.warning(face_message)
                        else:
                            st.success(
                                f"{message} {clean_message} {face_message}"
                            )
                        st.image(
                            safe_image, caption="Privacy-safe preview",
                            use_container_width=True,
                        )
                        st.session_state["screened_image"] = safe_image

    if st.session_state["analysis_pending"] and is_processing:
        label = (
            "Analyzing memory (mock data)..."
            if developer_mode
            else "Analyzing memory..."
        )
        anim_slot = None
        try:
            anim_slot = st.empty()
            if st_lottie is not None:
                animation = load_lottie_animation()
                if animation is not None:
                    show_centered_lottie(anim_slot, animation)
            with st.spinner(label):
                if st.session_state.developer_mode:
                    outcome = execute_analysis(
                        uploaded_file, regional_hint, True
                    )
                else:
                    outcome = execute_live_analysis_direct(
                        uploaded_file, regional_hint
                    )
            st.session_state["analysis_outcome"] = outcome
        except Exception as exc:
            st.session_state["analysis_outcome"] = {
                "kind": "error",
                "message": f"Analysis failed unexpectedly: {exc}",
            }
        finally:
            if anim_slot is not None:
                anim_slot.empty()
            st.session_state["analysis_pending"] = False
            st.session_state["is_processing"] = False
        st.rerun()

    upload_key = (
        f"{uploaded_file.name}:{uploaded_file.size}"
        if uploaded_file is not None
        else None
    )
    if st.session_state["last_upload_key"] != upload_key:
        st.session_state["last_upload_key"] = upload_key
        st.session_state["analysis_outcome"] = None
        st.session_state["screened_image"] = None
        st.session_state["sam_result"] = None
        st.session_state["wonder_result"] = None

    render_analysis_outcome(st.session_state["analysis_outcome"])


def render_how_it_works() -> None:
    st.title("How it Works")
    st.caption("From photo to place.")
    st.markdown(
        "1. **Upload** a JPEG, PNG, or WebP photo (max 10 MB). Files are "
        "stripped of metadata and normalized to RGB at full resolution.\n"
        "2. **Pre-processing gates** run automatically: uploads lacking "
        "visual data are halted, any face over 5% of the image area is "
        "rejected, smaller faces are blurred.\n"
        "3. **Add a regional hint** such as nearby landmarks or architecture.\n"
        "4. **Analyze** to get location hypotheses based on visual clues.\n"
        "5. **Live mode**: with Developer Mode off, Analyze runs SAM 3.1 "
        "segmentation and Muse Wonder geolocation directly via the Meta "
        "Model API."
    )


def render_privacy_safety() -> None:
    st.title("Privacy & Safety")
    st.warning(
        "No images containing unblurred faces are processed. "
        "Please blur or crop out faces before uploading."
    )
    st.markdown(
        "Uploads are validated by type and size, previewed locally in your "
        "browser, and never stored with identifying facial information."
    )


def main() -> None:
    st.set_page_config(page_title="SpatialGeoGuesser", layout="centered")
    inject_heading_styles()

    developer_mode = st.sidebar.toggle(
        "Developer Mode (Use Mock Data)",
        value=False,
        key="developer_mode",
        help="When on, Analyze Memory returns instant mock data instead of "
        "calling the Meta Model API directly.",
    )

    if option_menu is None:
        st.error(
            "Missing dependency `streamlit-option-menu`. "
            "Install it with: `pip install streamlit-option-menu`"
        )
        return

    selected = option_menu(
        menu_title=None,
        options=["Home (Upload)", "How it Works", "Privacy & Safety"],
        icons=["camera", "map", "shield"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#1E1E1E",
            },
            "icon": {"color": "white", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "center",
                "margin": "0px",
                "color": "white",
                "--hover-color": "#333333",
            },
            "nav-link-selected": {
                "background-color": "#00FFAA",
                "color": "#1E1E1E",
            },
        },
    )

    if selected == "Home (Upload)":
        render_home(developer_mode)
    elif selected == "How it Works":
        render_how_it_works()
    else:
        render_privacy_safety()


if __name__ == "__main__":
    main()
