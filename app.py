"""Basic SpatialGeoGuesser upload app."""

import asyncio
import json
import urllib.request

import pandas as pd
import streamlit as st

from analysis import execute_analysis
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
except ImportError:  # pragma: no cover
    sac = None

try:
    from streamlit_extras.metric_cards import style_metric_cards
except ImportError:  # pragma: no cover
    style_metric_cards = None

NEON_ACCENT = "#00FFAA"
METRIC_CARD_BACKGROUND = "#1E1E1E"

LOTTIE_RADAR_URL = (
    "https://raw.githubusercontent.com/"
    "soheltanvir0699/radar-scan-notification-logger/main/"
    "app/src/main/assets/lottie_radar.json"
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


def show_centered_lottie(slot, animation: dict) -> None:
    """Render a Lottie animation centered inside a placeholder slot."""
    with slot.container():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st_lottie(animation, height=240, key="analyzing_lottie")


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


async def _run_sam_segmentation(screened_image, on_retry):
    async with SamSegmenter(on_retry=on_retry) as segmenter:
        return await segmenter.segment_image(screened_image)


def render_sam_section(screened_image, is_processing: bool = False) -> None:
    """Live SAM 3.1 concept segmentation for the screened upload."""
    if screened_image is None:
        return
    st.space("small")
    st.subheader("SAM 3.1 live segmentation")
    st.caption(
        "Concept masks for text, architectural features, and vegetation "
        "via Meta Model API (requires META_API_KEY in secrets)."
    )
    if st.button(
        "Segment with SAM 3.1",
        key="sam_btn",
        disabled=is_processing,
        help="Sends the full-resolution screened image to SAM 3.1.",
    ):
        retry_status = st.empty()

        def _show_retry(attempt: int, wait_seconds: float) -> None:
            retry_status.info(RETRY_STATUS_TEXT)

        try:
            with st.spinner("Segmenting with SAM 3.1..."):
                sam_result = asyncio.run(
                    _run_sam_segmentation(screened_image, _show_retry)
                )
        except SamApiError as exc:
            retry_status.empty()
            st.session_state["sam_result"] = {"error": str(exc)}
        else:
            retry_status.empty()
            st.session_state["sam_result"] = {
                "summary": summarize_masks(sam_result),
                "annotated": draw_mask_boxes(
                    screened_image, sam_result.get("masks", [])
                ),
                "mask_count": sam_result.get("mask_count", 0),
            }
    stored = st.session_state.get("sam_result")
    if not stored:
        return
    if "error" in stored:
        st.error(stored["error"])
    else:
        st.success(f"SAM 3.1 found {stored['mask_count']} mask(s).")
        st.image(
            stored["annotated"],
            caption="SAM 3.1 concept boxes",
            use_container_width=True,
        )
        st.json(stored["summary"])


async def _run_wonder_reasoning(sam_summary, regional_hint, on_retry):
    async with SpatialReasoner(on_retry=on_retry) as reasoner:
        return await reasoner.reason(sam_summary, regional_hint)


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
        if result.get("latitude") is None:
            steps = result.get("reasoning_steps") or ["Analysis halted."]
            st.warning(f"Analysis halted: {steps[0]}")
            render_wonder_result(result)
        else:
            st.success(
                f"Located: {result['city']}, {result['country']} "
                f"({result['confidence_score']:.0%} confidence)."
            )
            render_wonder_result(result)
            render_result_map(
                {
                    "coordinates": {
                        "latitude": result["latitude"],
                        "longitude": result["longitude"],
                    }
                }
            )


def render_home(developer_mode: bool = True) -> None:
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("analysis_pending", False)
    st.session_state.setdefault("analysis_outcome", None)
    st.session_state.setdefault("last_upload_key", None)
    st.session_state.setdefault("screened_image", None)
    st.session_state.setdefault("sam_result", None)
    st.session_state.setdefault("wonder_result", None)
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
        anim_slot = st.empty()
        if st_lottie is not None:
            animation = load_lottie_animation()
            if animation is not None:
                show_centered_lottie(anim_slot, animation)
        with st.spinner(label):
            outcome = execute_analysis(
                uploaded_file, regional_hint, developer_mode
            )
        anim_slot.empty()
        st.session_state["analysis_outcome"] = outcome
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
    render_sam_section(
        st.session_state["screened_image"], is_processing=is_processing
    )
    sam_stored = st.session_state.get("sam_result") or {}
    render_wonder_section(
        sam_stored.get("summary"),
        regional_hint,
        is_processing=is_processing,
    )


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
        "5. **Go live** (optional): segment with SAM 3.1, then geolocate "
        "with Muse Wonder using your Meta API key."
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
        value=True,
        key="developer_mode",
        help="When on, Analyze Memory returns instant mock data instead of "
        "calling the FastAPI backend.",
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
