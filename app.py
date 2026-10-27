"""Basic SpatialGeoGuesser upload app."""

import pandas as pd
import streamlit as st

from analysis import execute_analysis
from face_screening import screen_image_faces
from file_sanitizer import is_valid_image, sanitize_uploaded_image
from image_quality import has_sufficient_visual_data

try:
    import streamlit_shadcn_ui as ui
except ImportError:  # pragma: no cover
    ui = None

try:
    from streamlit_option_menu import option_menu
except ImportError:  # pragma: no cover
    option_menu = None


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


def render_home(developer_mode: bool = True) -> None:
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("analysis_pending", False)
    st.session_state.setdefault("analysis_outcome", None)
    st.session_state.setdefault("last_upload_key", None)
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

    if st.session_state["analysis_pending"] and is_processing:
        label = (
            "Analyzing memory (mock data)..."
            if developer_mode
            else "Analyzing memory..."
        )
        with st.spinner(label):
            outcome = execute_analysis(
                uploaded_file, regional_hint, developer_mode
            )
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

    render_analysis_outcome(st.session_state["analysis_outcome"])


def render_how_it_works() -> None:
    st.title("How it Works")
    st.caption("Three steps from photo to place.")
    st.markdown(
        "1. **Upload** a JPEG, PNG, or WebP photo (max 10 MB). Files are "
        "stripped of metadata and normalized to RGB at full resolution.\n"
        "2. **Pre-processing gates** run automatically: uploads lacking "
        "visual data are halted, any face over 5% of the image area is "
        "rejected, smaller faces are blurred.\n"
        "3. **Add a regional hint** such as nearby landmarks or architecture.\n"
        "4. **Analyze** to get location hypotheses based on visual clues."
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
