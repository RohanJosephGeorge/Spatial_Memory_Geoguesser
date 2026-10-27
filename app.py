"""Basic SpatialGeoGuesser upload app."""

import os

import streamlit as st

try:
    import streamlit_shadcn_ui as ui
except ImportError:  # pragma: no cover
    ui = None

try:
    from streamlit_option_menu import option_menu
except ImportError:  # pragma: no cover
    option_menu = None


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


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
    if size > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds {MAX_FILE_SIZE_MB} MB limit."

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


def render_home() -> None:
    st.title("SpatialGeoGuesser")
    st.caption("Upload a memory photo and add an optional regional hint.")

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

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="JPEG, PNG, or WebP up to 10 MB.",
        )
        regional_hint = st.text_input(
            "Regional hint",
            placeholder="e.g. coastal town, temple architecture, alpine village",
        )
        analyze_clicked = ui.button("Analyze Memory", key="analyze_btn")

    if uploaded_file is not None:
        valid, message = is_valid_image(uploaded_file)
        if valid:
            st.success(message)
            st.image(uploaded_file, caption="Preview", use_container_width=True)
        else:
            st.error(message)

    if analyze_clicked:
        valid, message = is_valid_image(uploaded_file)
        if not valid:
            st.warning(f"Cannot analyze yet: {message}")
        else:
            hint_text = regional_hint.strip() or "(no hint provided)"
            st.info(
                f"Dummy action: would analyze '{uploaded_file.name}' "
                f"with regional hint: {hint_text}."
            )


def render_how_it_works() -> None:
    st.title("How it Works")
    st.caption("Three steps from photo to place.")
    st.markdown(
        "1. **Upload** a JPEG, PNG, or WebP photo (max 10 MB).\n"
        "2. **Add a regional hint** such as nearby landmarks or architecture.\n"
        "3. **Analyze** to get location hypotheses based on visual clues."
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
        render_home()
    elif selected == "How it Works":
        render_how_it_works()
    else:
        render_privacy_safety()


if __name__ == "__main__":
    main()
