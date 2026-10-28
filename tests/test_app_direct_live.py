"""Direct live pipeline: no localhost backend, toggle defaults off.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser, no network.
"""

import inspect
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import app


class StateDict(dict):
    """Dict with attribute access, mirroring Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeUpload(io.BytesIO):
    def __init__(self, data, name="photo.png", mime="image/png"):
        super().__init__(data)
        self.name = name
        self.type = mime
        self.size = len(data)


def png_upload(width=64, height=48, name="photo.png"):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    y = np.tile(np.arange(height, dtype=np.uint8)[:, None], (1, width))
    img = Image.fromarray(np.stack([x, y, x], axis=-1), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return FakeUpload(buf.getvalue(), name=name)


def clean_screener(image):
    return ("clean", "No faces detected.", image)


def wonder_payload():
    return {
        "latitude": 48.8584,
        "longitude": 2.2945,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }


class ToggleDefaultTests(unittest.TestCase):
    def test_render_home_defaults_to_live(self):
        sig = inspect.signature(app.render_home)
        self.assertFalse(sig.parameters["developer_mode"].default)

    def test_fresh_session_defaults_developer_mode_off(self):
        state = StateDict(
            {
                "is_processing": False,
                "analysis_pending": False,
                "analysis_outcome": None,
                "last_upload_key": None,
                "screened_image": None,
                "sam_result": None,
                "wonder_result": None,
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
        ):
            app.render_home()
        self.assertFalse(state["developer_mode"])

    def test_sidebar_toggle_defaults_off(self):
        mock_st = MagicMock()
        mock_st.sidebar.toggle.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "option_menu", MagicMock(return_value="Home (Upload)")),
            patch.object(app, "render_home") as mock_home,
        ):
            app.main()
        _, kwargs = mock_st.sidebar.toggle.call_args
        self.assertFalse(kwargs["value"])
        self.assertEqual(kwargs["key"], "developer_mode")


class DirectLivePipelineTests(unittest.TestCase):
    def test_live_path_bypasses_localhost_backend(self):
        state = StateDict(
            {
                "developer_mode": False,
                "is_processing": True,
                "analysis_pending": True,
                "analysis_outcome": None,
                "last_upload_key": None,
                "screened_image": None,
                "sam_result": None,
                "wonder_result": None,
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        mock_execute = MagicMock()
        mock_direct = MagicMock(
            return_value={"kind": "ok", "mock": False, "result": {}}
        )
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(app, "execute_analysis", mock_execute),
            patch.object(app, "execute_live_analysis_direct", mock_direct),
        ):
            app.render_home(developer_mode=False)
        mock_execute.assert_not_called()
        mock_direct.assert_called_once()

    def test_direct_missing_file_halts(self):
        outcome = app.execute_live_analysis_direct(
            None, "", screener=clean_screener
        )
        self.assertEqual(outcome["kind"], "halt")
        self.assertIn("No file", outcome["message"])

    def test_direct_success_returns_live_result_without_localhost(self):
        with (
            patch.object(
                app, "_run_live_pipeline_sync", return_value=wonder_payload()
            ) as mock_run,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            outcome = app.execute_live_analysis_direct(
                png_upload(), "riverside", screener=clean_screener
            )
        mock_run.assert_called_once()
        mock_urlopen.assert_not_called()
        self.assertEqual(outcome["kind"], "ok")
        self.assertFalse(outcome["mock"])
        self.assertEqual(outcome["result"]["mode"], "live")
        self.assertEqual(
            outcome["result"]["coordinates"],
            {"latitude": 48.858, "longitude": 2.295},
        )

    def test_direct_api_error_returns_error_outcome(self):
        from sam_segmentation import SamApiError

        with patch.object(
            app, "_run_live_pipeline_sync", side_effect=SamApiError("down")
        ):
            outcome = app.execute_live_analysis_direct(
                png_upload(), "", screener=clean_screener
            )
        self.assertEqual(outcome["kind"], "error")
        self.assertIn("down", outcome["message"])


if __name__ == "__main__":
    unittest.main()
