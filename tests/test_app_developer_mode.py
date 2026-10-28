"""Regression: sidebar toggle is source of truth; single Analyze button.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser.
"""

import unittest
from unittest.mock import MagicMock, patch

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


def base_state(**overrides):
    state = StateDict(
        {
            "developer_mode": True,
            "is_processing": False,
            "analysis_pending": False,
            "analysis_outcome": None,
            "last_upload_key": None,
            "screened_image": None,
            "sam_result": None,
            "wonder_result": None,
        }
    )
    state.update(overrides)
    return state


class DeveloperModeToggleTests(unittest.TestCase):
    def run_pending_analysis(self, state, param):
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        execute = MagicMock(
            return_value={"kind": "ok", "mock": True, "result": {}}
        )
        direct = MagicMock(
            return_value={"kind": "ok", "mock": False, "result": {}}
        )
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(app, "execute_analysis", execute),
            patch.object(app, "execute_live_analysis_direct", direct),
        ):
            app.render_home(developer_mode=param)
        return execute, direct

    def test_session_state_false_overrides_stale_true_param(self):
        state = base_state(
            developer_mode=False,
            is_processing=True,
            analysis_pending=True,
        )
        execute, direct = self.run_pending_analysis(state, param=True)
        execute.assert_not_called()
        direct.assert_called_once()

    def test_session_state_true_overrides_stale_false_param(self):
        state = base_state(
            developer_mode=True,
            is_processing=True,
            analysis_pending=True,
        )
        execute, direct = self.run_pending_analysis(state, param=False)
        execute.assert_called_once()
        passed_flag = execute.call_args.args[2]
        self.assertTrue(passed_flag)
        direct.assert_not_called()


class SingleButtonFlowTests(unittest.TestCase):
    def test_no_sam_section_or_extra_buttons(self):
        state = base_state(
            developer_mode=False,
            screened_image=MagicMock(),
            sam_result={
                "summary": [{"concept": "text"}],
                "mask_count": 1,
                "annotated": MagicMock(),
            },
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.button.return_value = False
        # Keep the upload-key guard from resetting our staged state.
        mock_st.session_state["last_upload_key"] = None
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
        ):
            app.render_home()
        for call in mock_st.subheader.call_args_list:
            self.assertNotIn("SAM 3.1", call.args[0])
        mock_st.button.assert_not_called()
        mock_ui.button.assert_called_once_with(
            "Analyze Memory", key="analyze_btn", disabled=False
        )


if __name__ == "__main__":
    unittest.main()
