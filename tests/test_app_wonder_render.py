"""Tests for Wonder result rendering: sac steps + styled metric cards.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit and sac are mocked: no browser.
"""

import unittest
from unittest.mock import MagicMock, patch

import app


def good_result():
    return {
        "latitude": 48.8584,
        "longitude": 2.2945,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower", "French signage"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }


def halt_result():
    return {
        "latitude": None,
        "longitude": None,
        "city": "",
        "country": "",
        "reasoning_steps": ["Not enough distinct features"],
        "confidence_score": 0.0,
        "precision_level": "Unknown",
    }


def mock_streamlit():
    mock_st = MagicMock()
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    return mock_st


class RenderWonderResultTests(unittest.TestCase):
    def test_reasoning_rendered_as_vertical_node_path(self):
        mock_st, mock_sac = mock_streamlit(), MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "sac", mock_sac),
        ):
            app.render_wonder_result(good_result())
        mock_sac.steps.assert_called_once()
        _, kwargs = mock_sac.steps.call_args
        self.assertEqual(
            kwargs["items"], ["Iron lattice tower", "French signage"]
        )
        self.assertEqual(kwargs["direction"], "vertical")
        self.assertTrue(kwargs["dot"])
        mock_st.json.assert_not_called()

    def test_confidence_and_precision_metric_cards_styled(self):
        mock_st, mock_style = mock_streamlit(), MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "sac", MagicMock()),
            patch.object(app, "style_metric_cards", mock_style),
        ):
            app.render_wonder_result(good_result())
        labels = [call.args[0] for call in mock_st.metric.call_args_list]
        self.assertEqual(labels, ["Confidence Score", "Precision Level"])
        values = [call.args[1] for call in mock_st.metric.call_args_list]
        self.assertEqual(values, ["95%", "Public Landmark"])
        mock_style.assert_called_once()
        _, kwargs = mock_style.call_args
        self.assertEqual(kwargs["background_color"], app.METRIC_CARD_BACKGROUND)
        self.assertEqual(kwargs["border_color"], app.NEON_ACCENT)
        self.assertEqual(kwargs["border_left_color"], app.NEON_ACCENT)

    def test_halt_result_renders_zero_and_unknown(self):
        mock_st = mock_streamlit()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "sac", MagicMock()),
        ):
            app.render_wonder_result(halt_result())
        values = [call.args[1] for call in mock_st.metric.call_args_list]
        self.assertEqual(values, ["0%", "Unknown"])

    def test_missing_sac_falls_back_to_bullets(self):
        mock_st = mock_streamlit()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "sac", None),
        ):
            app.render_wonder_result(good_result())
        mock_st.markdown.assert_called()
        mock_st.json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
