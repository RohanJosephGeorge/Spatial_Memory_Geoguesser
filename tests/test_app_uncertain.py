"""Tests for the uncertain Wonder result state (lottie + info, no map).

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit rendering is mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

import app


def good_result(**overrides):
    result = {
        "latitude": 48.8584,
        "longitude": 2.2945,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }
    result.update(overrides)
    return result


def mock_section_st(stored):
    mock_st = MagicMock()
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    mock_st.session_state.get.return_value = stored
    mock_st.button.return_value = False
    return mock_st


class IsUncertainResultTests(unittest.TestCase):
    def test_null_coordinates_uncertain(self):
        self.assertTrue(
            app.is_uncertain_result(
                good_result(latitude=None, longitude=None)
            )
        )
        self.assertTrue(app.is_uncertain_result(good_result(latitude=None)))
        self.assertTrue(app.is_uncertain_result(good_result(longitude=None)))
        self.assertTrue(app.is_uncertain_result({}))
        self.assertTrue(app.is_uncertain_result(None))

    def test_below_20_percent_uncertain(self):
        self.assertTrue(app.is_uncertain_result(good_result(confidence_score=0.19)))
        self.assertTrue(app.is_uncertain_result(good_result(confidence_score=0.0)))

    def test_exactly_20_percent_is_certain(self):
        self.assertFalse(app.is_uncertain_result(good_result(confidence_score=0.2)))

    def test_high_confidence_is_certain(self):
        self.assertFalse(app.is_uncertain_result(good_result()))

    def test_bad_confidence_fails_uncertain(self):
        no_score = good_result()
        del no_score["confidence_score"]
        self.assertTrue(app.is_uncertain_result(no_score))
        self.assertTrue(
            app.is_uncertain_result(good_result(confidence_score="high"))
        )
        self.assertTrue(
            app.is_uncertain_result(good_result(confidence_score=True))
        )


class RenderUncertainResultTests(unittest.TestCase):
    def render_with(self, result, animation):
        mock_st = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_lottie = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "st_lottie", mock_lottie),
            patch.object(
                app,
                "load_lottie_animation",
                MagicMock(return_value=animation),
            ) as mock_load,
        ):
            app.render_uncertain_result(result)
        return mock_st, mock_lottie, mock_load

    def test_animation_and_info_with_reasoning(self):
        mock_st, mock_lottie, mock_load = self.render_with(
            good_result(confidence_score=0.1),
            {"v": "5.5.9", "layers": []},
        )
        mock_load.assert_called_once_with(app.LOTTIE_SEARCH_EMPTY_URL)
        mock_lottie.assert_called_once()
        self.assertEqual(mock_lottie.call_args.kwargs["key"], "uncertain_lottie")
        mock_st.info.assert_called_once()
        text = mock_st.info.call_args.args[0]
        self.assertIn("couldn't be determined", text)
        self.assertIn("Iron lattice tower", text)
        mock_st.map.assert_not_called()

    def test_missing_animation_still_shows_info(self):
        mock_st, mock_lottie, _ = self.render_with(
            good_result(latitude=None, longitude=None), None
        )
        mock_lottie.assert_not_called()
        mock_st.info.assert_called_once()

    def test_missing_lottie_library_still_shows_info(self):
        mock_st = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "st_lottie", None),
        ):
            app.render_uncertain_result(good_result(confidence_score=0.1))
        mock_st.info.assert_called_once()

    def test_search_empty_url_shape(self):
        self.assertTrue(app.LOTTIE_SEARCH_EMPTY_URL.startswith("https://"))
        self.assertTrue(app.LOTTIE_SEARCH_EMPTY_URL.endswith("empty.json"))


class UncertainRoutingTests(unittest.TestCase):
    def render_section(self, result):
        mock_st = mock_section_st({"result": result})
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "render_uncertain_result") as mock_uncertain,
            patch.object(app, "render_wonder_result") as mock_result,
            patch.object(app, "render_wonder_folium_map") as mock_map,
        ):
            app.render_wonder_section([{"concept": "text"}], "", False)
        return mock_st, mock_uncertain, mock_result, mock_map

    def test_low_confidence_skips_map_and_success(self):
        mock_st, mock_uncertain, mock_result, mock_map = self.render_section(
            good_result(confidence_score=0.19)
        )
        mock_uncertain.assert_called_once()
        mock_result.assert_not_called()
        mock_map.assert_not_called()
        mock_st.success.assert_not_called()
        mock_st.map.assert_not_called()

    def test_null_coordinates_warn_and_show_uncertain(self):
        mock_st, mock_uncertain, _, mock_map = self.render_section(
            good_result(latitude=None, longitude=None)
        )
        mock_st.warning.assert_called_once()
        mock_uncertain.assert_called_once()
        mock_map.assert_not_called()

    def test_boundary_confidence_keeps_map(self):
        _, mock_uncertain, mock_result, mock_map = self.render_section(
            good_result(confidence_score=0.2)
        )
        mock_uncertain.assert_not_called()
        mock_result.assert_called_once()
        mock_map.assert_called_once()


if __name__ == "__main__":
    unittest.main()
