"""Tests for the Wonder folium precision map.

Run: python -m unittest discover -s tests -v
Requires folium, streamlit, and pandas. Streamlit rendering is mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

import app
from folium.vector_layers import Circle


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


def the_circle(fmap):
    circles = [c for c in fmap._children.values() if isinstance(c, Circle)]
    assert len(circles) == 1, f"expected 1 circle, found {len(circles)}"
    return circles[0]


class PrecisionRadiusTests(unittest.TestCase):
    def test_neighborhood_radius_is_500m(self):
        self.assertEqual(app.precision_to_radius("Neighborhood Radius"), 500)

    def test_street_level_is_100m(self):
        self.assertEqual(app.precision_to_radius("Street Level"), 100)

    def test_public_landmark_is_50m(self):
        self.assertEqual(app.precision_to_radius("Public Landmark"), 50)

    def test_unknown_level_fails_wide(self):
        self.assertEqual(app.precision_to_radius("Unknown"), 500)
        self.assertEqual(app.precision_to_radius(None), 500)


class BuildWonderFoliumMapTests(unittest.TestCase):
    def test_neighborhood_circle_radius_and_neon_style(self):
        fmap = app.build_wonder_folium_map(
            good_result(precision_level="Neighborhood Radius")
        )
        circle = the_circle(fmap)
        self.assertEqual(circle.options["radius"], 500)
        self.assertEqual(circle.options["color"], "#00FFAA")
        self.assertEqual(circle.options["fillColor"], "#00FFAA")
        self.assertEqual(circle.options["fillOpacity"], 0.3)
        self.assertEqual(circle.options["opacity"], 1.0)
        self.assertTrue(circle.options["fill"])
        self.assertEqual(circle.location, [48.8584, 2.2945])

    def test_street_level_circle_radius(self):
        fmap = app.build_wonder_folium_map(
            good_result(precision_level="Street Level")
        )
        self.assertEqual(the_circle(fmap).options["radius"], 100)

    def test_dark_tiles(self):
        html = (
            app.build_wonder_folium_map(good_result())
            .get_root()
            .render()
        )
        self.assertIn("dark_all", html)
        self.assertIn("cartocdn", html)

    def test_null_coordinates_returns_none(self):
        self.assertIsNone(
            app.build_wonder_folium_map(
                good_result(latitude=None, longitude=None)
            )
        )
        self.assertIsNone(app.build_wonder_folium_map({}))

    def test_missing_folium_returns_none(self):
        with patch.object(app, "folium", None):
            self.assertIsNone(
                app.build_wonder_folium_map(good_result())
            )


class RenderWonderFoliumMapTests(unittest.TestCase):
    def test_renders_with_sizing(self):
        mock_folium_render = MagicMock()
        with patch.object(app, "st_folium", mock_folium_render):
            app.render_wonder_folium_map(good_result())
        mock_folium_render.assert_called_once()
        args, kwargs = mock_folium_render.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(the_circle(args[0]).options["radius"], 50)
        self.assertEqual(kwargs["height"], app.WONDER_MAP_HEIGHT)
        self.assertTrue(kwargs["use_container_width"])

    def test_falls_back_to_st_map(self):
        mock_st = MagicMock()
        with (
            patch.object(app, "folium", None),
            patch.object(app, "st", mock_st),
        ):
            app.render_wonder_folium_map(good_result())
        mock_st.map.assert_called_once()

    def test_success_branch_lays_out_result_left_map_right(self):
        mock_st = MagicMock()
        left, right = MagicMock(), MagicMock()
        mock_st.columns.return_value = [left, right]
        mock_st.session_state.get.return_value = {"result": good_result()}
        mock_st.button.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "render_wonder_result") as mock_result,
            patch.object(app, "render_wonder_folium_map") as mock_map,
        ):
            app.render_wonder_section([{"concept": "text"}], "", False)
        mock_st.columns.assert_called_once_with([3, 2])
        mock_result.assert_called_once()
        mock_map.assert_called_once()
        mock_st.map.assert_not_called()


if __name__ == "__main__":
    unittest.main()
