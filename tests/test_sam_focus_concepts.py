"""Multi-category SAM extraction flows into the Muse Wonder prompt.

Run: python -m unittest discover -s tests -v
Requires no network: categories, geometry, and prompts are local only.
"""

import unittest

from sam_segmentation import (
    CONCEPT_CATEGORIES,
    FOCUS_CONCEPTS,
    category_for_concept,
    describe_position,
    format_visual_breakdown,
    normalize_box,
)
from spatial_reasoning import build_spatial_prompt

REQUIRED_CATEGORIES = (
    "Regional Architecture & Construction",
    "Flora, Agriculture & Soil",
    "Infrastructure & Utilities",
    "Local Signage, Text & Script",
    "Local Transport & Cultural Artifacts",
)

# Defining regional cues each category must query for.
CATEGORY_CUES = {
    "Regional Architecture & Construction": (
        "terracotta",
        "laterite",
        "veranda",
        "window grill",
    ),
    "Flora, Agriculture & Soil": (
        "coconut",
        "areca",
        "paddy",
        "laterite soil",
    ),
    "Infrastructure & Utilities": (
        "transformer",
        "cable clutter",
        "interlock paver",
        "curb",
    ),
    "Local Signage, Text & Script": (
        "hoarding",
        "Malayalam",
        "milestone",
        "nameplate",
    ),
    "Local Transport & Cultural Artifacts": (
        "auto-rickshaw",
        "KSRTC",
        "license plate",
    ),
}


def summary_for(concept, box=None, width=200, height=100):
    return {
        "concept": concept,
        "object_id": "0",
        "confidence": 0.92,
        "box": (
            box
            if box is not None
            else {"left": 10, "top": 20, "right": 110, "bottom": 70}
        ),
        "width": width,
        "height": height,
        "pixel_count": 5123,
    }


class ConceptPipelineTests(unittest.TestCase):
    def test_five_required_categories_exist(self):
        self.assertEqual(tuple(CONCEPT_CATEGORIES), REQUIRED_CATEGORIES)

    def test_each_category_queries_its_regional_cues(self):
        for category, cues in CATEGORY_CUES.items():
            joined = " ".join(CONCEPT_CATEGORIES[category]).lower()
            for cue in cues:
                self.assertIn(cue.lower(), joined, msg=category)

    def test_focus_concepts_flatten_all_categories(self):
        flattened = [
            phrase
            for phrases in CONCEPT_CATEGORIES.values()
            for phrase in phrases
        ]
        self.assertEqual(list(FOCUS_CONCEPTS), flattened)
        self.assertGreater(len(FOCUS_CONCEPTS), 4)

    def test_category_lookup_with_general_fallback(self):
        first_phrase = next(iter(CONCEPT_CATEGORIES.values()))[0]
        first_category = next(iter(CONCEPT_CATEGORIES))
        self.assertEqual(category_for_concept(first_phrase), first_category)
        self.assertEqual(category_for_concept("something else"), "General")


class GeometryHelperTests(unittest.TestCase):
    def test_normalize_box_uses_top_left_width_height(self):
        normalized = normalize_box(
            {"left": 10, "top": 20, "right": 110, "bottom": 70},
            200,
            100,
        )
        self.assertEqual(
            normalized, {"top": 0.2, "left": 0.05, "width": 0.5, "height": 0.5}
        )

    def test_normalize_box_rejects_missing_or_degenerate(self):
        self.assertIsNone(normalize_box(None, 200, 100))
        self.assertIsNone(
            normalize_box({"left": 0, "top": 0}, 200, 100)
        )
        self.assertIsNone(
            normalize_box(
                {"left": 0, "top": 0, "right": 10, "bottom": 10}, 0, 100
            )
        )

    def test_describe_position_names_zone_and_depth(self):
        upper_left_small = {"top": 0.0, "left": 0.0, "width": 0.1, "height": 0.1}
        position = describe_position(upper_left_small)
        self.assertIn("upper", position)
        self.assertIn("left", position)
        self.assertIn("background", position)
        lower_center_large = {"top": 0.5, "left": 0.2, "width": 0.6, "height": 0.5}
        position = describe_position(lower_center_large)
        self.assertIn("lower", position)
        self.assertIn("foreground", position)

    def test_describe_position_unknown_without_box(self):
        self.assertEqual(describe_position(None), "unknown")


class BreakdownPromptTests(unittest.TestCase):
    def test_breakdown_lists_label_category_box_and_position(self):
        concept = FOCUS_CONCEPTS[0]
        breakdown = format_visual_breakdown([summary_for(concept)])
        self.assertIn(concept, breakdown)
        self.assertIn(category_for_concept(concept), breakdown)
        self.assertIn("top=0.2", breakdown)
        self.assertIn("left=0.05", breakdown)
        self.assertIn("width=0.5", breakdown)
        self.assertIn("height=0.5", breakdown)
        self.assertIn("foreground", breakdown)

    def test_breakdown_handles_empty_result(self):
        breakdown = format_visual_breakdown([])
        self.assertIn("0 masks", breakdown)

    def test_wonder_prompt_contains_detailed_breakdown(self):
        concept = FOCUS_CONCEPTS[0]
        prompt = build_spatial_prompt([summary_for(concept)], "test hint")
        self.assertIn("VISUAL BREAKDOWN", prompt)
        self.assertIn(concept, prompt)
        self.assertIn(category_for_concept(concept), prompt)
        self.assertIn("top=0.2", prompt)
        self.assertIn("foreground", prompt)


if __name__ == "__main__":
    unittest.main()
