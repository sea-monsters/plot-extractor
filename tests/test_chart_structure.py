"""Tests for chart structural decomposition and position context rules.

Covers:
- ChartStructure dataclass construction
- decompose_chart_structure() from detected axes
- resolve_element_role() position context classification
- Edge cases: missing axes, degenerate regions, axis-only decomposition
"""
import pytest
import numpy as np
from plot_extractor.core.axis_detector import Axis
from plot_extractor.layout.chart_structure import (
    ChartStructure,
    StructuralArea,
    decompose_chart_structure,
    resolve_element_role,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_x_axis(position, side="bottom", plot_start=0, plot_end=640):
    return Axis(direction="x", position=position, side=side,
                plot_start=plot_start, plot_end=plot_end, ticks=[])


def _make_y_axis(position, side="left", plot_start=0, plot_end=480):
    return Axis(direction="y", position=position, side=side,
                plot_start=plot_start, plot_end=plot_end, ticks=[])


# Standard chart: 640x480, x-bottom at y=400, y-left at x=80
STANDARD_AXES = [
    _make_x_axis(400, "bottom", 80, 640),
    _make_y_axis(80, "left", 0, 400),
]


# ---------------------------------------------------------------------------
# ChartStructure dataclass
# ---------------------------------------------------------------------------

class TestChartStructure:
    def test_chart_structure_fields(self):
        cs = ChartStructure(
            plot_area=(80, 30, 600, 400),
            x_axis_area=(80, 400, 600, 480),
            y_axis_area=(0, 30, 80, 400),
            legend_area=(600, 30, 640, 400),
        )
        assert cs.plot_area == (80, 30, 600, 400)
        assert cs.x_axis_area == (80, 400, 600, 480)
        assert cs.y_axis_area == (0, 30, 80, 400)
        assert cs.legend_area == (600, 30, 640, 400)

    def test_structural_area_named_tuple(self):
        sa = StructuralArea(x1=10, y1=20, x2=100, y2=200)
        assert sa.x1 == 10
        assert sa.y2 == 200
        cx, cy = sa.center()
        assert cx == 55
        assert cy == 110

    def test_structural_area_contains(self):
        sa = StructuralArea(x1=10, y1=20, x2=100, y2=200)
        assert sa.contains(50, 100)
        assert sa.contains(10, 20)   # boundary inclusive
        assert not sa.contains(5, 100)
        assert not sa.contains(50, 250)

    def test_structural_area_area(self):
        sa = StructuralArea(x1=0, y1=0, x2=100, y2=50)
        assert sa.area() == 5000


# ---------------------------------------------------------------------------
# decompose_chart_structure
# ---------------------------------------------------------------------------

class TestDecomposeChartStructure:
    def test_standard_two_axes(self):
        cs = decompose_chart_structure((480, 640), STANDARD_AXES)
        # plot_area: left=80, top=0 (no top axis), right=640 (no right axis), bottom=400
        assert cs.plot_area.x1 == 80
        assert cs.plot_area.y2 == 400
        # x_axis_area below plot area
        assert cs.x_axis_area.y1 == 400
        assert cs.x_axis_area.y2 == 480
        # y_axis_area left of plot area
        assert cs.y_axis_area.x1 == 0
        assert cs.y_axis_area.x2 == 80

    def test_four_axes_with_right_y(self):
        axes = [
            _make_x_axis(400, "bottom", 80, 640),
            _make_y_axis(80, "left", 0, 400),
            _make_y_axis(600, "right", 0, 400),
        ]
        cs = decompose_chart_structure((480, 640), axes)
        assert cs.plot_area.x1 == 80
        assert cs.plot_area.x2 == 600
        assert cs.plot_area.y2 == 400

    def test_no_axes_fallback(self):
        cs = decompose_chart_structure((480, 640), [])
        # Fallback: entire image minus margin
        assert cs.plot_area.x1 >= 0
        assert cs.plot_area.y2 <= 480
        assert cs.confidence < 50  # low confidence

    def test_only_x_axis(self):
        axes = [_make_x_axis(400, "bottom", 0, 640)]
        cs = decompose_chart_structure((480, 640), axes)
        assert cs.plot_area.y2 == 400
        assert cs.x_axis_area.y1 == 400

    def test_only_y_axis(self):
        axes = [_make_y_axis(80, "left", 0, 480)]
        cs = decompose_chart_structure((480, 640), axes)
        assert cs.plot_area.x1 == 80
        assert cs.y_axis_area.x2 == 80

    def test_plot_bounds_from_plot_start_end(self):
        """Axes carry plot_start/plot_end that refine the orthogonal extent."""
        axes = [
            Axis(direction="x", position=400, side="bottom",
                 plot_start=100, plot_end=600, ticks=[]),
            Axis(direction="y", position=100, side="left",
                 plot_start=50, plot_end=400, ticks=[]),
        ]
        cs = decompose_chart_structure((480, 640), axes)
        # x-axis plot_start/plot_end refine left/right
        assert cs.plot_area.x1 == 100
        assert cs.plot_area.x2 == 600
        # y-axis plot_start/plot_end refine top/bottom
        assert cs.plot_area.y1 == 50
        assert cs.plot_area.y2 == 400


# ---------------------------------------------------------------------------
# resolve_element_role
# ---------------------------------------------------------------------------

class TestResolveElementRole:
    def setup_method(self):
        self.cs = ChartStructure(
            plot_area=StructuralArea(80, 30, 600, 400),
            x_axis_area=StructuralArea(80, 400, 600, 480),
            y_axis_area=StructuralArea(0, 30, 80, 400),
            legend_area=StructuralArea(600, 30, 640, 400),
        )

    def test_data_element_in_plot_area(self):
        role = resolve_element_role((200, 200, 210, 210), self.cs)
        assert role == "data_element"

    def test_text_in_plot_area_is_value_label(self):
        role = resolve_element_role((200, 200, 215, 212), self.cs, is_text=True)
        assert role == "value_label"

    def test_x_tick_label_in_x_axis_area(self):
        role = resolve_element_role((200, 420, 230, 440), self.cs, is_text=True)
        assert role == "x_tick_label"

    def test_y_tick_label_in_y_axis_area(self):
        role = resolve_element_role((20, 200, 70, 215), self.cs, is_text=True)
        assert role == "y_tick_label"

    def test_legend_label_in_legend_area(self):
        role = resolve_element_role((610, 100, 635, 112), self.cs, is_text=True)
        assert role == "legend_label"

    def test_legend_marker_in_legend_area(self):
        role = resolve_element_role((605, 100, 615, 110), self.cs, is_text=False)
        assert role == "legend_marker"

    def test_title_above_plot_area(self):
        role = resolve_element_role((200, 5, 400, 25), self.cs, is_text=True)
        assert role == "chart_title"

    def test_x_tick_mark_near_axis_edge(self):
        # Small bbox near bottom of plot area = tick mark, not data
        role = resolve_element_role((200, 395, 205, 405), self.cs, is_text=False)
        assert role == "x_tick_mark"

    def test_y_tick_mark_near_axis_edge(self):
        role = resolve_element_role((75, 200, 85, 205), self.cs, is_text=False)
        assert role == "y_tick_mark"

    def test_unknown_role(self):
        # Far outside all areas
        role = resolve_element_role((0, 460, 10, 475), self.cs)
        assert role == "others"
