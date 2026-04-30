"""Chart structural decomposition and position context rules.

Inspired by CACHED (2305.04151) 18-class framework:
decomposes chart into 4 structural areas (plot_area, x_axis_area,
y_axis_area, legend_area) and resolves element roles by position context.

Pure geometric implementation — no deep learning required.
"""
from dataclasses import dataclass
from typing import List, NamedTuple

from plot_extractor.core.axis_detector import Axis


class StructuralArea(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def center(self):
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class ChartStructure:
    plot_area: StructuralArea
    x_axis_area: StructuralArea
    y_axis_area: StructuralArea
    legend_area: StructuralArea
    confidence: float = 90.0


def decompose_chart_structure(
    image_shape: tuple,
    axes: List[Axis],
    margin: int = 40,
) -> ChartStructure:
    """Decompose chart into 4 structural areas from detected axes.

    Equivalent to CACHED's 4 Structural Area Objects, but using pure
    geometric rules derived from detected axis positions.

    Args:
        image_shape: (h, w) of the image.
        axes: List of detected Axis objects.
        margin: fallback margin from image edge when axes missing.

    Returns:
        ChartStructure with 4 structural areas and a confidence score.
    """
    h, w = image_shape[:2]

    x_axes = [a for a in axes if a.direction == "x"]
    y_axes = [a for a in axes if a.direction == "y"]

    bottom_axis = next((a for a in x_axes if a.side == "bottom"), None)
    top_axis = next((a for a in x_axes if a.side == "top"), None)
    left_axis = next((a for a in y_axes if a.side == "left"), None)
    right_axis = next((a for a in y_axes if a.side == "right"), None)

    has_axes = bool(x_axes) or bool(y_axes)
    confidence = 90.0 if has_axes else 30.0

    # --- Plot area bounds ---
    # Left/right: from Y-axis positions, refined by X-axis plot_start/plot_end
    plot_left = left_axis.position if left_axis else margin
    plot_right = right_axis.position if right_axis else (w - margin)

    # Top/bottom: from X-axis positions, refined by Y-axis plot_start/plot_end
    plot_top = top_axis.position if top_axis else margin // 2
    plot_bottom = bottom_axis.position if bottom_axis else (h - margin)

    # Refine from axis plot_start/plot_end (orthogonal extent)
    if bottom_axis and bottom_axis.plot_start > 0:
        plot_left = max(plot_left, bottom_axis.plot_start)
    if bottom_axis and bottom_axis.plot_end < w:
        plot_right = min(plot_right, bottom_axis.plot_end)
    if top_axis and top_axis.plot_start > 0:
        plot_left = max(plot_left, top_axis.plot_start)
    if top_axis and top_axis.plot_end < w:
        plot_right = min(plot_right, top_axis.plot_end)

    if left_axis and left_axis.plot_start > 0:
        plot_top = max(plot_top, left_axis.plot_start)
    if left_axis and left_axis.plot_end < h:
        plot_bottom = min(plot_bottom, left_axis.plot_end)
    if right_axis and right_axis.plot_start > 0:
        plot_top = max(plot_top, right_axis.plot_start)
    if right_axis and right_axis.plot_end < h:
        plot_bottom = min(plot_bottom, right_axis.plot_end)

    # Clamp to image bounds
    plot_left = max(0, plot_left)
    plot_top = max(0, plot_top)
    plot_right = min(w, plot_right)
    plot_bottom = min(h, plot_bottom)

    # Ensure minimum size
    if plot_right - plot_left < w * 0.2:
        plot_left = margin // 2
        plot_right = w - margin // 2
        confidence = min(confidence, 50.0)
    if plot_bottom - plot_top < h * 0.2:
        plot_top = margin // 2
        plot_bottom = h - margin // 2
        confidence = min(confidence, 50.0)

    plot_area = StructuralArea(plot_left, plot_top, plot_right, plot_bottom)

    # --- Structural areas ---
    x_axis_area = StructuralArea(
        plot_left, plot_bottom, plot_right, h,
    )
    y_axis_area = StructuralArea(
        0, plot_top, plot_left, plot_bottom,
    )

    # Legend area: largest non-axis external region
    candidates = [
        StructuralArea(plot_right, plot_top, w, plot_bottom),  # right
        StructuralArea(plot_left, 0, plot_right, plot_top),    # top
    ]
    legend_area = max(candidates, key=lambda r: r.area())

    return ChartStructure(
        plot_area=plot_area,
        x_axis_area=x_axis_area,
        y_axis_area=y_axis_area,
        legend_area=legend_area,
        confidence=confidence,
    )


def resolve_element_role(
    bbox_xyxy: tuple,
    cs: ChartStructure,
    is_text: bool = False,
    axis_proximity: int = 10,
) -> str:
    """Resolve element role from position context.

    Equivalent to CACHED PCE module, but using deterministic rules.

    Args:
        bbox_xyxy: (x1, y1, x2, y2) candidate element bounding box.
        cs: ChartStructure from decompose_chart_structure().
        is_text: Whether the element was identified as text.
        axis_proximity: Pixels from axis edge to classify as tick mark.

    Returns:
        Role string: data_element, x_tick_label, y_tick_label,
        x_tick_mark, y_tick_mark, legend_label, legend_marker,
        value_label, chart_title, axis_title, others.
    """
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    bw = bbox_xyxy[2] - bbox_xyxy[0]
    bh = bbox_xyxy[3] - bbox_xyxy[1]

    pa = cs.plot_area
    xa = cs.x_axis_area
    ya = cs.y_axis_area
    la = cs.legend_area

    # Priority 1: inside plot area
    if pa.contains(cx, cy):
        # Near axis edge → tick mark
        if abs(cy - pa.y2) <= axis_proximity:
            return "x_tick_mark"
        if abs(cy - pa.y1) <= axis_proximity:
            return "x_tick_mark"
        if abs(cx - pa.x1) <= axis_proximity:
            return "y_tick_mark"
        if abs(cx - pa.x2) <= axis_proximity:
            return "y_tick_mark"
        if is_text:
            return "value_label"
        return "data_element"

    # Priority 2: X axis area
    if xa.contains(cx, cy):
        return "x_tick_label" if is_text else "x_tick_mark"

    # Priority 3: Y axis area
    if ya.contains(cx, cy):
        return "y_tick_label" if is_text else "y_tick_mark"

    # Priority 4: Legend area
    if la.contains(cx, cy):
        return "legend_label" if is_text else "legend_marker"

    # Priority 5: Above plot area → title or axis title
    if cy < pa.y1:
        if is_text:
            # Narrow text spanning the plot width → axis title
            if bw > pa.x2 - pa.x1 and bh < 30:
                return "axis_title"
            return "chart_title"
        return "others"

    # Below plot area but outside x_axis_area → axis title or others
    if cy > pa.y2 and is_text:
        return "axis_title"

    return "others"
