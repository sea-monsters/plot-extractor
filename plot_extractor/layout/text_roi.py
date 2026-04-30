"""Text ROI (Region of Interest) extraction.

Defines regions for tick labels, legend, and title text
based on plot area location.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from plot_extractor.layout.panel_split import Panel
from plot_extractor.layout.plot_area import PlotArea


@dataclass
class TextRoiSet:
    """Text regions for OCR."""
    x_tick_band: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    y_tick_band_left: Optional[tuple[int, int, int, int]]
    y_tick_band_right: Optional[tuple[int, int, int, int]]
    legend_candidates: List[tuple[int, int, int, int]]


def propose_text_rois(panel: Panel, plot_area: PlotArea) -> TextRoiSet:
    """
    Propose text regions based on plot area.

    Regions:
    - X tick band: below plot area (50px height)
    - Y tick band left: left of plot area (80px width)
    - Y tick band right: right of plot area (80px width)
    - Legend candidates:右上、右侧、图内空白

    Args:
        panel: Panel containing the chart
        plot_area: Detected plot area

    Returns:
        TextRoiSet with proposed regions
    """
    h, w = panel.image.shape[:2]
    x0, y0, x1, y1 = plot_area.rect

    # X tick band: plot_area下方50px
    x_tick_band = (x0, y1 + 5, x1, min(h, y1 + 50))

    # Y tick band left: plot_area左侧80px
    if x0 > 80:
        y_tick_band_left = (max(0, x0 - 80), y0, x0 - 5, y1)
    else:
        y_tick_band_left = None

    # Y tick band right: plot_area右侧80px
    if x1 < w - 80:
        y_tick_band_right = (x1 + 5, y0, min(w, x1 + 80), y1)
    else:
        y_tick_band_right = None

    # Legend candidates
    legend_candidates = _detect_legend_candidates(panel, plot_area)

    return TextRoiSet(
        x_tick_band=x_tick_band,
        y_tick_band_left=y_tick_band_left,
        y_tick_band_right=y_tick_band_right,
        legend_candidates=legend_candidates,
    )


def _detect_legend_candidates(panel: Panel, plot_area: PlotArea) -> List[tuple]:
    """
    Detect potential legend regions.

    Methods:
    -右上角
    -右侧空白
    -图内空白区域（背景色）
    """
    h, w = panel.image.shape[:2]
    x0, y0, x1, y1 = plot_area.rect

    candidates = []

    # Candidate 1:右上角
    top_right = (x1 + 10, y0 - 60, min(w, x1 + 150), y0 - 10)
    if _is_valid_roi(top_right, panel.image):
        candidates.append(top_right)

    # Candidate 2:右侧
    right_side = (x1 + 10, y0 + 10, min(w, x1 + 150), y1 - 10)
    if _is_valid_roi(right_side, panel.image):
        candidates.append(right_side)

    # Candidate 3:图内空白区域（TODO:检测背景色块）

    return candidates


def _is_valid_roi(roi: tuple, image: np.ndarray) -> bool:
    """Check if ROI is valid (non-empty, reasonable size)."""
    x0, y0, x1, y1 = roi
    h, w = image.shape[:2]

    if x1 <= x0 or y1 <= y0:
        return False

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return False

    width = x1 - x0
    height = y1 - y0

    # Must be at least 30x30
    return width >= 30 and height >= 30
