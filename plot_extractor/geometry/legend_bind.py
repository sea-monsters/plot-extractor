"""Legend parsing and series binding.

Matches legend entries to extracted series by color distance and marker shape.
"""
from typing import List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class LegendEntry:
    """Single legend entry with label and color sample."""
    label: str
    sample_roi: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    color_rgb: tuple[int, int, int]  # Dominant color in sample_roi


def parse_legend_entries(legend_roi: np.ndarray) -> List[LegendEntry]:
    """
    Parse legend region to extract label + color sample pairs.

    Strategy:
    - Detect colored rectangles/markers on left side of each row
    - OCR text on right side of each row
    - Extract dominant color from sample ROI

    Args:
        legend_roi: RGB image region containing legend

    Returns:
        List of LegendEntry objects
    """
    # TODO: Implement legend parsing via connected components + OCR
    # Placeholder: return empty list for now
    return []


def bind_legend_to_series(
    legend_entries: List[LegendEntry],
    series_colors: Dict[str, tuple[int, int, int]],
) -> Dict[str, str]:
    """
    Match legend entries to series by color proximity.

    Args:
        legend_entries: List of LegendEntry with label and color
        series_colors: Dict mapping series_name to representative color

    Returns:
        Dict mapping series_name to legend_label
    """
    if not legend_entries or not series_colors:
        return {}

    bindings = {}
    for series_name, series_color in series_colors.items():
        best_match = None
        best_distance = float('inf')

        for entry in legend_entries:
            # Euclidean distance in RGB space
            distance = np.sqrt(
                (series_color[0] - entry.color_rgb[0]) ** 2 +
                (series_color[1] - entry.color_rgb[1]) ** 2 +
                (series_color[2] - entry.color_rgb[2]) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best_match = entry.label

        if best_match and best_distance < 100:  # Threshold: ~0.4 in normalized RGB
            bindings[series_name] = best_match

    return bindings
