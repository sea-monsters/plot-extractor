"""Rebuild a plot from extracted data for validation."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_extractor.core.axis_calibrator import CalibratedAxis


def _axis_quality_score(ca: CalibratedAxis) -> float:
    score = 10.0 if ca.axis.side in ("bottom", "left") else 0.0
    score += len(ca.tick_map or []) * 2.0
    score += ca.formula_anchor_count * 6.0
    score += ca.tesseract_anchor_count * 2.0
    if ca.tick_source == "formula_generated":
        score += 35.0
    elif ca.tick_source == "formula":
        score += 28.0
    elif ca.tick_source == "fused":
        score += 22.0
    elif ca.tick_source == "tesseract":
        score += 8.0
    elif ca.tick_source == "heuristic":
        score -= 8.0
    if ca.residual < 1:
        score += 20.0
    elif ca.residual < 10:
        score += 14.0
    elif ca.residual < 100:
        score += 6.0
    elif ca.residual > 1e5:
        score -= 20.0
    return score


def _choose_axis_with_primary_hysteresis(cals: List[CalibratedAxis], primary_sides: set[str]):
    """Keep bottom/left axes unless a secondary axis clearly wins."""
    if not cals:
        return None
    best = max(cals, key=_axis_quality_score)
    primary = [ca for ca in cals if ca.axis.side in primary_sides]
    if not primary or best.axis.side in primary_sides:
        return best
    primary_best = max(primary, key=_axis_quality_score)
    margin = 30.0
    if best.axis_type != primary_best.axis_type:
        margin += 15.0
    if len(primary_best.tick_map or []) >= 2:
        margin += 10.0
    if _axis_quality_score(best) < _axis_quality_score(primary_best) + margin:
        return primary_best
    return best


def rebuild_plot(data_dict: Dict[str, Dict], calibrated_axes: List[CalibratedAxis],
                 output_path: Path, figsize=(6, 4), dpi=100,
                 is_scatter: bool = False, has_grid: bool = True):
    """Rebuild a matplotlib plot from extracted data."""
    x_cals = [ca for ca in calibrated_axes if ca.axis.direction == "x"]
    y_cals = [ca for ca in calibrated_axes if ca.axis.direction == "y"]

    x_cal = _choose_axis_with_primary_hysteresis(x_cals, {"bottom"})
    y_cal_left = _choose_axis_with_primary_hysteresis(y_cals, {"left"})
    y_cal_right = next((ca for ca in y_cals if ca.axis.side == "right"), None)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    colors = ["blue", "red", "green", "orange", "purple", "brown", "teal", "navy"]

    for idx, (name, series) in enumerate(data_dict.items()):
        x = np.array(series["x"])
        y = np.array(series["y"])
        color = colors[idx % len(colors)]
        if is_scatter:
            ax.plot(x, y, 'o', color=color, markersize=6, label=name)
        else:
            ax.plot(x, y, color=color, linewidth=2, label=name)

    # Set axis scales
    if x_cal and x_cal.axis_type == "log":
        ax.set_xscale("log")
    if y_cal_left and y_cal_left.axis_type == "log":
        ax.set_yscale("log")

    # Set limits from calibrator tick maps
    if x_cal and x_cal.tick_map:
        vals = [v for _, v in x_cal.tick_map]
        ax.set_xlim(min(vals), max(vals))
    if y_cal_left and y_cal_left.tick_map:
        vals = [v for _, v in y_cal_left.tick_map]
        ax.set_ylim(min(vals), max(vals))

    # Handle inverted axis
    if y_cal_left and y_cal_left.inverted:
        ax.invert_yaxis()

    # Add twin y-axis if detected
    if y_cal_right and y_cal_right.tick_map:
        ax2 = ax.twinx()
        if y_cal_right.axis_type == "log":
            ax2.set_yscale("log")
        vals_r = [v for _, v in y_cal_right.tick_map]
        ax2.set_ylim(min(vals_r), max(vals_r))
        if y_cal_right.inverted:
            ax2.invert_yaxis()

    # Add grid lines only if original had them
    if has_grid:
        if x_cal and x_cal.axis_type == "log" and y_cal_left and y_cal_left.axis_type == "log":
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        elif x_cal and x_cal.axis_type == "log":
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        elif y_cal_left and y_cal_left.axis_type == "log":
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        else:
            ax.grid(True, linestyle="--", alpha=0.5)

    if len(data_dict) > 1:
        ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    return output_path
