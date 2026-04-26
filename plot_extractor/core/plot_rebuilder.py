"""Rebuild a plot from extracted data for validation."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_extractor.core.axis_calibrator import CalibratedAxis


def rebuild_plot(data_dict: Dict[str, Dict], calibrated_axes: List[CalibratedAxis],
                 output_path: Path, figsize=(6, 4), dpi=100,
                 is_scatter: bool = False, has_grid: bool = True):
    """Rebuild a matplotlib plot from extracted data."""
    x_cals = [ca for ca in calibrated_axes if ca.axis.direction == "x"]
    y_cals = [ca for ca in calibrated_axes if ca.axis.direction == "y"]

    x_cal = next((ca for ca in x_cals if ca.axis.side == "bottom"), x_cals[0] if x_cals else None)
    y_cal_left = next((ca for ca in y_cals if ca.axis.side == "left"), y_cals[0] if y_cals else None)
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
