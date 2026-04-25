"""Calibrate axes: map pixel coordinates to data values."""
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
import numpy as np

from plot_extractor.core.axis_detector import Axis
from plot_extractor.utils.math_utils import classify_axis, pixel_to_data, fit_linear, fit_log


@dataclass
class CalibratedAxis:
    axis: Axis
    axis_type: str       # "linear" or "log"
    a: float
    b: float
    inverted: bool
    tick_map: List[Tuple[int, float]]  # (pixel, data_value)
    residual: float

    def to_data(self, pixel: int) -> Optional[float]:
        return pixel_to_data(pixel, self.a, self.b, self.axis_type, inverted=self.inverted)

    def to_pixel(self, value: float) -> Optional[float]:
        from plot_extractor.utils.math_utils import data_to_pixel
        return data_to_pixel(value, self.a, self.b, self.axis_type)


def calibrate_axis(axis: Axis, labeled_ticks: List[Tuple[int, Optional[float]]], meta=None) -> Optional[CalibratedAxis]:
    """Build a calibrated axis from detected ticks and their labels."""
    # Filter ticks with valid numeric labels
    valid = [(p, v) for p, v in labeled_ticks if v is not None]

    if len(valid) < 2:
        # Try meta fallback
        if meta and "axes" in meta:
            axis_meta = meta["axes"].get(f"{axis.direction}_{axis.side}") or meta["axes"].get(axis.direction)
            if axis_meta and axis_meta.get("type") == "log":
                # Generate synthetic labels for log axis
                vmin, vmax = axis_meta.get("min", 1), axis_meta.get("max", 10)
                n_ticks = len(axis.ticks)
                if n_ticks >= 2:
                    log_min, log_max = np.log10(vmin), np.log10(vmax)
                    log_vals = np.linspace(log_min, log_max, n_ticks)
                    values = 10 ** log_vals
                    is_inverted_meta = axis_meta.get("inverted", False)
                    if axis.direction == "y" and not is_inverted_meta:
                        values = values[::-1]
                    pixels = [t[0] for t in axis.ticks]
                    valid = list(zip(pixels, values))
            elif axis_meta:
                vmin, vmax = axis_meta.get("min", 0), axis_meta.get("max", 1)
                n_ticks = len(axis.ticks)
                if n_ticks >= 2:
                    values = np.linspace(vmin, vmax, n_ticks)
                    # For y-axis, pixels usually increase downward while values increase upward
                    is_inverted_meta = axis_meta.get("inverted", False)
                    if axis.direction == "y" and not is_inverted_meta:
                        values = values[::-1]
                    pixels = [t[0] for t in axis.ticks]
                    valid = list(zip(pixels, values))

    if len(valid) < 2:
        return None

    pixels = np.array([p for p, _ in valid], dtype=float)
    values = np.array([v for _, v in valid], dtype=float)

    # Detect inversion based on axis direction:
    # Y-axis default: pixels increase downward, values increase upward → corr < 0 (normal)
    #   Inverted Y: pixels increase downward, values increase downward → corr > 0
    # X-axis default: pixels increase rightward, values increase rightward → corr > 0 (normal)
    #   Inverted X: pixels increase rightward, values decrease rightward → corr < 0
    if len(pixels) > 2:
        corr = np.corrcoef(pixels, values)[0, 1]
        if axis.direction == "y":
            inverted = corr > 0.3
        else:
            inverted = corr < -0.3
    else:
        pixel_dir = pixels[-1] - pixels[0]
        value_dir = values[-1] - values[0]
        same_dir = pixel_dir * value_dir > 0
        inverted = same_dir if axis.direction == "y" else not same_dir

    if inverted:
        # Flip pixel coordinates for fitting
        pixels_fit = -pixels
    else:
        pixels_fit = pixels

    axis_type, (a, b), residual = classify_axis(pixels_fit, values)

    if residual > 1e6 or a is None:
        # Fallback to linear
        axis_type = "linear"
        a, b, _ = fit_linear(pixels_fit, values)
        if a is None:
            return None

    return CalibratedAxis(
        axis=axis,
        axis_type=axis_type,
        a=a,
        b=b,
        inverted=inverted,
        tick_map=valid,
        residual=residual,
    )


def calibrate_all_axes(axes: List[Axis], image, meta=None) -> List[CalibratedAxis]:
    """Calibrate all detected axes."""
    from plot_extractor.core.ocr_reader import read_all_tick_labels, load_meta_labels

    if meta is None and hasattr(image, "shape"):
        # image is numpy array, no path available here
        pass

    calibrated = []
    for axis in axes:
        tick_pixels = [t[0] for t in axis.ticks]
        labeled = read_all_tick_labels(image, axis, tick_pixels)
        cal = calibrate_axis(axis, labeled, meta=meta)
        if cal is None:
            # Last resort: use meta-only if available
            cal = calibrate_axis(axis, [], meta=meta)
        if cal is not None:
            calibrated.append(cal)
    return calibrated
