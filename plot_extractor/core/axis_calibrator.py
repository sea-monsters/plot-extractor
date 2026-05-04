"""Calibrate axes: map pixel coordinates to data values."""
import os
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

import numpy as np

from plot_extractor.core.axis_detector import Axis
from plot_extractor.core.scale_detector import should_treat_as_log
from plot_extractor.utils.math_utils import (
    pixel_to_data,
    fit_linear,
    fit_linear_ransac,
    fit_log_ransac,
)


@dataclass
class CalibrationResult:
    """Result of multi-hypothesis axis calibration."""

    model_type: str  # "linear", "log", "polynomial"
    params: tuple  # fit coefficients (a, b) or (c, a, b)
    inlier_count: int
    residual: float
    is_plausible: bool
    inlier_indices: list[int] = None  # indices into original tick_map


def is_calibration_plausible(
    model_type: str, params: tuple, inlier_count: int
) -> bool:
    """Gate calibration results on physical plausibility."""
    if inlier_count < 3:
        return False

    if model_type == "linear":
        if len(params) < 2:
            return False
        a = params[0]
        if abs(a) > 1e6:
            return False
        if abs(a) < 1e-6:
            return False
        return True

    if model_type == "log":
        if len(params) < 2:
            return False
        a = params[0]
        if abs(a) < 1e-6:
            return False
        # ``a`` is pixels per decade for the log fit.  Values below this
        # range usually mean a huge linear numeric ramp was force-fit as log.
        if abs(a) < 15.0 or abs(a) > 1000.0:
            return False
        return True

    if model_type == "polynomial":
        if len(params) < 3:
            return False
        curvature = abs(params[0])
        if curvature > 1.0:
            return False
        effective_slope = abs(params[1])
        if effective_slope < 1e-6:
            return False
        return True

    return False


def grade_tick_quality(
    ocr_confidence: float = 70.0,
    crop_height: int = 20,
    median_crop_height: float = 20.0,
    position_deviation: float = 0.0,
) -> float:
    """Score a tick label's reliability on [0, 1]."""
    conf_score = min(ocr_confidence / 100.0, 1.0)
    height_ratio = min(crop_height / max(median_crop_height, 1.0), 1.5)
    height_score = max(0.0, min(1.0 - abs(1.0 - height_ratio) * 0.5, 1.0))
    pos_score = max(0.0, 1.0 - position_deviation / 20.0)
    return conf_score * (0.5 + 0.15 * height_score + 0.25 * pos_score)


def _detect_decade_width_and_boundaries(
    pixels: list[float],
) -> tuple[float | None, list[int]]:
    """Detect decade width and boundary positions from tick spacing.

    In TMLOG-style log axes, decade boundaries show as large spacing gaps
    (e.g. 9→10 is much wider than 1→2).  Returns (decade_width, boundary_indices).
    """
    if len(pixels) < 3:
        return None, []
    spacings = np.diff(pixels)
    median_s = float(np.median(spacings))
    if median_s <= 0:
        return None, []
    total_span = float(pixels[-1] - pixels[0])

    def _try_threshold(threshold_mult: float) -> tuple[float | None, list[int]]:
        boundaries = [i for i, s in enumerate(spacings) if s > median_s * threshold_mult]
        if len(boundaries) >= 2:
            decade_widths = [spacings[b] for b in boundaries]
            decade_width = float(np.median(decade_widths))
            if total_span / decade_width <= 8.0:
                return decade_width, boundaries
            # Implied decades > 8: try periodicity
            intervals = np.diff(boundaries)
            if len(intervals) > 0:
                from collections import Counter
                interval_counts = Counter([int(i) for i in intervals if i >= 3])
                mode_interval = interval_counts.most_common(1)
                if mode_interval and mode_interval[0][1] >= 2:
                    ticks_per_decade = mode_interval[0][0]
                    n_decades = max(1, round(len(spacings) / ticks_per_decade))
                    return total_span / n_decades, boundaries
            return total_span / 4.0, boundaries
        if len(boundaries) == 1:
            decade_width = float(spacings[boundaries[0]])
            if total_span / decade_width <= 8.0:
                return decade_width, boundaries
            return total_span / 4.0, boundaries
        return None, []

    # Primary: strict threshold
    result = _try_threshold(1.5)
    if result[0] is not None:
        return result
    # Secondary: relaxed threshold for minor-tick axes
    result = _try_threshold(1.2)
    if result[0] is not None:
        return result
    # No boundaries found — check for uniform/near-uniform spacing.
    mean_s = float(np.mean(spacings))
    if mean_s > 0 and float(np.std(spacings)) / mean_s < 0.15:
        decade_width = mean_s
        if total_span / decade_width <= 6.0:
            return decade_width, []
    # Ambiguous — estimate decades from tick count (typical log axis has
    # 2–9 ticks per decade depending on minor ticks and grid lines).
    estimated_decades = max(2, round(len(spacings) / 5.0))
    return total_span / estimated_decades, []


def _estimate_decade_width_from_anchors(
    anchors: list[tuple[float, float]],
) -> float | None:
    """Compute decade width from anchor pairs with the most consistent spacing.

    Returns the median decade width across all anchor pairs, or None if
    fewer than 2 positive anchors are available.
    """
    positive = [(float(p), float(v)) for p, v in anchors if v is not None and v > 0]
    if len(positive) < 2:
        return None
    estimates = []
    for i in range(len(positive)):
        for j in range(i + 1, len(positive)):
            p1, v1 = positive[i]
            p2, v2 = positive[j]
            pix_diff = abs(p2 - p1)
            log_diff = abs(np.log10(v2) - np.log10(v1))
            if log_diff > 1e-3 and pix_diff > 1e-3:
                estimates.append(pix_diff / log_diff)
    if not estimates:
        return None
    return float(np.median(estimates))


def infer_log_values_from_spacing(
    pixels: list[float],
    anchor_pixel: float | None = None,
    anchor_value: float | None = None,
    anchors: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]] | None:
    """Infer log tick values from pixel spacing using TMLOG decade fingerprint.

    Two-phase approach:
    1. Detect decade width from spacing pattern (large gaps = boundaries).
    2. If an anchor (pixel, value) is provided, calibrate absolute scale;
       otherwise assign synthetic 10^n values starting at the first tick.

    This is robust to missing or extra intra-decade ticks because it uses
    decade boundaries (large gaps) as structural anchors rather than
    requiring a perfect 1-2-3-...-9-10 tick sequence.
    """
    if len(pixels) < 2:
        return None
    pixels_sorted = np.array(sorted(pixels), dtype=float)
    total_span = float(pixels_sorted[-1] - pixels_sorted[0])
    decade_width, boundaries = _detect_decade_width_and_boundaries(pixels_sorted)
    if decade_width is None or decade_width <= 0:
        return None

    # Cross-check with anchor-based decade width when multiple anchors available
    if anchors and len(anchors) >= 2:
        anchor_dw = _estimate_decade_width_from_anchors(anchors)
        if anchor_dw is not None and anchor_dw > 0:
            detected_decades = total_span / decade_width
            anchor_decades = total_span / anchor_dw
            if anchor_decades < detected_decades:
                decade_width = anchor_dw

    # Guard: without boundaries or anchor, uniform spacing with many ticks is
    # almost certainly linear spacing misidentified as log decades.  High CV
    # suggests log minor ticks (decreasing spacings within each decade).
    if not boundaries and anchor_pixel is None and len(pixels_sorted) > 6:
        spacings = np.diff(pixels_sorted)
        cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
        if cv < 0.15:
            return None

    # Decade offset: what power-of-10 does the first tick correspond to?
    if anchor_pixel is not None and anchor_value is not None and anchor_value > 0:
        anchor_idx = int(np.argmin(np.abs(pixels_sorted - anchor_pixel)))
        anchor_px = float(pixels_sorted[anchor_idx])
        n_decades_before = sum(1 for b in boundaries if pixels_sorted[b + 1] <= anchor_px)
        intra = (anchor_px - pixels_sorted[0] - n_decades_before * decade_width) / decade_width
        decade_offset = np.log10(anchor_value) - (n_decades_before + intra)
    else:
        # P1 mode: no single anchor passed, but if we have any anchors at all,
        # solve for decade_offset via consensus (median + inlier filter).
        # This fixes the "always starts at 1.0" problem when OCR gives even
        # 1-2 noisy reads.
        decade_offset = 0.0
        if anchors:
            offsets = []
            for ap, av in anchors:
                if av is None or av <= 0:
                    continue
                anchor_idx = int(np.argmin(np.abs(pixels_sorted - ap)))
                anchor_px = float(pixels_sorted[anchor_idx])
                n_decades_before = sum(1 for b in boundaries if pixels_sorted[b + 1] <= anchor_px)
                intra = (anchor_px - pixels_sorted[0] - n_decades_before * decade_width) / decade_width
                expected_log10 = n_decades_before + intra
                offsets.append(float(np.log10(av) - expected_log10))
            if offsets:
                offsets_arr = np.array(offsets)
                median_off = float(np.median(offsets_arr))
                inliers = np.abs(offsets_arr - median_off) < 0.5
                n_inliers = int(np.sum(inliers))
                if n_inliers >= 2:
                    decade_offset = float(np.median(offsets_arr[inliers]))
                elif n_inliers == 1 and len(offsets) == 1:
                    # Single anchor: only trust it if it is a clean power of ten
                    # (OCR errors on non-power-of-ten labels are common)
                    single_val = [
                        av for ap, av in anchors
                        if av is not None and av > 0
                        and abs(np.log10(av) - round(np.log10(av))) < 0.05
                    ]
                    if single_val:
                        decade_offset = median_off
                    else:
                        decade_offset = 0.0
                elif n_inliers >= 1:
                    decade_offset = float(np.median(offsets_arr[inliers]))
                else:
                    decade_offset = 0.0

    result = []
    for p in pixels_sorted:
        n_decades = sum(1 for b in boundaries if pixels_sorted[b + 1] <= p)
        fraction = (p - pixels_sorted[0] - n_decades * decade_width) / decade_width
        log10_value = decade_offset + n_decades + fraction
        result.append((int(p), float(10.0 ** log10_value)))
    return result


def fit_axis_multi_hypothesis(
    tick_map: list[tuple[float, float]],
    preferred_type: str | None = None,
) -> CalibrationResult | None:
    """Fit multiple axis models and return the best plausible one.

    Tries linear, log, and polynomial fits simultaneously. Selects
    the model with the most inliers and lowest residual, subject to
    the physical plausibility gate.
    """
    if len(tick_map) < 3:
        return None

    pixels = np.array([p for p, _ in tick_map], dtype=float)
    values = np.array([v for _, v in tick_map], dtype=float)

    candidates: list[CalibrationResult] = []

    # Linear RANSAC
    a_lin, b_lin, res_lin, inliers_lin = fit_linear_ransac(pixels, values)
    if a_lin is not None and len(inliers_lin) >= 2:
        plausible = is_calibration_plausible("linear", (a_lin, b_lin), len(inliers_lin))
        candidates.append(CalibrationResult(
            model_type="linear",
            params=(float(a_lin), float(b_lin)),
            inlier_count=len(inliers_lin),
            residual=float(res_lin),
            is_plausible=plausible,
            inlier_indices=list(inliers_lin),
        ))

    # Log RANSAC
    a_log, b_log, res_log, inliers_log = fit_log_ransac(pixels, values)
    if a_log is not None and len(inliers_log) >= 2:
        positive_values = values[values > 0]
        if len(positive_values) >= 2:
            decade_span = np.log10(positive_values.max()) - np.log10(positive_values.min())
            # Real log axes can span < 1 decade (e.g. 1–5 ≈ 0.7 decades).
            # Require at least 0.3 decades to have any discriminative power.
            log_span_ok = 0.3 <= decade_span <= 6.0
        else:
            log_span_ok = False
        plausible = bool(
            is_calibration_plausible("log", (a_log, b_log), len(inliers_log))
            and log_span_ok
        )
        candidates.append(CalibrationResult(
            model_type="log",
            params=(float(a_log), float(b_log)),
            inlier_count=len(inliers_log),
            residual=float(res_log),
            is_plausible=plausible,
            inlier_indices=list(inliers_log),
        ))

    if not candidates:
        return None

    # Select best: prefer plausible, then highest inlier count, then lowest residual.
    # When preferred_type is given, boost that model's priority so it wins ties.
    plausible = [c for c in candidates if c.is_plausible]
    pool = plausible if plausible else candidates

    def _sort_key(c: CalibrationResult):
        inlier_penalty = -c.inlier_count
        # If this is the preferred type and it's plausible, give it a
        # small residual bonus (lower effective residual) so it wins
        # ties against the non-preferred type.
        effective_residual = c.residual
        if preferred_type and c.model_type == preferred_type and c.is_plausible:
            effective_residual = c.residual * 0.5
        return (inlier_penalty, effective_residual)

    pool.sort(key=_sort_key)
    return pool[0]


def _is_plausible_ocr_tick_sequence(
    labeled_ticks: List[Tuple[int, Optional[float]]],
    preferred_type: str | None = None,
) -> bool:
    """Lightweight sanity check for OCR-derived tick values.

    Scatteract-style: only reject *obviously* broken reads.  RANSAC
    handles outlier rejection during fitting, so we keep more ticks
    than the previous strict validator.
    """
    valid = [(p, v) for p, v in labeled_ticks if v is not None]
    if len(valid) < 2:
        return True

    pixels = np.array([p for p, _ in valid], dtype=float)
    values = np.array([v for _, v in valid], dtype=float)

    if not np.all(np.isfinite(values)):
        return False
    if np.ptp(values) < 1e-9:
        return False

    # Log-preferred axes need at least 2 positive values; non-positive
    # entries are filtered by fit_log_ransac rather than rejected here.
    if preferred_type == "log":
        positive_count = int(np.sum(values > 0))
        if positive_count < 2:
            return False

    # OCR noise often creates many repeated values (e.g. 0,0,0,...).
    unique_ratio = len(np.unique(np.round(values, 8))) / len(values)
    if unique_ratio < 0.5:
        return False

    # Monotonic trend check (relaxed: allow 40% outliers)
    order = np.argsort(pixels)
    ordered_values = values[order]
    diffs = np.diff(ordered_values)
    non_zero = diffs[np.abs(diffs) > 1e-9]
    if len(non_zero) >= 2:
        trend_consistency = max(np.mean(non_zero > 0), np.mean(non_zero < 0))
        if trend_consistency < 0.6:
            return False

    # Value range sanity check
    value_range = np.ptp(values)
    value_mean = np.mean(np.abs(values))
    if value_range > 1e6 and value_mean < 1000:
        # Extreme range but small mean: likely OCR garbage
        return False

    return True


def _snap_to_power_of_ten(v: float) -> float:
    """Snap OCR-noisy values to nearest power of ten when close enough.

    Tesseract often misreads '100' as '95', '102', '430', etc.
    If log10(v) is within 0.15 of an integer, snap to that power of 10.
    Skip the [10,19] and [100,110] ranges where superscript decoding
    (10^n) is the more likely interpretation.
    """
    if v <= 0:
        return v
    if (10 <= v <= 19) or (100 <= v <= 110):
        return v
    logv = np.log10(v)
    rounded = round(logv)
    if abs(logv - rounded) < 0.15:
        return float(10.0 ** rounded)
    return v


def _fix_log_superscript_ocr(
    labeled_ticks: List[Tuple[int, Optional[float]]],
) -> List[Tuple[int, Optional[float]]]:
    """Fix Tesseract misreads of superscript log labels (e.g. 10² → 102).

    Log axis labels like 10⁰, 10¹, 10² are often concatenated by OCR into
    '100', '101', '102', etc.  When MOST valid values fall in [100, 110]
    we convert 100+n → 10ⁿ.
    """
    valid = [(p, v) for p, v in labeled_ticks if v is not None]
    if len(valid) < 2:
        return labeled_ticks

    values = [v for _, v in valid]

    # Pattern 1: values like 100, 101, 102, 103... (superscript concatenation)
    in_hundreds = [v for v in values if 100 <= v <= 110]
    if len(in_hundreds) >= max(2, len(values) * 0.5):
        fixed = []
        for p, v in labeled_ticks:
            if v is not None and 100 <= v <= 110:
                exp = int(round(v - 100))
                fixed.append((p, 10.0 ** exp))
            else:
                fixed.append((p, v))
        return fixed

    # Pattern 2: values like 10, 11, 12, 13... (missing superscript)
    in_tens = [v for v in values if 10 <= v <= 19]
    if len(in_tens) >= max(2, len(values) * 0.5):
        fixed = []
        for p, v in labeled_ticks:
            if v is not None and 10 <= v <= 19:
                exp = int(round(v - 10))
                fixed.append((p, 10.0 ** exp))
            else:
                fixed.append((p, v))
        return fixed

    return labeled_ticks


def _force_log_superscript_ocr(
    labeled_ticks: List[Tuple[int, Optional[float]]],
) -> List[Tuple[int, Optional[float]]]:
    """Rewrite common Tesseract superscript flattening once log notation is confirmed.

    Unlike ``_fix_log_superscript_ocr`` this does not require a majority
    pattern.  It is used only after FormulaOCR has already provided log
    evidence on the same axis, so every 100+n / 10+n label can be interpreted
    as 10^n at its original tick anchor.
    """
    fixed = []
    for pixel, value in labeled_ticks:
        if value is None:
            fixed.append((pixel, value))
            continue
        if 100 <= value <= 110:
            fixed.append((pixel, 10.0 ** int(round(value - 100))))
        elif 10 <= value <= 19:
            fixed.append((pixel, 10.0 ** int(round(value - 10))))
        elif value > 0:
            log10 = np.log10(value)
            if abs(log10 - round(log10)) < 0.02:
                fixed.append((pixel, value))
            else:
                fixed.append((pixel, None))
        else:
            fixed.append((pixel, None))
    return fixed


@dataclass
class CalibratedAxis:
    axis: Axis
    axis_type: str       # "linear" or "log"
    a: float
    b: float
    inverted: bool
    tick_map: List[Tuple[int, float]]  # (pixel, data_value)
    residual: float
    tick_source: str = "heuristic"
    anchor_count: int = 0
    formula_anchor_count: int = 0
    formula_log_score: float = 0.0
    formula_selected_count: int = 0
    tesseract_anchor_count: int = 0
    label_anchor_count: int = 0
    formula_batch_candidate_count: int = 0
    formula_batch_kept_count: int = 0
    formula_batch_chunks: int = 0
    formula_batch_requested: int = 0
    formula_batch_returned: int = 0
    formula_batch_ms: float = 0.0
    debug_trace: dict | None = None

    def to_data(self, pixel: int) -> Optional[float]:
        return pixel_to_data(pixel, self.a, self.b, self.axis_type, inverted=self.inverted)

    def to_pixel(self, value: float) -> Optional[float]:
        from plot_extractor.utils.math_utils import data_to_pixel  # pylint: disable=import-outside-toplevel
        return data_to_pixel(value, self.a, self.b, self.axis_type)


def calibrate_axis(
    axis: Axis,
    labeled_ticks: List[Tuple[int, Optional[float]]],
    image=None,
    policy=None,
    preferred_type: str | None = None,
    is_log: bool | None = None,
    tick_source: str = "heuristic",
    anchor_count: int = 0,
    formula_anchor_count: int = 0,
    formula_log_score: float = 0.0,
    formula_selected_count: int = 0,
    tesseract_anchor_count: int = 0,
    label_anchor_count: int = 0,
    formula_batch_candidate_count: int = 0,
    formula_batch_kept_count: int = 0,
    formula_batch_chunks: int = 0,
    formula_batch_requested: int = 0,
    formula_batch_returned: int = 0,
    formula_batch_ms: float = 0.0,
    apply_log_superscript_fix: bool = True,
) -> Optional[CalibratedAxis]:
    """Build a calibrated axis from detected ticks and their labels."""
    # Gate log superscript fix behind visual log-scale detection.
    # When is_log is pre-computed (cross-axis detection), skip internal check.
    if is_log is None and image is not None:
        is_log = should_treat_as_log(image, axis)
    if is_log and apply_log_superscript_fix:
        labeled_ticks = _fix_log_superscript_ocr(labeled_ticks)

    # Filter ticks with valid numeric labels
    valid = [(p, v) for p, v in labeled_ticks if v is not None]
    used_heuristic_fallback = False

    # For log-preferred axes, discard non-positive values (they break log fit)
    if preferred_type == "log":
        valid = [(p, v) for p, v in valid if v > 0]

    # Guardrail: if OCR values look internally inconsistent, ignore them
    # and let heuristic fallback take over.
    if len(valid) >= 3 and not _is_plausible_ocr_tick_sequence(valid, preferred_type):
        valid = []

    # For log-preferred axes with only 2 OCR ticks, the scale is under-
    # determined.  Reject clearly implausible pairings (identical values,
    # or a decade width that is far outside the 15–400 px/decade range
    # seen in real charts).  A decade width < 15 px/decade means the two
    # ticks are extremely close in log-space relative to pixel distance,
    # making the fit hypersensitive to OCR error.
    if preferred_type == "log" and len(valid) == 2:
        p1, v1 = valid[0]
        p2, v2 = valid[1]
        if v1 > 0 and v2 > 0:
            log_diff = abs(np.log10(v2) - np.log10(v1))
            pix_diff = abs(p2 - p1)
            if log_diff < 1e-3:
                valid = []
            else:
                decade_width = pix_diff / log_diff
                if decade_width < 15.0 or decade_width > 400.0:
                    valid = []

    if len(valid) < 2:
        # Heuristic fallback: generate synthetic values from tick spacing pattern.
        # Pass any available OCR anchors to TMLOG decade fingerprint inference
        # so that a single reliable FormulaOCR read can calibrate the entire axis.
        anchors = [(p, v) for p, v in labeled_ticks if v is not None]
        synthetic_valid = _build_heuristic_ticks(axis, anchors=anchors, preferred_type=preferred_type)
        if len(synthetic_valid) >= 2:
            valid = synthetic_valid
            used_heuristic_fallback = True

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
    elif len(pixels) == 2:
        pixel_dir = pixels[-1] - pixels[0]
        value_dir = values[-1] - values[0]
        same_dir = pixel_dir * value_dir > 0
        inverted = same_dir if axis.direction == "y" else not same_dir
    else:
        return None

    if inverted:
        # Flip pixel coordinates for fitting
        pixels_fit = -pixels
    else:
        pixels_fit = pixels

    # Use multi-candidate solver for axis mapping
    from plot_extractor.core.axis_candidates import solve_axis_multi_candidate  # pylint: disable=import-outside-toplevel

    candidate_result = solve_axis_multi_candidate(
        axis,
        [(int(p), float(v)) for p, v in zip(pixels_fit, values)],
        preferred_type=preferred_type,
    )

    best_candidate = candidate_result.best
    promoted_primary_log = False
    promoted_primary_log_reason = None
    if (
        preferred_type == "log"
        and axis.side in ("bottom", "left")
        and getattr(candidate_result, "all", None)
    ):
        log_candidates = [
            cand
            for cand in candidate_result.all
            if getattr(cand, "scale", None) == "log" and getattr(cand, "a", None) is not None
        ]
        if log_candidates:
            best_log = min(
                log_candidates,
                key=lambda cand: float(getattr(cand, "residual", 1e9) or 1e9),
            )
            best_residual = float(getattr(best_candidate, "residual", 1e9) or 1e9)
            best_log_residual = float(getattr(best_log, "residual", 1e9) or 1e9)
            current_is_unstable = (
                getattr(best_candidate, "scale", None) != "log"
                or best_residual > 1000.0
                or (
                    getattr(best_candidate, "source", None) == "ocr"
                    and len(valid) <= 3
                    and best_residual > 100.0
                )
            )
            # Main-axis robustness guard:
            # only promote when the log candidate is materially better,
            # so we don't override stable linear fits by soft routing noise.
            has_clear_log_advantage = best_log_residual <= max(5.0, best_residual * 0.35)
            if current_is_unstable and has_clear_log_advantage:
                best_candidate = best_log
                promoted_primary_log = True
                promoted_primary_log_reason = "preferred_log_primary_axis_guard"

    axis_type = best_candidate.scale
    a = best_candidate.a
    b = best_candidate.b
    residual = best_candidate.residual

    if residual > 1e6 or a is None:
        # Fallback to linear
        axis_type = "linear"
        a, b, _ = fit_linear(pixels_fit, values)
        if a is None:
            return None

    debug_trace = {
        "preferred_type": preferred_type,
        "is_log_input": bool(is_log) if is_log is not None else None,
        "valid_tick_count": int(len(valid)),
        "used_heuristic_fallback": bool(used_heuristic_fallback),
        "input_labeled_count": int(sum(1 for _p, v in labeled_ticks if v is not None)),
        "candidate_best_scale": getattr(best_candidate, "scale", None),
        "candidate_best_source": getattr(best_candidate, "source", None),
        "candidate_best_confidence": float(getattr(best_candidate, "confidence", 0.0)),
        "candidate_primary_log_promoted": promoted_primary_log,
        "candidate_primary_log_reason": promoted_primary_log_reason,
    }

    return CalibratedAxis(
        axis=axis,
        axis_type=axis_type,
        a=a,
        b=b,
        inverted=inverted,
        tick_map=valid,
        residual=residual,
        tick_source="heuristic" if used_heuristic_fallback else tick_source,
        anchor_count=anchor_count,
        formula_anchor_count=formula_anchor_count,
        formula_log_score=formula_log_score,
        formula_selected_count=formula_selected_count,
        tesseract_anchor_count=tesseract_anchor_count,
        label_anchor_count=label_anchor_count,
        formula_batch_candidate_count=formula_batch_candidate_count,
        formula_batch_kept_count=formula_batch_kept_count,
        formula_batch_chunks=formula_batch_chunks,
        formula_batch_requested=formula_batch_requested,
        formula_batch_returned=formula_batch_returned,
        formula_batch_ms=formula_batch_ms,
        debug_trace=debug_trace,
    )


def _anchor_aligns_with_spacing(
    anchor_pixel: float,
    anchor_value: float,
    pixels: np.ndarray,
    spacings: np.ndarray,
) -> bool:
    """Check if a power-of-ten anchor aligns with detected decade boundaries.

    Log decade boundaries appear as large spacing gaps.  If the anchor
    value is a clean power of ten, its pixel position should be close to
    one of those boundaries.  Misaligned anchors are usually OCR errors
    (e.g. 40 read as 430) and should be rejected.
    """
    # TEMPORARILY DISABLED: causes regression on log_y Y-axis calibration
    # where decade boundaries are detected differently than X-axis.
    # TODO: re-enable with axis-direction-aware boundary logic.
    return True
    # if anchor_value <= 0:
    #     return False
    # logv = np.log10(anchor_value)
    # if abs(logv - round(logv)) >= 0.05:
    #     return True  # non-power-of-ten: cannot validate geometrically
    # median_s = float(np.median(spacings))
    # if median_s <= 0:
    #     return True
    # boundaries = [i for i, s in enumerate(spacings) if s > median_s * 1.5]
    # if not boundaries:
    #     return True
    # boundary_pixels = [float(pixels[b + 1]) for b in boundaries]
    # nearest_dist = min(abs(bp - anchor_pixel) for bp in boundary_pixels)
    # decade_width = float(np.median([spacings[b] for b in boundaries]))
    # return nearest_dist < decade_width * 0.25


def _build_heuristic_ticks(axis, anchors=None, preferred_type=None):
    """Generate synthetic (pixel, value) ticks from spacing pattern when OCR fails.

    Uses tick spacing uniformity to guess linear vs log, then assigns
    synthetic values.  When ``anchors`` (list of (pixel, value)) are
    provided, they are fed to the TMLOG decade-fingerprint inference as
    absolute calibration points instead of falling back to arbitrary
    synthetic values.
    """
    ticks = sorted([t[0] for t in (axis.ticks or [])])
    if len(ticks) < 2:
        return []

    pixels = np.array(ticks, dtype=float)
    spacings = np.diff(pixels)
    if len(spacings) == 0 or np.mean(spacings) <= 0:
        return []

    # Coefficient of variation: low = uniform (linear), high = geometric (log)
    cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))

    # Geometric indicator: ratios of consecutive spacings
    ratios = []
    for i in range(1, len(spacings)):
        if spacings[i - 1] > 1:
            ratios.append(spacings[i] / spacings[i - 1])
    mean_ratio = float(np.mean(ratios)) if ratios else 1.0

    # Classify axis type from spacing pattern
    is_log = cv > 0.3 and 0.5 < mean_ratio < 2.0 and not 0.9 < mean_ratio < 1.1
    if preferred_type == "log":
        is_log = True

    n = len(ticks)
    if is_log:
        # P0: Try TMLOG decade fingerprint inference with OCR anchors first
        if anchors:
            # Prefer FormulaOCR anchors (typically more reliable)
            sorted_anchors = sorted(
                anchors, key=lambda a: (
                    0 if a[1] is not None and a[1] > 0 and
                    (100 <= a[1] <= 110 or 10 <= a[1] <= 19 or
                     (a[1] > 0 and abs(np.log10(a[1]) - round(np.log10(a[1]))) < 0.02))
                    else 1,
                    a[0]
                )
            )
            for anchor_pixel, anchor_value in sorted_anchors:
                if anchor_value is not None and anchor_value > 0:
                    if not _anchor_aligns_with_spacing(
                        float(anchor_pixel), float(anchor_value), pixels, spacings
                    ):
                        continue
                    inferred = infer_log_values_from_spacing(
                        ticks, anchor_pixel=float(anchor_pixel), anchor_value=float(anchor_value),
                        anchors=[(float(p), float(v)) for p, v in anchors if v is not None and v > 0],
                    )
                    if inferred is not None and len(inferred) >= 2:
                        if axis.direction == "y":
                            inferred = inferred[::-1]
                        min_val = max(min(v for _, v in inferred), 1e-9)
                        max_val = max(v for _, v in inferred)
                        decades = float(np.log10(max_val / min_val))
                        if decades <= 4.0:
                            return inferred
                        if decades <= 10.0:
                            scale = 4.0 / decades
                            rescaled = []
                            log_min = float(np.log10(min_val))
                            for p, v in inferred:
                                log_v = float(np.log10(max(v, 1e-9)))
                                new_log_v = (log_v - log_min) * scale + log_min
                                rescaled.append((p, float(10.0 ** new_log_v)))
                            inferred = rescaled
                            return inferred
                        # Span > 10 decades: treat as garbage, fall through to P1/P2

        # P1: TMLOG without anchors
        inferred = infer_log_values_from_spacing(ticks)
        if inferred is not None and len(inferred) >= 2:
            min_val = max(min(v for _, v in inferred), 1e-9)
            max_val = max(v for _, v in inferred)
            decades = float(np.log10(max_val / min_val))
            # If span is reasonable, use TMLOG inference.  Garbage tick
            # detection (high CV, no clear periodicity) produces inflated
            # decade counts; fall back to safe single-decade logspace.
            if decades <= 4.0:
                if axis.direction == "y":
                    inferred = inferred[::-1]
                return inferred
            # Span > 4 decades but ≤ 10: rescale to 4 decades to keep
            # some shape while avoiding catastrophic calibration.
            if decades <= 10.0:
                scale = 4.0 / decades
                rescaled = []
                log_min = float(np.log10(min_val))
                for p, v in inferred:
                    log_v = float(np.log10(max(v, 1e-9)))
                    new_log_v = (log_v - log_min) * scale + log_min
                    rescaled.append((p, float(10.0 ** new_log_v)))
                inferred = rescaled
                if axis.direction == "y":
                    inferred = inferred[::-1]
                return inferred
            # Span > 10 decades: treat as garbage, fall through to P2.

        # P2: Safe synthetic log values — np.logspace(0, 1, n) always
        # spans exactly one decade.  This is robust to garbage tick
        # detection because the scale factor is determined by the linear
        # fit to pixel span, not by tick-count heuristics.
        values = np.logspace(0, 1, n)
    else:
        # Synthetic linear values: 0, 1, 2, ..., n-1
        values = np.arange(n, dtype=float)

    # Handle inversion: for y-axis, pixels usually increase downward while
    # values increase upward.  Without external info, assume standard orientation.
    if axis.direction == "y":
        # Standard: top tick = largest value, bottom tick = smallest value
        values = values[::-1]

    return list(zip(ticks, values))


def _build_formula_generated_log_ticks(axis, anchors) -> List[Tuple[int, Optional[float]]]:
    """Generate a full log tick sequence from FormulaOCR exponent anchors.

    FormulaOCR values are authoritative, but one Formula label only fixes one
    absolute point.  With a single formula anchor, Tesseract is used only to
    infer additional exponent candidates such as ``100`` -> ``10^0`` or
    ``103`` -> ``10^3``; its numeric value is never used directly.
    """
    tick_pixels = sorted([int(t[0]) for t in (axis.ticks or [])])
    if len(tick_pixels) < 2:
        return []

    for anchor in anchors:
        formula_value = getattr(anchor, "formula_value", None)
        exponent = None
        if formula_value is not None and formula_value > 0:
            log10 = np.log10(formula_value)
            if abs(log10 - round(log10)) < 0.02:
                exponent = int(round(log10))
        anchor._formula_exponent = exponent  # pylint: disable=protected-access

    exponent_anchors = []
    for anchor in anchors:
        exponent = getattr(anchor, "_formula_exponent", None)
        if exponent is not None:
            exponent_anchors.append((int(anchor.tick_pixel), int(exponent), "formula"))

    for anchor in anchors:
        value = getattr(anchor, "tesseract_value", None)
        if value is None:
            continue
        exponent = None
        if 100 <= value <= 110:
            exponent = int(round(value - 100))
        elif 10 <= value <= 19:
            exponent = int(round(value - 10))
        elif value > 0:
            log10 = np.log10(value)
            if abs(log10 - round(log10)) < 0.02:
                exponent = int(round(log10))
        if exponent is not None:
            exponent_anchors.append((int(anchor.tick_pixel), int(exponent), "tesseract_hint"))

    formula_count = sum(1 for _p, _e, source in exponent_anchors if source == "formula")
    unique = {}
    for pixel, exponent, source in exponent_anchors:
        old = unique.get(pixel)
        if old is None or source == "formula":
            unique[pixel] = (exponent, source)
    exponent_anchors = [(pixel, exp, source) for pixel, (exp, source) in unique.items()]

    if formula_count < 1 or len(exponent_anchors) < 2:
        return []

    # Need at least two distinct exponent anchors to determine range/step.
    # With only one Formula anchor, the defining pair must include it; two
    # Tesseract hints alone are not allowed to determine the whole axis.
    pairs = []
    for i, (p1, e1, _s1) in enumerate(exponent_anchors):
        for p2, e2, s2 in exponent_anchors[i + 1:]:
            if p1 == p2 or e1 == e2:
                continue
            s1 = exponent_anchors[i][2]
            if formula_count < 2 and s1 != "formula" and s2 != "formula":
                continue
            pairs.append((p1, e1, s1, p2, e2, s2))
    if not pairs:
        return []

    ticks = np.array(tick_pixels, dtype=float)
    best = None
    best_error = float("inf")
    for p1, e1, _s1, p2, e2, _s2 in pairs:
        slope = (e2 - e1) / (p2 - p1)
        intercept = e1 - slope * p1
        formula_errors = []
        hint_errors = []
        for pixel, exponent, _source in exponent_anchors:
            error = abs((slope * pixel + intercept) - exponent)
            if _source == "formula":
                formula_errors.append(error)
            else:
                hint_errors.append(error)
        if formula_errors and max(formula_errors) > 0.2:
            continue
        if hint_errors and float(np.mean(hint_errors)) > 0.8:
            continue
        error = (
            float(np.mean(formula_errors)) * 2.0
            + (float(np.mean(hint_errors)) if hint_errors else 0.0)
        )
        if error < best_error:
            best_error = error
            best = (slope, intercept)

    if best is None or best_error > 0.9:
        return []

    slope, intercept = best
    generated_exponents = slope * ticks + intercept
    if not np.all(np.isfinite(generated_exponents)):
        return []
    if float(np.ptp(generated_exponents)) > 8.0:
        return []
    if axis.direction == "x" and slope < 0:
        return []
    if axis.direction == "y" and slope > 0:
        return []

    generated = []
    for pixel, exponent in zip(ticks, generated_exponents):
        generated.append((int(pixel), float(10.0 ** exponent)))
    return generated


def _crop_axis_region(image: np.ndarray, axis: Axis, padding: int = 20) -> np.ndarray:
    """Crop a region around an axis for LLM vision input."""
    h, w = image.shape[:2]
    if axis.direction == "x":
        # Horizontal axis: labels below
        x1 = max(0, int(axis.plot_start) - padding)
        x2 = min(w, int(axis.plot_end) + padding)
        y1 = max(0, int(axis.position) - padding)
        y2 = min(h, int(axis.position) + padding * 4)
    else:
        # Vertical axis: labels to the left or right
        y1 = max(0, int(axis.plot_start) - padding)
        y2 = min(h, int(axis.plot_end) + padding)
        if axis.side == "right":
            x1 = max(0, int(axis.position) - padding)
            x2 = min(w, int(axis.position) + padding * 5)
        else:
            x1 = max(0, int(axis.position) - padding * 5)
            x2 = min(w, int(axis.position) + padding)
    if x2 <= x1:
        x2 = min(w, x1 + 10)
    if y2 <= y1:
        y2 = min(h, y1 + 10)
    return image[y1:y2, x1:x2]


def _crop_tick_labels(
    image: np.ndarray, axis: Axis, max_crops: int = 4,
) -> list:
    """Crop individual tick-label regions around each tick position.

    Returns up to *max_crops* small image patches suitable for per-tick
    OCR with PP-FormulaNet.  Crops are sampled evenly across the axis.
    """
    h, w = image.shape[:2]
    tick_pixels = sorted([t[0] for t in (axis.ticks or [])])
    if not tick_pixels:
        return []

    # Sample evenly to cover the full range
    n = len(tick_pixels)
    if n <= max_crops:
        indices = list(range(n))
    else:
        step = max(1, (n - 1) / (max_crops - 1))
        indices = [int(i * step) for i in range(max_crops)]
        indices[-1] = min(indices[-1], n - 1)

    crops = []
    for idx in indices:
        tick_px = tick_pixels[idx]
        strip = 35  # label patch half-size
        if axis.direction == "x":
            x1 = max(0, tick_px - strip)
            x2 = min(w, tick_px + strip)
            y1 = max(0, axis.position + 5)
            y2 = min(h, axis.position + 55)
        else:
            y1 = max(0, tick_px - strip // 2)
            y2 = min(h, tick_px + strip // 2)
            if axis.side == "right":
                x1 = max(0, axis.position + 5)
                x2 = min(w, axis.position + 60)
            else:
                x1 = max(0, axis.position - 60)
                x2 = max(x1 + 10, axis.position - 5)
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def _align_formula_values_to_ticks(
    tick_pixels: list,
    formula_values: list,
    descending: bool = False,
) -> list:
    """Map FormulaOCR values to tick pixel positions as anchor points.

    FormulaOCR typically returns fewer values than there are tick positions.
    Instead of spreading values across all ticks (which creates duplicates
    that confuse RANSAC), we place each value at the tick whose position
    is closest to an evenly-spaced anchor grid.  Remaining ticks get None
    and are filled by the calibration fitter.
    """
    pixels = sorted(tick_pixels, reverse=descending)
    values = list(formula_values)
    n_pix = len(pixels)
    n_val = len(values)

    if n_val == 0:
        return [(int(p), None) for p in tick_pixels]

    if n_val >= n_pix:
        step = max(1, n_val / n_pix)
        return [(int(pixels[i]), float(values[int(i * step)])) for i in range(n_pix)]

    # Place M values at M anchor positions evenly spaced across the pixel range
    p_min, p_max = pixels[0], pixels[-1]
    if descending:
        p_min, p_max = p_max, p_min

    anchors = []
    for i in range(n_val):
        if n_val == 1:
            t = 0.5
        else:
            t = i / (n_val - 1)
        anchors.append(p_min + t * (p_max - p_min))

    # For each anchor, find the closest tick pixel
    used_indices = set()
    assignments = {}
    for val, anchor in zip(values, anchors):
        best_idx = min(range(n_pix), key=lambda j: abs(pixels[j] - anchor))
        if best_idx not in used_indices:
            used_indices.add(best_idx)
            assignments[best_idx] = float(val)

    # Build labeled list: assigned ticks get FormulaOCR values, rest get None
    labeled = []
    for i, p in enumerate(pixels):
        if i in assignments:
            labeled.append((int(p), assignments[i]))
        else:
            labeled.append((int(p), None))
    return labeled


def _llm_enhance_axis_labels(
    image: np.ndarray,
    axis: Axis,
    ocr_labeled: List[Tuple[int, Optional[float]]],
) -> Optional[List[Tuple[int, Optional[float]]]]:
    """Use LLM vision to read axis labels when OCR yields insufficient values."""
    from plot_extractor.core.llm_policy_router import (  # pylint: disable=import-outside-toplevel
        llm_available,
        _detect_provider,
        _image_array_to_data_url,
        llm_read_axis_labels,
    )

    if not llm_available():
        return None

    provider, base_url, model = _detect_provider()
    if not provider:
        return None

    tick_pixels = sorted([t[0] for t in axis.ticks])
    crop = _crop_axis_region(image, axis)
    image_data_url = _image_array_to_data_url(crop)

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    if not api_key:
        return None

    parsed = llm_read_axis_labels(
        image_data_url, provider, api_key, base_url, model,
        axis.direction, len(tick_pixels),
    )
    if not parsed or "tick_values" not in parsed:
        return None

    tick_values = parsed["tick_values"]
    if not isinstance(tick_values, list):
        return None

    # Normalize length to match detected ticks
    n = len(tick_pixels)
    if len(tick_values) < n:
        tick_values = tick_values + [None] * (n - len(tick_values))
    elif len(tick_values) > n:
        tick_values = tick_values[:n]

    # Convert to float where possible
    llm_labeled = []
    for pix, val in zip(tick_pixels, tick_values):
        if val is None:
            llm_labeled.append((pix, None))
        else:
            try:
                llm_labeled.append((pix, float(val)))
            except (ValueError, TypeError):
                llm_labeled.append((pix, None))

    # Merge: keep OCR values where available, fill gaps with LLM
    merged = []
    for (pix_ocr, val_ocr), (pix_llm, val_llm) in zip(ocr_labeled, llm_labeled):
        if val_ocr is not None:
            merged.append((pix_ocr, val_ocr))
        else:
            merged.append((pix_llm, val_llm))
    return merged


def _axis_relative_position(axis: Axis, tick_pixel: int) -> float:
    """Return normalized tick position along the axis span in [0, 1]."""
    span = max(1, int(axis.plot_end) - int(axis.plot_start))
    if axis.direction == "x":
        return float((tick_pixel - int(axis.plot_start)) / span)
    return float((tick_pixel - int(axis.plot_start)) / span)


def _score_formula_anchor_candidate(axis: Axis, anchor, anchor_idx: int, total: int) -> float:
    """Score how useful an anchor crop is for FormulaOCR."""
    if anchor is None or getattr(anchor, "crop", None) is None or anchor.crop.size == 0:
        return -1.0
    if getattr(anchor, "label_bbox", (0, 0, 0, 0)) == (0, 0, 0, 0):
        return -1.0

    score = float(getattr(anchor, "confidence", 0.0))
    value = getattr(anchor, "tesseract_value", None)
    text = (getattr(anchor, "tesseract_text", "") or "").strip()
    source = getattr(anchor, "source", "")

    if not _anchor_has_label_evidence(anchor):
        return -1.0

    if value is None:
        score += 2.2
    else:
        if 100 <= value <= 110 or 10 <= value <= 19:
            score += 2.5
        elif value == 0:
            score += 0.4
        else:
            score += 0.1

    if text:
        if any(ch in text for ch in ("^", "e", "E", "×", "x", "X", "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")):
            score += 2.0
        elif any(ch.isdigit() for ch in text):
            score += 0.4
    else:
        score += 0.5

    if source == "synthetic_ocr":
        if text and any(ch.isdigit() for ch in text):
            score += 0.6
        if text and any(ch in text for ch in ("^", "e", "E", "×", "x", "X", "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")):
            score += 1.5
        if value is None and not text:
            score -= 1.0

    # Prefer crops that cover the axis span instead of duplicates near one end.
    rel_pos = _axis_relative_position(axis, int(anchor.tick_pixel))
    if rel_pos <= 0.2 or rel_pos >= 0.8:
        score += 0.8
    elif 0.4 <= rel_pos <= 0.6:
        score += 0.3

    # Spread selection across the crop list to avoid all crops coming from one cluster.
    if total > 1:
        spread_bias = abs((anchor_idx / max(total - 1, 1)) - rel_pos)
        score += max(0.0, 0.4 - spread_bias)

    return score


def _anchor_has_label_evidence(anchor) -> bool:
    """Return True when an anchor represents a visible labeled tick.

    Dense axes can contain many unlabeled minor ticks.  Those ticks still
    matter geometrically, but they must not dilute axis-type evidence or be
    sent to FormulaOCR as if they carried text.
    """
    if anchor is None:
        return False
    if getattr(anchor, "tesseract_value", None) is not None:
        return True
    text = (getattr(anchor, "tesseract_text", "") or "").strip()
    if text and any(ch.isdigit() for ch in text):
        return True
    if getattr(anchor, "formula_value", None) is not None:
        return True
    latex = (getattr(anchor, "formula_latex", "") or "").strip()
    return bool(latex)


def _select_formula_anchor_indices(
    axis: Axis,
    anchors,
    max_crops: int,
) -> list[int]:
    """Choose a small, informative subset of anchor crops for FormulaOCR."""
    scored = []
    for idx, anchor in enumerate(anchors):
        score = _score_formula_anchor_candidate(axis, anchor, idx, len(anchors))
        if score > 0:
            scored.append((idx, score))
    if not scored:
        return []

    # Bucket by position to preserve coverage across the axis span.  This is
    # intentionally conservative: pure high-score selection can pair labels
    # that look OCR-friendly but imply the wrong axis direction.
    buckets = {0: [], 1: [], 2: []}
    for idx, score in scored:
        rel_pos = _axis_relative_position(axis, int(anchors[idx].tick_pixel))
        bucket = min(2, max(0, int(rel_pos * 3)))
        buckets[bucket].append((idx, score))

    selected: list[int] = []
    for bucket in (0, 1, 2):
        if not buckets[bucket]:
            continue
        best_idx = max(buckets[bucket], key=lambda item: item[1])[0]
        if best_idx not in selected:
            selected.append(best_idx)
        if len(selected) >= max_crops:
            return selected[:max_crops]

    for idx, _ in sorted(scored, key=lambda item: item[1], reverse=True):
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_crops:
            break

    return selected[:max_crops]


def _should_use_formula_label_values(
    formula_anchor_count: int,
    formula_value_count: int,
    formula_log_score: float,
) -> bool:
    """Decide when FormulaOCR is strong enough to override tesseract labels.

    A single formula read is useful as a hint, but it is not enough to
    replace a well-populated tesseract axis.  Requiring multiple formula
    values keeps the batch planner selective while still allowing sparse
    axes to fall back to FormulaOCR when tesseract is missing.
    """
    if formula_log_score < 0.3:
        return False
    if formula_anchor_count < 2:
        return False
    if formula_value_count < 2:
        return False
    return True


def _should_use_formula_log_hint(
    formula_anchor_count: int,
    formula_value_count: int,
    formula_log_score: float,
) -> bool:
    """Require repeated formula evidence before changing the axis scale."""
    return _should_use_formula_label_values(
        formula_anchor_count,
        formula_value_count,
        formula_log_score,
    )


def _candidate_calibration_score(
    cal: CalibratedAxis,
    valid_count: int,
    formula_log_score: float,
) -> float:
    """Rank per-source axis candidates without trusting one OCR source blindly."""
    score = valid_count * 8.0

    if cal.residual < 1:
        score += 45
    elif cal.residual < 10:
        score += 35
    elif cal.residual < 100:
        score += 20
    elif cal.residual < 1000:
        score += 5
    else:
        score -= 30

    if cal.tick_source == "formula":
        score += 50 if formula_log_score >= 0.3 else -10
    elif cal.tick_source == "fused":
        score += 30 if cal.formula_anchor_count >= 2 else 5 if cal.formula_anchor_count > 0 else 0
        score += 20 if formula_log_score >= 0.3 else 0
    elif cal.tick_source == "tesseract":
        score += 8
    elif cal.tick_source == "heuristic":
        score -= 20

    if cal.axis_type == "log" and formula_log_score >= 0.3:
        score += 15

    return score


def _select_best_ocr_calibration(
    axis: Axis,
    candidate_maps: list[tuple[str, list[tuple[int, Optional[float]]]]],
    image,
    policy,
    axis_preferred: str | None,
    is_log: bool,
    formula_anchor_count: int,
    formula_log_score: float,
    formula_selected_count: int,
    axis_anchor_stats: dict,
    formula_plan: Any,
    formula_batch_stats,
) -> Optional[CalibratedAxis]:
    """Calibrate multiple OCR maps and keep the lowest-risk candidate."""
    best_cal = None
    best_score = -1e9
    tesseract_residual = None
    source_scores = []

    for source, labeled in candidate_maps:
        valid_count = sum(1 for _, value in labeled if value is not None)
        if source != "heuristic" and valid_count < 2:
            continue

        cal = calibrate_axis(
            axis,
            labeled,
            image=image,
            policy=policy,
            preferred_type=axis_preferred,
            is_log=is_log,
            tick_source=source,
            anchor_count=axis_anchor_stats.get("anchor_count", 0),
            formula_anchor_count=formula_anchor_count,
            formula_log_score=formula_log_score,
            formula_selected_count=formula_selected_count,
            tesseract_anchor_count=axis_anchor_stats.get("tesseract_count", 0),
            label_anchor_count=axis_anchor_stats.get("label_count", 0),
            formula_batch_candidate_count=formula_plan.requested_count if formula_plan else 0,
            formula_batch_kept_count=formula_plan.kept_count if formula_plan else 0,
            formula_batch_chunks=formula_batch_stats.chunks if formula_batch_stats else 0,
            formula_batch_requested=formula_batch_stats.requested if formula_batch_stats else 0,
            formula_batch_returned=formula_batch_stats.returned if formula_batch_stats else 0,
            formula_batch_ms=formula_batch_stats.elapsed_ms if formula_batch_stats else 0.0,
            apply_log_superscript_fix=(source == "tesseract" or (source == "fused" and formula_anchor_count >= 2)),
        )
        if cal is None:
            continue
        if source == "tesseract":
            tesseract_residual = cal.residual

        # FormulaOCR returns a parsed value for the specific crop anchor.  For
        # superscript/log labels this is more authoritative than tesseract's
        # flattened text at the same anchor, so prefer the fused map unless it
        # produces a clearly worse fit.
        if (
            source == "fused"
            and formula_anchor_count < 2
            and tesseract_residual is not None
            and cal.residual > (tesseract_residual * 1.2 + 0.1)
        ):
            continue
        if source == "fused" and formula_anchor_count >= 2:
            tesseract_values = None
            for other_source, other_labeled in candidate_maps:
                if other_source == "tesseract":
                    tesseract_values = {
                        int(pixel): value for pixel, value in other_labeled if value is not None
                    }
                    break
            if tesseract_values:
                changed_values = 0
                for pixel, value in labeled:
                    if value is None:
                        continue
                    old_value = tesseract_values.get(int(pixel))
                    if old_value is not None and abs(float(old_value) - float(value)) > 1e-9:
                        changed_values += 1
                if changed_values:
                    candidate_score = _candidate_calibration_score(cal, valid_count, formula_log_score)
                    candidate_score += changed_values * 25.0
                    source_scores.append({
                        "source": source,
                        "valid_count": int(valid_count),
                        "score": float(candidate_score),
                        "axis_type": cal.axis_type,
                        "residual": float(cal.residual),
                        "changed_values": int(changed_values),
                    })
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_cal = cal
                    continue

        candidate_score = _candidate_calibration_score(cal, valid_count, formula_log_score)
        source_scores.append({
            "source": source,
            "valid_count": int(valid_count),
            "score": float(candidate_score),
            "axis_type": cal.axis_type,
            "residual": float(cal.residual),
        })
        if candidate_score > best_score:
            best_score = candidate_score
            best_cal = cal

    if best_cal is not None:
        trace = dict(best_cal.debug_trace or {})
        trace["candidate_sources"] = source_scores
        trace["selected_candidate_score"] = float(best_score)
        trace["axis_preferred"] = axis_preferred
        trace["axis_is_log"] = bool(is_log)
        best_cal.debug_trace = trace
    return best_cal


@dataclass
class FormulaBatchRequest:
    axis_id: int
    anchor_idx: int
    crop: np.ndarray
    score: float


@dataclass
class FormulaBatchPlan:
    requests: list[FormulaBatchRequest]
    requested_count: int
    kept_count: int
    max_total_crops: int
    batch_size_hint: int

    @property
    def dropped_count(self) -> int:
        return max(0, self.requested_count - self.kept_count)


@dataclass
class FormulaLabelContext:
    """Prepared label-routing state for one image/panel calibration pass."""

    log_y_prob: float
    log_x_prob: float
    axis_is_log: dict[int, bool]
    tesseract_log_scores: dict[int, float]
    axis_anchor_map: dict[int, list]
    axis_anchor_meta: dict[int, dict[str, Any]]
    formula_plan: FormulaBatchPlan


def prepare_formula_label_context(
    axes: List[Axis],
    image,
    policy=None,
    use_ocr: bool = False,
    type_probs: dict | None = None,
    formula_batch_max_crops: int | None = None,
) -> FormulaLabelContext:
    """Prepare label anchors and crop plan without running FormulaOCR.

    This is the shared planning step used by both the single-image path and
    the cross-image batch pipeline.  It keeps crop selection and axis routing
    in one place while allowing OCR execution to happen later, once crops from
    multiple charts have been merged into a global batch.
    """
    from plot_extractor.core.ocr_reader import (  # pylint: disable=import-outside-toplevel
        detect_tick_label_anchors,
    )

    if type_probs:
        loglog_prob = type_probs.get("loglog", 0.0)
        log_x_specific = type_probs.get("log_x", 0.0)
        log_y_specific = type_probs.get("log_y", 0.0)

        log_y_prob = log_y_specific
        if loglog_prob > 0.25 and log_x_specific < 0.2:
            log_y_prob = max(log_y_prob, loglog_prob)

        log_x_prob = log_x_specific
        if loglog_prob > 0.25 and log_y_specific < 0.2:
            log_x_prob = max(log_x_prob, loglog_prob)
    else:
        log_y_prob = 0.0
        log_x_prob = 0.0

    x_axes = [a for a in axes if a.direction == "x"]
    y_axes = [a for a in axes if a.direction == "y"]

    tesseract_log_scores: dict[int, float] = {}
    try:
        from plot_extractor.core.scale_detector import detect_log_notation_ocr  # pylint: disable=import-outside-toplevel
    except Exception:
        detect_log_notation_ocr = None

    for axis in axes:
        if not (axis.ticks and len(axis.ticks) >= 3):
            tesseract_log_scores[id(axis)] = 0.0
            continue
        type_prior_log = (
            (axis.direction == "x" and log_x_prob > 0.25)
            or (axis.direction == "y" and log_y_prob > 0.25)
        )
        dense_tick_axis = len(axis.ticks or []) > 24
        if (
            use_ocr
            and detect_log_notation_ocr is not None
            and not type_prior_log
            and not dense_tick_axis
        ):
            tesseract_log_scores[id(axis)] = detect_log_notation_ocr(
                image, axis, policy=policy,
            )
        else:
            tesseract_log_scores[id(axis)] = 0.0

    log_notation_scores = dict(tesseract_log_scores)

    x_log = {}
    for axis in x_axes:
        if axis.ticks and len(axis.ticks) >= 3:
            prior_x_log = any(v for k, v in x_log.items())
            x_log[id(axis)] = should_treat_as_log(
                image, axis, cross_axis_log=prior_x_log,
                log_notation_score=log_notation_scores.get(id(axis), 0.0),
            )
            if log_x_prob > 0.25:
                x_log[id(axis)] = True
        else:
            x_log[id(axis)] = False

    any_x_log = any(x_log.values())

    y_log = {}
    for axis in y_axes:
        if axis.ticks and len(axis.ticks) >= 3:
            prior_y_log = any(v for k, v in y_log.items())
            # For log_x charts, X-axis log status should not leak into Y-axis
            # detection via cross_axis_log — that causes false-log classification
            # on linear Y axes.  loglog charts are handled by log_y_prob > 0.25.
            if log_x_prob > 0.25:
                cross_axis = prior_y_log
            else:
                cross_axis = any_x_log or prior_y_log
            y_log[id(axis)] = should_treat_as_log(
                image, axis, cross_axis_log=cross_axis,
                log_notation_score=log_notation_scores.get(id(axis), 0.0),
            )
            if log_y_prob > 0.25:
                y_log[id(axis)] = True
        else:
            y_log[id(axis)] = False

    axis_is_log = {**x_log, **y_log}
    axis_anchor_map: dict[int, list] = {}
    axis_anchor_meta: dict[int, dict[str, Any]] = {}
    primary_directions = {
        axis.direction for axis in axes
        if axis.side in ("bottom", "left")
    }
    formula_plan = FormulaBatchPlan(
        requests=[],
        requested_count=0,
        kept_count=0,
        max_total_crops=0,
        batch_size_hint=0,
    )
    if use_ocr:
        for axis in axes:
            tick_pixels = [t[0] for t in (axis.ticks or [])]
            skip_secondary_axis_ocr = (
                axis.side in ("top", "right")
                and axis.direction in primary_directions
            )
            if skip_secondary_axis_ocr:
                # Throughput guard: top/right duplicate axes usually mirror
                # bottom/left scales. Skip expensive OCR probing and allow
                # heuristic calibration on the secondary axis.
                axis_anchor_map[id(axis)] = []
                axis_anchor_meta[id(axis)] = {
                    "anchor_count": 0,
                    "tesseract_count": 0,
                    "label_count": 0,
                    "geometry_label_count": 0,
                    "selected_count": 0,
                    "formula_log_score": 0.0,
                    "selected_indices": [],
                }
                continue

            anchors = detect_tick_label_anchors(
                image,
                axis,
                tick_pixels,
                policy=policy,
                force_geometry_fallback=axis_is_log.get(id(axis), False),
            )
            axis_anchor_map[id(axis)] = anchors

            tesseract_count = sum(1 for anchor in anchors if anchor.tesseract_value is not None)
            label_count = sum(1 for anchor in anchors if _anchor_has_label_evidence(anchor))
            geometry_label_count = sum(
                1 for anchor in anchors
                if anchor.label_bbox != (0, 0, 0, 0)
            )
            axis_anchor_meta[id(axis)] = {
                "anchor_count": len(anchors),
                "tesseract_count": tesseract_count,
                "label_count": label_count,
                "geometry_label_count": geometry_label_count,
                "selected_count": 0,
                "formula_log_score": 0.0,
                "selected_indices": [],
            }

    if use_ocr:
        formula_plan = _plan_formula_batch_requests(
            axes,
            axis_anchor_map,
            axis_is_log,
            max_total_crops=formula_batch_max_crops,
        )
        for meta in axis_anchor_meta.values():
            meta["selected_indices"] = []
            meta["selected_count"] = 0
        for req in formula_plan.requests:
            meta = axis_anchor_meta.get(req.axis_id)
            if not meta:
                continue
            meta["selected_indices"].append(req.anchor_idx)
            meta["selected_count"] = len(meta["selected_indices"])

    return FormulaLabelContext(
        log_y_prob=log_y_prob,
        log_x_prob=log_x_prob,
        axis_is_log=axis_is_log,
        tesseract_log_scores=tesseract_log_scores,
        axis_anchor_map=axis_anchor_map,
        axis_anchor_meta=axis_anchor_meta,
        formula_plan=formula_plan,
    )


def _plan_formula_batch_requests(
    axes: List[Axis],
    axis_anchor_map: dict[int, list],
    axis_is_log: dict[int, bool],
    max_total_crops: int | None = None,
) -> FormulaBatchPlan:
    """Build a small, score-ranked FormulaOCR batch plan."""
    staged: list[tuple[int, bool, FormulaBatchRequest]] = []
    request_pos = 0
    primary_directions = {
        axis.direction for axis in axes
        if axis.side in ("bottom", "left")
    }

    for axis in axes:
        anchors = axis_anchor_map.get(id(axis), [])
        if not anchors:
            continue
        if axis.side in ("top", "right") and axis.direction in primary_directions:
            continue

        is_axis_log = axis_is_log.get(id(axis), False)
        tesseract_count = sum(1 for anchor in anchors if getattr(anchor, "tesseract_value", None) is not None)
        label_count = sum(1 for anchor in anchors if _anchor_has_label_evidence(anchor))

        # Use unified gate function (Docling's is_processable pattern)
        from plot_extractor.core.label_crop_planner import is_processable_for_formula  # pylint: disable=import-outside-toplevel
        needs_formula = is_processable_for_formula(
            axis_is_log=is_axis_log,
            tesseract_count=tesseract_count,
            label_count=label_count,
            anchors=anchors,
        )
        if not needs_formula:
            continue

        sparse_or_suspicious = tesseract_count < 3 or tesseract_count < max(3, int(label_count * 0.6))
        if is_axis_log:
            max_crops = 3 if sparse_or_suspicious else 2
        elif axis.direction == "y" and tesseract_count < 3:
            max_crops = 1
        else:
            max_crops = 1
        selected_indices = _select_formula_anchor_indices(axis, anchors, max_crops=max_crops)
        for anchor_idx in selected_indices:
            anchor = anchors[anchor_idx]
            if getattr(anchor, "crop", None) is None or anchor.crop.size == 0:
                continue
            score = _score_formula_anchor_candidate(axis, anchor, anchor_idx, len(anchors))
            staged.append(
                (
                    request_pos,
                    is_axis_log,
                    FormulaBatchRequest(
                        axis_id=id(axis),
                        anchor_idx=anchor_idx,
                        crop=anchor.crop,
                        score=score,
                    ),
                )
            )
            request_pos += 1

    requested_count = len(staged)
    if max_total_crops is None:
        max_total_crops = 4 if any(axis_is_log.values()) else 2

    if requested_count <= max_total_crops:
        requests = [req for _, _, req in staged]
        return FormulaBatchPlan(
            requests=requests,
            requested_count=requested_count,
            kept_count=requested_count,
            max_total_crops=max_total_crops,
            batch_size_hint=requested_count,
        )

    keep_positions: set[int] = set()
    log_axis_best: dict[int, tuple[int, FormulaBatchRequest]] = {}
    for pos, is_log, req in staged:
        if not is_log:
            continue
        old = log_axis_best.get(req.axis_id)
        if old is None or req.score > old[1].score:
            log_axis_best[req.axis_id] = (pos, req)
    for pos, _req in sorted(log_axis_best.values(), key=lambda item: item[1].score, reverse=True):
        if len(keep_positions) >= max_total_crops:
            break
        keep_positions.add(pos)
    for pos, _is_log, _req in sorted(
        staged,
        key=lambda item: (1 if item[1] else 0, item[2].score),
        reverse=True,
    ):
        if len(keep_positions) >= max_total_crops:
            break
        keep_positions.add(pos)
    requests = [req for pos, _is_log, req in staged if pos in keep_positions]
    return FormulaBatchPlan(
        requests=requests,
        requested_count=requested_count,
        kept_count=len(requests),
        max_total_crops=max_total_crops,
        batch_size_hint=min(max_total_crops, len(requests)),
    )


def calibrate_all_axes(
    axes: List[Axis], image, policy=None,
    use_ocr: bool = False, use_llm: bool = False,
    type_probs: dict | None = None,
    formula_batch_max_crops: int | None = None,
    formula_context: FormulaLabelContext | None = None,
    formula_request_results: dict[tuple[int, int], tuple[Optional[str], Optional[float]]] | None = None,
    formula_batch_stats=None,
) -> List[CalibratedAxis]:
    """Calibrate all detected axes.

    If *use_ocr* is True, tick labels are read via OCR (requires tesseract).
    If False, calibration falls back to heuristic synthetic ticks.
    """
    from plot_extractor.core.formula_ocr import (  # pylint: disable=import-outside-toplevel
        score_latex_log_notation,
    )

    if formula_context is None:
        formula_context = prepare_formula_label_context(
            axes,
            image,
            policy=policy,
            use_ocr=use_ocr,
            type_probs=type_probs,
            formula_batch_max_crops=formula_batch_max_crops,
        )

    log_y_prob = formula_context.log_y_prob
    log_x_prob = formula_context.log_x_prob
    axis_anchor_map = formula_context.axis_anchor_map
    axis_anchor_meta = formula_context.axis_anchor_meta
    formula_plan = formula_context.formula_plan

    formula_request_results = formula_request_results or {}

    log_notation_scores: dict[int, float] = dict(formula_context.tesseract_log_scores)

    x_axes = [a for a in axes if a.direction == "x"]
    y_axes = [a for a in axes if a.direction == "y"]

    x_log = {}
    for axis in x_axes:
        if axis.ticks and len(axis.ticks) >= 3:
            prior_x_log = any(v for k, v in x_log.items())
            x_log[id(axis)] = should_treat_as_log(
                image, axis, cross_axis_log=prior_x_log,
                log_notation_score=log_notation_scores.get(id(axis), 0.0),
            )
            if log_x_prob > 0.25:
                x_log[id(axis)] = True
        else:
            x_log[id(axis)] = False

    any_x_log = any(x_log.values())

    y_log = {}
    for axis in y_axes:
        if axis.ticks and len(axis.ticks) >= 3:
            prior_y_log = any(v for k, v in y_log.items())
            # For log_x charts, X-axis log status should not leak into Y-axis
            # detection via cross_axis_log — that causes false-log classification
            # on linear Y axes.  loglog charts are handled by log_y_prob > 0.25.
            if log_x_prob > 0.25:
                cross_axis = prior_y_log
            else:
                cross_axis = any_x_log or prior_y_log
            y_log[id(axis)] = should_treat_as_log(
                image, axis, cross_axis_log=cross_axis,
                log_notation_score=log_notation_scores.get(id(axis), 0.0),
            )
            if log_y_prob > 0.25:
                y_log[id(axis)] = True
        else:
            y_log[id(axis)] = False

    axis_is_log = {**x_log, **y_log}

    # Infer guessed chart type from probabilities
    guessed_type = None
    if type_probs:
        guessed_type = max(type_probs, key=type_probs.get)

    # For log_x charts, Y axis is linear by definition.  Force it here to
    # prevent should_treat_as_log false-positives from grid/tick analysis.
    # loglog charts are handled by log_y_prob > 0.25 (guessed_type=loglog).
    if guessed_type == "log_x":
        for axis in y_axes:
            axis_is_log[id(axis)] = False

    for axis in axes:
        axis_id = id(axis)
        meta = axis_anchor_meta.get(axis_id)
        if not meta:
            continue
        selected_indices = meta.get("selected_indices", []) or []
        formula_scores = []
        formula_values = []
        for anchor_idx in selected_indices:
            latex, value = formula_request_results.get((axis_id, anchor_idx), (None, None))
            if latex and score_latex_log_notation is not None:
                formula_scores.append(score_latex_log_notation(latex))
            if value is not None:
                formula_values.append(value)
        meta["formula_log_score"] = max(formula_scores) if formula_scores else 0.0
        meta["formula_value_count"] = len(formula_values)

    # Infer guessed chart type from probabilities; use it as a stronger
    # signal than raw probability thresholds for axis_preferred.
    guessed_type = None
    if type_probs:
        guessed_type = max(type_probs, key=type_probs.get)

    calibrated = []
    for axis in axes:
        is_log = axis_is_log.get(id(axis), False)

        # Set per-axis preferred type
        # Visual log detection (should_treat_as_log) is more reliable than
        # chart-type softmax probabilities for sparse-tick axes.
        axis_preferred = None
        if is_log:
            axis_preferred = "log"
        elif guessed_type in ("log_y", "loglog") and axis.direction == "y":
            axis_preferred = "log"
        elif guessed_type in ("log_x", "loglog") and axis.direction == "x":
            axis_preferred = "log"
        elif axis.direction == "y" and log_y_prob > 0.25:
            axis_preferred = "log"
        elif axis.direction == "x" and log_x_prob > 0.25:
            axis_preferred = "log"

        # For log_x charts, Y axis is linear.  Force linear preference to
        # prevent fit_axis_multi_hypothesis from choosing log when OCR values
        # happen to have lower log-fit residual.
        if guessed_type == "log_x" and axis.direction == "y":
            axis_preferred = "linear"

        tick_pixels = [t[0] for t in axis.ticks]
        axis_anchor_stats = axis_anchor_meta.get(id(axis), {
            "anchor_count": 0,
            "tesseract_count": 0,
            "label_count": 0,
            "selected_count": 0,
            "formula_log_score": 0.0,
            "formula_value_count": 0,
        })
        formula_anchor_count = sum(
            1 for idx in range(axis_anchor_stats.get("anchor_count", 0))
            if formula_request_results.get((id(axis), idx), (None, None))[1] is not None
        )
        formula_batch_requested = formula_batch_stats.requested if formula_batch_stats else 0
        formula_batch_returned = formula_batch_stats.returned if formula_batch_stats else 0
        formula_batch_ms = formula_batch_stats.elapsed_ms if formula_batch_stats else 0.0
        formula_log_score = float(axis_anchor_stats.get("formula_log_score", 0.0))
        formula_value_count = int(axis_anchor_stats.get("formula_value_count", 0))
        formula_selected_count = int(axis_anchor_stats.get("selected_count", 0))
        formula_batch_candidate_count = formula_plan.requested_count if formula_plan else 0
        formula_batch_kept_count = formula_plan.kept_count if formula_plan else 0
        formula_batch_chunks = formula_batch_stats.chunks if formula_batch_stats else 0
        use_formula_label_values = _should_use_formula_label_values(
            formula_anchor_count,
            formula_value_count,
            formula_log_score,
        )
        strong_formula_log = _should_use_formula_log_hint(
            formula_anchor_count,
            formula_value_count,
            formula_log_score,
        )
        if use_ocr:
            anchors = axis_anchor_map.get(id(axis), [])
            if anchors:
                tesseract_labeled = []
                formula_labeled = []
                fused_labeled = []
                for anchor_idx, anchor in enumerate(anchors):
                    formula_key = (id(axis), anchor_idx)
                    formula_latex, formula_value = formula_request_results.get(formula_key, (None, None))
                    if formula_latex is not None:
                        anchor.formula_latex = formula_latex
                    if formula_value is not None:
                        anchor.formula_value = formula_value

                    tesseract_labeled.append((anchor.tick_pixel, anchor.tesseract_value))
                    formula_labeled.append((anchor.tick_pixel, anchor.formula_value))
                    fused_value = anchor.formula_value if anchor.formula_value is not None else anchor.tesseract_value
                    fused_labeled.append((anchor.tick_pixel, fused_value))
                    if anchor.formula_value is not None and anchor.tesseract_value is not None:
                        anchor.source = "fused"
                    elif anchor.formula_value is not None:
                        anchor.source = "formula"
                    elif anchor.tesseract_value is not None:
                        anchor.source = "tesseract"
                    else:
                        anchor.source = "missing"

                if is_log:
                    tesseract_for_fusion = _fix_log_superscript_ocr(tesseract_labeled)
                else:
                    tesseract_for_fusion = tesseract_labeled
                if formula_log_score >= 0.3 and formula_value_count >= 1:
                    tesseract_for_fusion = _force_log_superscript_ocr(tesseract_for_fusion)
                corrected_tesseract_by_tick = {
                    int(pixel): value for pixel, value in tesseract_for_fusion if value is not None
                }
                fused_labeled = []
                for anchor_idx, anchor in enumerate(anchors):
                    if anchor.formula_value is not None:
                        fused_value = anchor.formula_value
                    else:
                        fused_value = corrected_tesseract_by_tick.get(
                            int(anchor.tick_pixel),
                            anchor.tesseract_value,
                        )
                    fused_labeled.append((anchor.tick_pixel, fused_value))

                # Use fixed priority table for candidate selection
                from plot_extractor.core.label_crop_planner import build_candidate_maps  # pylint: disable=import-outside-toplevel
                candidate_maps = build_candidate_maps(
                    formula_log_score=formula_log_score,
                    formula_value_count=formula_value_count,
                    formula_anchor_count=formula_anchor_count,
                    tesseract_count=axis_anchor_stats.get("tesseract_count", 0),
                    label_count=axis_anchor_stats.get("label_count", 0),
                    anchors=anchors,
                    formula_labeled=formula_labeled,
                    fused_labeled=fused_labeled,
                    tesseract_labeled=tesseract_labeled,
                    axis=axis,
                )
                # Handle formula_generated side effects
                if candidate_maps and candidate_maps[0][0] == "formula_generated":
                    is_log = True
                    axis_preferred = "log"
                    # Fill in the actual generated ticks
                    if not candidate_maps[0][1]:
                        generated = _build_formula_generated_log_ticks(axis, anchors)
                        candidate_maps[0] = ("formula_generated", generated)
                        if not generated:
                            # Fall back to heuristic
                            candidate_maps = [("heuristic", [])]
                if not candidate_maps:
                    candidate_maps = [("heuristic", [])]
            else:
                candidate_maps = [("heuristic", [(p, None) for p in tick_pixels])]
        else:
            candidate_maps = [("heuristic", [(p, None) for p in tick_pixels])]

        primary_labeled = candidate_maps[0][1] if candidate_maps else []
        valid_count = sum(1 for _, v in primary_labeled if v is not None)

        # LLM fallback when OCR yields insufficient labels
        if use_llm:
            n_ticks = len(tick_pixels)
            if valid_count < 2 or (n_ticks >= 6 and valid_count < n_ticks * 0.4):
                enhanced = _llm_enhance_axis_labels(image, axis, primary_labeled)
                if enhanced is not None:
                    candidate_maps = [("llm", enhanced)]
                    valid_count = sum(1 for _, v in enhanced if v is not None)

        if strong_formula_log:
            is_log = True
            axis_preferred = "log"

        cal = _select_best_ocr_calibration(
            axis,
            candidate_maps,
            image,
            policy,
            axis_preferred,
            is_log,
            formula_anchor_count,
            formula_log_score,
            formula_selected_count,
            axis_anchor_stats,
            formula_plan,
            formula_batch_stats,
        )
        if cal is None:
            cal = calibrate_axis(
                axis, [], image=image, policy=policy, preferred_type=axis_preferred,
                is_log=is_log,
                tick_source="heuristic",
                anchor_count=axis_anchor_stats.get("anchor_count", 0),
                formula_anchor_count=formula_anchor_count,
                formula_log_score=formula_log_score,
                formula_selected_count=formula_selected_count,
                tesseract_anchor_count=axis_anchor_stats.get("tesseract_count", 0),
                label_anchor_count=axis_anchor_stats.get("label_count", 0),
                formula_batch_candidate_count=formula_batch_candidate_count,
                formula_batch_kept_count=formula_batch_kept_count,
                formula_batch_chunks=formula_batch_chunks,
                formula_batch_requested=formula_batch_requested,
                formula_batch_returned=formula_batch_returned,
                formula_batch_ms=formula_batch_ms,
            )
        if cal is not None:
            calibrated.append(cal)
    return calibrated
