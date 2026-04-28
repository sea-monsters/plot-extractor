"""Calibrate axes: map pixel coordinates to data values."""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from plot_extractor.core.axis_detector import Axis
from plot_extractor.core.scale_detector import should_treat_as_log, detect_log_notation_ocr
from plot_extractor.utils.math_utils import pixel_to_data, fit_linear


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
        from plot_extractor.utils.math_utils import data_to_pixel  # pylint: disable=import-outside-toplevel
        return data_to_pixel(value, self.a, self.b, self.axis_type)


def calibrate_axis(
    axis: Axis,
    labeled_ticks: List[Tuple[int, Optional[float]]],
    image=None,
    policy=None,
    preferred_type: str | None = None,
    is_log: bool | None = None,
) -> Optional[CalibratedAxis]:
    """Build a calibrated axis from detected ticks and their labels."""
    # Gate log superscript fix behind visual log-scale detection.
    # When is_log is pre-computed (cross-axis detection), skip internal check.
    if is_log is None and image is not None:
        is_log = should_treat_as_log(image, axis)
    if is_log:
        labeled_ticks = _fix_log_superscript_ocr(labeled_ticks)

    # Filter ticks with valid numeric labels
    valid = [(p, v) for p, v in labeled_ticks if v is not None]

    # For log-preferred axes, discard non-positive values (they break log fit)
    if preferred_type == "log":
        valid = [(p, v) for p, v in valid if v > 0]

    # Guardrail: if OCR values look internally inconsistent, ignore them
    # and let heuristic fallback take over.
    if len(valid) >= 3 and not _is_plausible_ocr_tick_sequence(valid, preferred_type):
        valid = []

    if len(valid) < 2:
        # Heuristic fallback: generate synthetic values from tick spacing pattern
        synthetic_valid = _build_heuristic_ticks(axis)
        if len(synthetic_valid) >= 2:
            valid = synthetic_valid

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

    return CalibratedAxis(
        axis=axis,
        axis_type=axis_type,
        a=a,
        b=b,
        inverted=inverted,
        tick_map=valid,
        residual=residual,
    )


def _build_heuristic_ticks(axis):
    """Generate synthetic (pixel, value) ticks from spacing pattern when OCR fails.

    Uses tick spacing uniformity to guess linear vs log, then assigns
    synthetic values.  Absolute scale is arbitrary, but preserves shape.
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

    n = len(ticks)
    if is_log:
        # Synthetic log values: 1, 10, 100, ... or 1, 2, 5, 10, ...
        # Use uniform log spacing as simplest fallback
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


def calibrate_all_axes(
    axes: List[Axis], image, policy=None,
    use_ocr: bool = False, use_llm: bool = False,
    type_probs: dict | None = None,
) -> List[CalibratedAxis]:
    """Calibrate all detected axes.

    If *use_ocr* is True, tick labels are read via OCR (requires tesseract).
    If False, calibration falls back to heuristic synthetic ticks.
    """
    from plot_extractor.core.ocr_reader import read_all_tick_labels  # pylint: disable=import-outside-toplevel

    # Derive preferred axis type from chart-type probabilities.
    # loglog is only applied to an axis when the specific log type for the
    # OTHER axis is not also significant (prevents log_x charts from getting
    # their y-axis forced to log, and vice versa).
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

    # Staged axis evaluation: X-axes first, then Y-axes.
    # Each axis is checked with the 3-level hierarchical detector.
    # Cross-axis signal propagates in processing order: later axes
    # (especially Y-axes) can reference earlier confirmed-log axes
    # to break ties on ambiguous spacing (e.g. dense minor grid on loglog).
    x_axes = [a for a in axes if a.direction == "x"]
    y_axes = [a for a in axes if a.direction == "y"]

    # Pre-compute log-notation scores for each axis.
    # Two-tier, priority-ordered:
    #   Tier 1 (FormulaOCR) — PP-FormulaNet_plus-S, reads superscripts.
    #       One crop per axis.  Takes PRIORITY when available.
    #   Tier 2 (tesseract) — lightweight fallback when FormulaOCR absent.
    log_notation_scores: dict[int, float] = {}
    _formula_ocr = None
    try:
        from plot_extractor.core.formula_ocr import get_formula_ocr  # pylint: disable=import-outside-toplevel
        _formula_ocr = get_formula_ocr()
    except Exception:
        _formula_ocr = None

    # FormulaOCR axis results — stored for later use as tick labels
    _formula_axis_results: dict[int, object] = {}

    # Collect all axis crops for batch inference (one call total)
    _batch_axes = []
    _batch_crops = []
    for axis in axes:
        if axis.ticks and len(axis.ticks) >= 3:
            _batch_axes.append(axis)
            _batch_crops.append(_crop_axis_region(image, axis, padding=15))

    if _formula_ocr is not None and _batch_crops:
        # Tier 1: single batch call for all axes
        batch_results = _formula_ocr.read_axes_batch(_batch_crops)
        for axis, result in zip(_batch_axes, batch_results):
            if result is not None:
                log_notation_scores[id(axis)] = result.log_confidence
                if result.count_10pow >= 2:
                    _formula_axis_results[id(axis)] = result
            else:
                log_notation_scores[id(axis)] = 0.0

    # Fill remaining axes (those with <3 ticks) and tesseract fallback
    for axis in axes:
        if id(axis) in log_notation_scores:
            continue  # Already processed by FormulaOCR batch
        if not (axis.ticks and len(axis.ticks) >= 3):
            log_notation_scores[id(axis)] = 0.0
        elif use_ocr:
            log_notation_scores[id(axis)] = detect_log_notation_ocr(
                image, axis, policy=policy,
            )
        else:
            log_notation_scores[id(axis)] = 0.0

    # ---- Stage 1: X-axes ----
    x_log = {}  # id(axis) -> bool
    for axis in x_axes:
        if axis.ticks and len(axis.ticks) >= 3:
            # Cross-axis from earlier X-axes (within-group)
            prior_x_log = any(v for k, v in x_log.items())
            x_log[id(axis)] = should_treat_as_log(
                image, axis, cross_axis_log=prior_x_log,
                log_notation_score=log_notation_scores.get(id(axis), 0.0),
            )
        else:
            x_log[id(axis)] = False

    any_x_log = any(x_log.values())

    # ---- Stage 2: Y-axes ----
    y_log = {}
    for axis in y_axes:
        if axis.ticks and len(axis.ticks) >= 3:
            # Cross-axis from X-stage results OR earlier Y-axes
            prior_y_log = any(v for k, v in y_log.items())
            y_log[id(axis)] = should_treat_as_log(
                image, axis, cross_axis_log=any_x_log or prior_y_log,
                log_notation_score=log_notation_scores.get(id(axis), 0.0),
            )
        else:
            y_log[id(axis)] = False

    # Merge results
    axis_is_log = {**x_log, **y_log}

    calibrated = []
    for axis in axes:
        is_log = axis_is_log.get(id(axis), False)

        # Set per-axis preferred type
        axis_preferred = None
        if axis.direction == "y" and log_y_prob > 0.25:
            axis_preferred = "log"
        elif axis.direction == "x" and log_x_prob > 0.25:
            axis_preferred = "log"

        tick_pixels = [t[0] for t in axis.ticks]
        if use_ocr:
            labeled = read_all_tick_labels(image, axis, tick_pixels, policy=policy)
        else:
            labeled = [(p, None) for p in tick_pixels]

        # FormulaOCR value injection: when FormulaOCR has high-confidence
        # values AND tesseract OCR produced few valid labels, use FormulaOCR
        # values as anchor points for calibration.
        formula_result = _formula_axis_results.get(id(axis))
        if formula_result is not None and formula_result.count_10pow >= 2:
            valid_count = sum(1 for _, v in labeled if v is not None)
            if valid_count < 3:
                formula_labeled = _align_formula_values_to_ticks(
                    tick_pixels, formula_result.values,
                    descending=(axis.direction == "y"),
                )
                # Merge: FormulaOCR fills None slots
                merged = []
                for (p_ocr, v_ocr), (p_fm, v_fm) in zip(labeled, formula_labeled):
                    if v_ocr is not None:
                        merged.append((p_ocr, v_ocr))
                    else:
                        merged.append((p_fm, v_fm))
                labeled = merged

        # LLM fallback when OCR yields insufficient labels
        if use_llm:
            valid_count = sum(1 for _, v in labeled if v is not None)
            n_ticks = len(tick_pixels)
            if valid_count < 2 or (n_ticks >= 6 and valid_count < n_ticks * 0.4):
                enhanced = _llm_enhance_axis_labels(image, axis, labeled)
                if enhanced is not None:
                    labeled = enhanced

        cal = calibrate_axis(
            axis, labeled, image=image, policy=policy, preferred_type=axis_preferred,
            is_log=is_log,
        )
        if cal is None:
            cal = calibrate_axis(
                axis, [], image=image, policy=policy, preferred_type=axis_preferred,
                is_log=is_log,
            )
        if cal is not None:
            calibrated.append(cal)
    return calibrated
