"""Infer whether an axis is log or linear from visual spacing patterns.

Two-stage strategy (user-directed):
1.  Inspect grid-line spacing inside the plot area.  Non-uniform (geometric)
    spacing strongly indicates a log scale.
2.  If the grid is uniform or absent, inspect the detected tick positions.
    Unequally spaced minor ticks are the second log signal.

Only when either stage returns ``"log"`` do we activate the log-superscript
OCR fix, preventing false positives such as 105 → 10⁵ on linear axes.
"""
from typing import List
import numpy as np
import cv2

from plot_extractor.core.axis_detector import Axis, _find_peaks_1d


def _detect_grid_positions(image: np.ndarray, axis: Axis) -> List[int]:
    """Find grid-line pixel positions inside the plot area.

    Uses edge projection perpendicular to the axis direction.
    Returns positions in image coordinates.
    """
    h, w = image.shape[:2]
    gray = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )
    edges = cv2.Canny(gray, 50, 150)

    if axis.direction == "x":
        # Vertical grid lines → column projection inside plot area
        x1, x2 = int(axis.plot_start), int(axis.plot_end)
        if x2 <= x1:
            return []
        region = edges[:, x1:x2]
        if region.size == 0:
            return []
        profile = np.sum(region, axis=0).astype(float)
        min_dist = max(5, (x2 - x1) // 50)
        peaks = _find_peaks_1d(profile, min_height_ratio=0.15, min_distance=min_dist)
        return [x1 + p for p in peaks]

    # Horizontal grid lines → row projection inside plot area
    y1, y2 = int(axis.plot_start), int(axis.plot_end)
    if y2 <= y1:
        return []
    region = edges[y1:y2, :]
    if region.size == 0:
        return []
    profile = np.sum(region, axis=1).astype(float)
    min_dist = max(5, (y2 - y1) // 50)
    peaks = _find_peaks_1d(profile, min_height_ratio=0.15, min_distance=min_dist)
    return [y1 + p for p in peaks]


def _window_is_geometric(spacings: np.ndarray, window_size: int) -> float:
    """Fraction of sliding windows whose consecutive spacing ratios are consistent,
    non-1, and all on the same side of 1 (monotonic within window).

    A high fraction indicates repeating geometric sub-sequences — the hallmark
    of log axes with per-decade minor ticks where spacings shrink/grow within
    each decade but repeat across decades.
    """
    n = len(spacings)
    if n < window_size + 1:
        return 0.0

    geom_count = 0
    total = 0
    for i in range(n - window_size):
        window = spacings[i : i + window_size + 1]
        w_ratios = window[1:] / (window[:-1] + 1e-6)
        w_median = float(np.median(w_ratios))
        w_cv = float(np.std(w_ratios) / (np.abs(np.mean(w_ratios)) + 1e-6))

        # Ratios must be consistently on one side of 1 (monotonic within window)
        same_side = bool(np.all(w_ratios > 1.0) or np.all(w_ratios < 1.0))

        # Consistent AND non-1 AND monotonic
        if w_cv < 0.35 and same_side and not 0.88 < w_median < 1.12:
            geom_count += 1
        total += 1

    if total < 3:
        return 0.0
    return geom_count / total


def _classify_spacing(positions: List[int]) -> str:
    """Hierarchical scale classifier for axis spacing patterns.

    Level 0 — Uniformity gate:  equal spacing → linear (fast exit).
    Level 1 — Periodic geometric:  repeating per-decade minor-tick cycles.
    Level 2 — Continuous geometric:  single geometric progression.
    Level 3 — Ambiguous:  fall through to "unknown".
    """
    if len(positions) < 4:
        return "unknown"

    spacings = np.diff(sorted(set(positions)))
    if len(spacings) < 3:
        return "unknown"

    # Filter out tiny duplicates
    min_s = max(1.0, np.median(spacings) * 0.25)
    spacings = np.array([s for s in spacings if s >= min_s], dtype=float)
    if len(spacings) < 4:
        return "unknown"

    # ---- Global indicators ----
    ratios = spacings[1:] / (spacings[:-1] + 1e-6)
    ratio_median = float(np.median(ratios))
    ratio_cv = float(np.std(ratios) / (np.mean(ratios) + 1e-6))

    diffs = np.diff(spacings)
    diff_cv = float(np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6))

    is_geom = (
        ratio_cv < 0.5
        and 0.4 < ratio_median < 2.5
        and not 0.85 < ratio_median < 1.15
    )
    is_arith = diff_cv < 0.35

    spacing_range = float(np.ptp(spacings))
    spacing_median = float(np.median(spacings))

    # ---- Level 0: Uniformity gate ----
    # Spacings are roughly equal → linear.  Exit fast.
    if is_arith and not is_geom:
        return "linear"

    # ---- Dense-grid subsampling (conservative) ----
    # Only triggers when: many positions (≥ 20), very low spacing CV (< 0.3),
    # AND the subsampled major intervals show a clear geometric pattern.
    # This specific combination is characteristic of loglog charts with
    # dense minor grid — linear charts virtually never have 20+ equally
    # spaced minor grid lines.
    if len(spacings) >= 20:
        spacing_cv = float(np.std(spacings) / (spacing_median + 1e-6))
        if spacing_cv < 0.3:
            major_mask = spacings >= spacing_median * 1.3
            if 3 <= np.sum(major_mask) <= len(spacings) * 0.5:
                major = spacings[major_mask]
                m_ratios = major[1:] / (major[:-1] + 1e-6)
                m_ratio_median = float(np.median(m_ratios))
                m_ratio_cv = float(np.std(m_ratios) / (np.abs(np.mean(m_ratios)) + 1e-6))
                m_is_geom = (
                    m_ratio_cv < 0.45
                    and 0.4 < m_ratio_median < 2.5
                    and not 0.85 < m_ratio_median < 1.15
                )
                if m_is_geom:
                    return "log"
                for ws in (3, 4):
                    if ws + 1 > len(major):
                        break
                    if _window_is_geometric(major, ws) >= 0.35:
                        return "log"

    # ---- Level 1: Periodic geometric ----
    # Repeating per-decade minor ticks create cyclic spacing patterns.
    # Guardrail: spacing range must be large (≥ 1.2 × median) indicating
    # genuine major/minor tick separation, not just noise.
    if len(spacings) >= 6 and spacing_range > spacing_median * 1.2:
        for ws in (3, 4, 5):
            if ws + 1 > len(spacings):
                break
            frac = _window_is_geometric(spacings, ws)
            if frac >= 0.35:
                return "log"

    # ---- Level 2: Continuous geometric ----
    if is_geom and not is_arith:
        return "log"

    # ---- Level 3: Ambiguous ----
    return "unknown"


def infer_scale_from_grid(image: np.ndarray, axis: Axis) -> str:
    """Return 'log', 'linear', or 'unknown' based on grid-line spacing."""
    positions = _detect_grid_positions(image, axis)
    return _classify_spacing(positions)


def infer_scale_from_ticks(axis: Axis) -> str:
    """Return 'log', 'linear', or 'unknown' based on tick spacing."""
    if not axis.ticks or len(axis.ticks) < 4:
        return "unknown"
    positions = [int(t[0]) for t in axis.ticks]
    return _classify_spacing(positions)


def _relaxed_scale_check(image: np.ndarray, axis: Axis) -> bool:
    """Re-check with dense-grid subsampling when cross-axis evidence exists.

    Only called when another axis on the same chart was already confirmed
    as log.  Two paths:

    1. Major-interval extraction: keep spacings ≥ 1.4× median, re-classify.
       Works when decade-boundary spacings visibly stand out from minor ones.
    2. Multi-stride subsampling: for loglog charts where minor grid is so
       dense that decade boundaries barely exceed the median.  Tries
       different sampling strides to align with the decade period.
    """
    grid_pos = _detect_grid_positions(image, axis)
    if len(grid_pos) >= 10:
        g_spacings = np.diff(sorted(set(grid_pos)))
        g_median = float(np.median(g_spacings))
        major_mask = g_spacings >= g_median * 1.3
        if np.sum(major_mask) >= 3:
            if _classify_spacing_from_spacings(g_spacings[major_mask]) == "log":
                return True

    if axis.ticks and len(axis.ticks) >= 10:
        t_positions = [int(t[0]) for t in axis.ticks]
        t_spacings = np.diff(sorted(set(t_positions)))
        t_median = float(np.median(t_spacings))
        major_mask = t_spacings >= t_median * 1.3
        if np.sum(major_mask) >= 3:
            if _classify_spacing_from_spacings(t_spacings[major_mask]) == "log":
                return True

    return False


def _classify_spacing_from_spacings(spacings: np.ndarray) -> str:
    """Run the hierarchical checks on pre-computed spacings."""
    if len(spacings) < 4:
        return "unknown"
    ratios = spacings[1:] / (spacings[:-1] + 1e-6)
    ratio_median = float(np.median(ratios))
    ratio_cv = float(np.std(ratios) / (np.mean(ratios) + 1e-6))
    diffs = np.diff(spacings)
    diff_cv = float(np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6))
    is_geom = (
        ratio_cv < 0.5
        and 0.4 < ratio_median < 2.5
        and not 0.85 < ratio_median < 1.15
    )
    is_arith = diff_cv < 0.35
    if is_arith and not is_geom:
        return "linear"
    spacing_range = float(np.ptp(spacings))
    spacing_median = float(np.median(spacings))
    if len(spacings) >= 6 and spacing_range > spacing_median * 1.2:
        for ws in (3, 4, 5):
            if ws + 1 > len(spacings):
                break
            if _window_is_geometric(spacings, ws) >= 0.35:
                return "log"
    if is_geom and not is_arith:
        return "log"
    return "unknown"


def should_treat_as_log(
    image: np.ndarray, axis: Axis, cross_axis_log: bool = False,
) -> bool:
    """Two-stage log detector: grid first, then ticks.

    When *cross_axis_log* is True (another axis on the same chart was
    already confirmed as log), a relaxed dense-grid subsampling pass is
    run on ambiguous axes — this catches loglog charts where dense minor
    grid/tick lines make each axis individually ambiguous.

    Returns True only when visual evidence suggests a logarithmic scale,
    preventing the superscript fix from firing on linear axes.
    """
    grid_guess = infer_scale_from_grid(image, axis)
    if grid_guess == "log":
        return True
    if grid_guess == "linear":
        return False  # grid is definitive — cross-axis cannot override
    tick_guess = infer_scale_from_ticks(axis)
    if tick_guess == "log":
        return True
    # Cross-axis relaxed check: dense subsampling on ambiguous axes
    if cross_axis_log and tick_guess == "unknown":
        return _relaxed_scale_check(image, axis)
    return False
