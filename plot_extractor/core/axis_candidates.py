"""Multi-candidate axis solving framework.

Provides top-k axis mapping candidates from OCR and heuristic sources,
ranked by confidence scores. This eliminates single-path catastrophic failures.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from plot_extractor.core.axis_detector import Axis
from plot_extractor.utils.math_utils import (
    fit_linear,
    fit_log,
    fit_linear_ransac,
    fit_log_ransac,
    _is_arithmetic_sequence,
    _is_geometric_sequence,
)


@dataclass
class AxisMappingCandidate:
    """Single axis mapping hypothesis."""
    scale: str  # "linear" | "log"
    a: float  # coefficient a
    b: float  # coefficient b
    residual: float  # fit residual
    confidence: float  # 0-100 score
    source: str  # "ocr" | "heuristic" | "default"
    tick_map: List[Tuple[int, float]]  # (pixel, value) pairs used


@dataclass
class AxisCandidatesResult:
    """Multi-candidate axis solving result."""
    all: List[AxisMappingCandidate]
    best: AxisMappingCandidate
    warnings: List[str]


def solve_axis_multi_candidate(
    axis: Axis,
    ocr_ticks: List[Tuple[int, Optional[float]]],
    preferred_type: str | None = None,
) -> AxisCandidatesResult:
    """Generate multiple axis mapping candidates from different sources.

    Candidates are ranked by confidence. This allows fallback when OCR fails.
    """
    candidates = []
    warnings = []

    # Candidate 1: OCR-derived (if sufficient valid ticks)
    valid_ocr = [(p, v) for p, v in ocr_ticks if v is not None]
    if len(valid_ocr) >= 2:
        ocr_cand = _solve_from_ocr(valid_ocr, preferred_type)
        if ocr_cand:
            candidates.append(ocr_cand)
        else:
            warnings.append("OCR-derived candidate failed (insufficient ticks)")

    # Candidate 2: Heuristic synthetic (always available as fallback)
    heur_cand = _solve_heuristic(axis, preferred_type)
    if heur_cand:
        candidates.append(heur_cand)

    # Sort by confidence (descending)
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    if not candidates:
        # Last resort: create a default linear candidate
        default = AxisMappingCandidate(
            scale="linear",
            a=1.0,
            b=0.0,
            residual=1e6,
            confidence=0.0,
            source="default",
            tick_map=[],
        )
        candidates.append(default)
        warnings.append("No valid axis candidate; using default linear mapping")

    return AxisCandidatesResult(
        all=candidates,
        best=candidates[0],
        warnings=warnings,
    )


def _solve_from_ocr(
    ticks: List[Tuple[int, float]],
    preferred_type: str | None = None,
) -> Optional[AxisMappingCandidate]:
    """Solve axis mapping from OCR-derived ticks using RANSAC robust regression."""
    pixels = np.array([p for p, _ in ticks], dtype=float)
    values = np.array([v for _, v in ticks], dtype=float)
    n = len(pixels)

    # Use RANSAC when enough points to benefit from outlier rejection
    use_ransac = n >= 3

    if use_ransac:
        lin_a, lin_b, lin_res, lin_inliers = fit_linear_ransac(pixels, values)
        log_a, log_b, log_res, log_inliers = fit_log_ransac(pixels, values)
    else:
        lin_a, lin_b, lin_res = fit_linear(pixels, values)
        lin_inliers = list(range(n))
        log_a, log_b, log_res = fit_log(pixels, values)
        log_inliers = list(range(n)) if log_a is not None else []

    # Build inlier-only tick maps for scoring
    lin_ticks = [ticks[i] for i in lin_inliers] if lin_inliers else ticks
    log_ticks = [ticks[i] for i in log_inliers] if log_inliers else ticks

    # Score both candidates
    lin_score = _score_axis_fit(lin_ticks, lin_a, lin_b, lin_res, "linear")
    if log_a is not None:
        log_score = _score_axis_fit(log_ticks, log_a, log_b, log_res, "log")
    else:
        log_score = 0.0

    # Prefer log if preferred_type is "log" and score is reasonable
    if preferred_type == "log" and log_a is not None and log_score > 0:
        best_scale = "log"
        best_a, best_b, best_res = log_a, log_b, log_res
        best_score = log_score
        best_ticks = log_ticks
    else:
        # Choose by score
        if lin_score >= log_score:
            best_scale = "linear"
            best_a, best_b, best_res = lin_a, lin_b, lin_res
            best_score = lin_score
            best_ticks = lin_ticks
        else:
            best_scale = "log"
            best_a, best_b, best_res = log_a, log_b, log_res
            best_score = log_score
            best_ticks = log_ticks

    if best_a is None:
        return None

    return AxisMappingCandidate(
        scale=best_scale,
        a=best_a,
        b=best_b,
        residual=best_res,
        confidence=best_score,
        source="ocr",
        tick_map=best_ticks,
    )


def _solve_heuristic(
    axis: Axis,
    preferred_type: str | None = None,
) -> Optional[AxisMappingCandidate]:
    """Solve axis mapping from heuristic synthetic ticks."""
    from plot_extractor.core.axis_calibrator import _build_heuristic_ticks  # pylint: disable=import-outside-toplevel

    ticks = _build_heuristic_ticks(axis)
    if len(ticks) < 2:
        return None

    pixels = np.array([p for p, _ in ticks], dtype=float)
    values = np.array([v for _, v in ticks], dtype=float)

    # Determine scale from value sequence pattern
    is_geom = _is_geometric_sequence(values, tol=0.3)
    is_arith = _is_arithmetic_sequence(values, tol=0.3)

    if preferred_type == "log":
        scale = "log"
    elif preferred_type == "linear":
        scale = "linear"
    elif is_geom and not is_arith:
        scale = "log"
    else:
        scale = "linear"

    if scale == "log" and np.all(values > 0):
        a, b, residual = fit_log(pixels, values)
    else:
        a, b, residual = fit_linear(pixels, values)

    if a is None:
        return None

    # Heuristic has low confidence (arbitrary scale)
    score = 20.0

    return AxisMappingCandidate(
        scale=scale,
        a=a,
        b=b,
        residual=residual,
        confidence=score,
        source="heuristic",
        tick_map=ticks,
    )


def _score_axis_fit(
    ticks: List[Tuple[int, float]],
    a: Optional[float],
    b: Optional[float],
    residual: float,
    scale: str,
) -> float:
    """Score axis fit quality for confidence ranking.

    Higher score = more reliable mapping.
    """
    if a is None:
        return 0.0

    score = 0.0

    # Tick count (weight: 10 per tick)
    score += len(ticks) * 10

    # Residual penalty
    if residual < 100:
        score += 50
    elif residual < 1000:
        score += 30
    elif residual < 10000:
        score += 10
    else:
        score -= 50

    # Monotonicity check
    values = [v for _, v in ticks]
    pixels = [p for p, _ in ticks]
    order = np.argsort(pixels)
    ordered_values = [values[i] for i in order]
    diffs = np.diff(ordered_values)
    if len(diffs) > 0:
        monotonic_ratio = max(
            np.mean(diffs > 0),
            np.mean(diffs < 0)
        )
        if monotonic_ratio > 0.9:
            score += 40
        elif monotonic_ratio > 0.7:
            score += 20

    # Sequence regularity check
    if scale == "linear" and _is_arithmetic_sequence(np.array(values), tol=0.3):
        score += 30
    if scale == "log" and _is_geometric_sequence(np.array(values), tol=0.3):
        score += 30

    # Clamp score to 0-100
    return max(0.0, min(100.0, score))
