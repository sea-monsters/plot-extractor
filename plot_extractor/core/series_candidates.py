"""Series multi-candidate extraction framework.

Extracts data series using multiple strategies and ranks by confidence.
"""
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

from plot_extractor.core.axis_calibrator import CalibratedAxis
from plot_extractor.config import MIN_DATA_POINTS, FOREGROUND_MIN_AREA


@dataclass
class SeriesMappingCandidate:
    """Single series extraction candidate."""
    series_name: str
    x_data: List[float]
    y_data: List[float]
    continuity_score: float  # 0-100, measures point spacing regularity
    coverage_score: float  # 0-100, measures x-axis coverage
    point_count: int
    confidence: float  # weighted average: continuity*0.4 + coverage*0.3 + count*0.3
    source: str  # "color_path" | "color_median" | "cc_centroid"


@dataclass
class SeriesCandidatesResult:
    """All series candidates from multiple extraction methods."""
    all: List[SeriesMappingCandidate]
    best: List[SeriesMappingCandidate]  # Top candidates per series
    warnings: List[str]


def extract_series_multi_candidate(
    image: np.ndarray,
    mask: np.ndarray,
    x_cal: CalibratedAxis,
    y_cal: CalibratedAxis,
    is_scatter: bool = False,
) -> SeriesCandidatesResult:
    """
    Extract series using multiple strategies.

    Strategies:
    1. Color clustering + path tracking (existing method)
    2. Color clustering + column median
    3. Connected component centroid (scatter-focused)

    Args:
        image: RGB plot image
        mask: Foreground mask
        x_cal: Calibrated x-axis
        y_cal: Calibrated y-axis
        is_scatter: Whether chart is scatter type

    Returns:
        SeriesCandidatesResult with ranked candidates
    """
    candidates = []
    warnings = []

    # Strategy 1: Color path tracking (existing)
    try:
        path_series = _extract_color_path_tracking(image, mask, x_cal, y_cal)
        for name, xs, ys in path_series:
            candidate = _score_series_candidate(name, xs, ys, x_cal, "color_path")
            candidates.append(candidate)
    except Exception as e:
        warnings.append(f"Color path tracking failed: {str(e)}")

    # Strategy 2: Color median (if color separation succeeded)
    try:
        median_series = _extract_color_median(image, mask, x_cal, y_cal)
        for name, xs, ys in median_series:
            candidate = _score_series_candidate(name, xs, ys, x_cal, "color_median")
            candidates.append(candidate)
    except Exception as e:
        warnings.append(f"Color median extraction failed: {str(e)}")

    # Strategy 3: Connected component centroid (scatter or dense)
    if is_scatter or len(candidates) < 2:
        try:
            cc_series = _extract_cc_centroid(mask, x_cal, y_cal)
            for name, xs, ys in cc_series:
                candidate = _score_series_candidate(name, xs, ys, x_cal, "cc_centroid")
                candidates.append(candidate)
        except Exception as e:
            warnings.append(f"CC centroid extraction failed: {str(e)}")

    # Sort by confidence and select best per series
    if not candidates:
        warnings.append("No series extracted from any method")
        return SeriesCandidatesResult(all=[], best=[], warnings=warnings)

    candidates.sort(key=lambda c: c.confidence, reverse=True)

    # Deduplicate by series name (keep highest confidence)
    best_by_name = {}
    for cand in candidates:
        if cand.series_name not in best_by_name:
            best_by_name[cand.series_name] = cand

    best = list(best_by_name.values())

    return SeriesCandidatesResult(all=candidates, best=best, warnings=warnings)


def _extract_color_path_tracking(
    image: np.ndarray,
    mask: np.ndarray,
    x_cal: CalibratedAxis,
    y_cal: CalibratedAxis,
) -> List[Tuple[str, List[float], List[float]]]:
    """Extract via color clustering + path tracking."""
    from plot_extractor.core.data_extractor import _separate_series_by_color, _extract_from_mask

    # Separate series by color
    series_list = _separate_series_by_color(image, mask, n_clusters=5)

    if not series_list:
        return []

    results = []
    for idx, (_, series_img, series_mask) in enumerate(series_list):
        # Path tracking via _extract_from_mask
        xs, ys = _extract_from_mask(series_mask, x_cal, y_cal)

        if len(xs) >= MIN_DATA_POINTS:
            series_name = f"series_{idx+1}"
            results.append((series_name, xs, ys))

    return results


def _extract_color_median(
    image: np.ndarray,
    mask: np.ndarray,
    x_cal: CalibratedAxis,
    y_cal: CalibratedAxis,
) -> List[Tuple[str, List[float], List[float]]]:
    """Extract via color clustering + per-column median y."""
    from plot_extractor.core.data_extractor import _separate_series_by_color

    # Separate series by color
    series_list = _separate_series_by_color(image, mask, n_clusters=5)

    if not series_list:
        return []

    results = []
    for idx, (_, series_img, series_mask) in enumerate(series_list):
        h, w = series_mask.shape

        # Extract per-column median y
        xs = []
        ys = []
        for x_col in range(w):
            col_mask = series_mask[:, x_col]
            fg_pixels = np.where(col_mask > 0)[0]

            if len(fg_pixels) >= 3:  # Minimum pixels to compute median
                y_median = int(np.median(fg_pixels))
                xd = x_cal.to_data(x_col)
                yd = y_cal.to_data(y_median)

                if xd is not None and yd is not None:
                    xs.append(xd)
                    ys.append(yd)

        if len(xs) >= MIN_DATA_POINTS:
            series_name = f"series_{idx+1}_median"
            results.append((series_name, xs, ys))

    return results


def _extract_cc_centroid(
    mask: np.ndarray,
    x_cal: CalibratedAxis,
    y_cal: CalibratedAxis,
) -> List[Tuple[str, List[float], List[float]]]:
    """Extract via connected component centroid."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    xs = []
    ys = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < FOREGROUND_MIN_AREA:
            continue
        cx, cy = centroids[i]
        xd = x_cal.to_data(int(cx))
        yd = y_cal.to_data(int(cy))
        if xd is not None and yd is not None:
            xs.append(xd)
            ys.append(yd)

    if len(xs) < MIN_DATA_POINTS:
        return []

    return [("series_0", xs, ys)]


def _score_series_candidate(
    name: str,
    xs: List[float],
    ys: List[float],
    x_cal: CalibratedAxis,
    source: str,
) -> SeriesMappingCandidate:
    """
    Score series extraction quality.

    Factors:
    - Continuity: adjacent point spacing regularity
    - Coverage: x-axis range coverage
    - Point count: number of extracted points
    """
    if not xs or not ys:
        return SeriesMappingCandidate(
            series_name=name,
            x_data=xs,
            y_data=ys,
            continuity_score=0.0,
            coverage_score=0.0,
            point_count=0,
            confidence=0.0,
            source=source,
        )

    # Continuity: measure spacing variance
    if len(xs) > 1:
        sorted_xs = sorted(xs)
        gaps = [sorted_xs[i+1] - sorted_xs[i] for i in range(len(sorted_xs) - 1)]
        if gaps:
            gap_std = np.std(gaps)
            gap_mean = np.mean(gaps)
            # Low variance = high continuity
            continuity = 100.0 * (1.0 - min(gap_std / (gap_mean + 1e-6), 1.0))
        else:
            continuity = 50.0
    else:
        continuity = 10.0

    # Coverage: x-axis range coverage
    x_min = float(min(xs))
    x_max = float(max(xs))
    axis_min = float(min(v for _, v in x_cal.tick_map)) if x_cal.tick_map else x_min
    axis_max = float(max(v for _, v in x_cal.tick_map)) if x_cal.tick_map else x_max
    axis_range = axis_max - axis_min
    series_range = x_max - x_min
    coverage = 100.0 * (series_range / (axis_range + 1e-6)) if axis_range > 0 else 50.0

    # Point count: scale logarithmically
    count_score = min(100.0, 10.0 * np.log10(len(xs) + 1))

    # Weighted confidence
    confidence = continuity * 0.4 + coverage * 0.3 + count_score * 0.3

    return SeriesMappingCandidate(
        series_name=name,
        x_data=xs,
        y_data=ys,
        continuity_score=continuity,
        coverage_score=coverage,
        point_count=len(xs),
        confidence=confidence,
        source=source,
    )
