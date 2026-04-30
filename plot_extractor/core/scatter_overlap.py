"""Overlapping scatter point separation.

Based on 0809.1802 (2008 pure-CV 2D plot extraction):
detects abnormally large connected components (indicating overlapping
scatter points) and separates them using greedy shape matching.

The original paper uses simulated annealing; we use a greedy approach
for lower computational cost while maintaining effectiveness.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ShapeTemplate:
    mean_area: float
    mean_radius: float


def detect_overlap_candidates(
    cc_stats: list,
    area_ratio: float = 1.5,
    min_cc_count: int = 5,
) -> list:
    """Identify CCs whose area is significantly larger than the median.

    Args:
        cc_stats: List of CC stat dicts with 'area' key.
        area_ratio: CC must be > area_ratio * median_area to be flagged.
        min_cc_count: Need at least this many CCs to estimate median.

    Returns:
        List of CC stat dicts flagged as overlap candidates.
    """
    if len(cc_stats) < min_cc_count:
        return []

    areas = [cc["area"] for cc in cc_stats]
    median_area = float(np.median(areas))
    if median_area == 0:
        return []

    threshold = area_ratio * median_area
    return [cc for cc in cc_stats if cc["area"] > threshold]


def extract_template_from_ccs(
    mask: np.ndarray,
    cc_stats: list,
    max_outlier_ratio: float = 2.0,
) -> Optional[ShapeTemplate]:
    """Extract a shape template from non-overlapping CCs.

    Uses the median CC as a representative shape, filtering outliers.

    Args:
        mask: Binary foreground mask.
        cc_stats: List of CC stat dicts.
        max_outlier_ratio: Filter CCs with area > max_outlier_ratio * median.

    Returns:
        ShapeTemplate with mean_area and mean_radius, or None.
    """
    if len(cc_stats) < 2:
        return None

    areas = np.array([cc["area"] for cc in cc_stats])
    median_area = float(np.median(areas))

    # Filter out overlap candidates
    regular = [cc for cc in cc_stats if cc["area"] <= max_outlier_ratio * median_area]
    if not regular:
        return None

    regular_areas = np.array([cc["area"] for cc in regular])
    mean_area = float(np.mean(regular_areas))
    # Estimate radius from area (assuming circular markers)
    mean_radius = np.sqrt(mean_area / np.pi)

    return ShapeTemplate(mean_area=mean_area, mean_radius=mean_radius)


def separate_overlap_greedy(
    mask: np.ndarray,
    overlap_cc: dict,
    template: ShapeTemplate,
    max_splits: int = 5,
) -> List[tuple]:
    """Separate an overlapping CC into individual point centroids.

    Greedy approach:
    1. Find the pixel with highest local density in the overlap region
    2. Mark a circular region of template radius around it
    3. Repeat on remaining pixels

    Args:
        mask: Binary foreground mask (uint8, >0 = foreground).
        overlap_cc: CC stat dict for the overlapping region.
        template: Shape template from non-overlapping points.
        max_splits: Maximum number of splits to attempt.

    Returns:
        List of (cx, cy) centroid estimates for each split point.
    """
    r = max(2, int(round(template.mean_radius)))
    x, y, cw, ch = overlap_cc["x"], overlap_cc["y"], overlap_cc["w"], overlap_cc["h"]

    # Extract local patch
    h, w = mask.shape
    x2 = min(x + cw, w)
    y2 = min(y + ch, h)
    patch = mask[y:y2, x:x2].copy()

    if patch.sum() == 0:
        return []

    centroids = []
    remaining = (patch > 0).astype(np.uint8)

    for _ in range(max_splits):
        if remaining.sum() == 0:
            break

        # Find pixel with highest local density
        ys_fg, xs_fg = np.where(remaining > 0)
        if len(xs_fg) == 0:
            break

        # Compute local density for each foreground pixel
        best_score = -1
        best_xy = None

        # Sample up to 50 candidate positions to avoid O(n^2)
        indices = np.linspace(0, len(xs_fg) - 1, min(50, len(xs_fg)), dtype=int)
        for idx in indices:
            px, py = int(xs_fg[idx]), int(ys_fg[idx])
            # Count foreground pixels in template-radius disk
            count = _count_in_disk(remaining, px, py, r)
            if count > best_score:
                best_score = count
                best_xy = (px, py)

        if best_xy is None:
            break

        px, py = best_xy
        centroids.append((x + px, y + py))

        # Remove the disk from remaining
        _clear_disk(remaining, px, py, r)

    return centroids


def _count_in_disk(mask: np.ndarray, cx: int, cy: int, r: int) -> int:
    h, w = mask.shape
    y_lo, y_hi = max(0, cy - r), min(h, cy + r + 1)
    x_lo, x_hi = max(0, cx - r), min(w, cx + r + 1)
    patch = mask[y_lo:y_hi, x_lo:x_hi]
    ys, xs = np.ogrid[y_lo - cy: y_hi - cy, x_lo - cx: x_hi - cx]
    disk = (xs ** 2 + ys ** 2) <= r ** 2
    return int(patch[disk].sum())


def _clear_disk(mask: np.ndarray, cx: int, cy: int, r: int):
    h, w = mask.shape
    y_lo, y_hi = max(0, cy - r), min(h, cy + r + 1)
    x_lo, x_hi = max(0, cx - r), min(w, cx + r + 1)
    ys, xs = np.ogrid[y_lo - cy: y_hi - cy, x_lo - cx: x_hi - cx]
    disk = (xs ** 2 + ys ** 2) <= r ** 2
    mask[y_lo:y_hi, x_lo:x_hi][disk] = 0
