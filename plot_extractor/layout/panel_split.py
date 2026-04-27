"""Panel boundary detection and splitting.

Detects multi-panel figures (common in scientific papers with (a)(b)(c) subplots)
and splits them into individual panels for independent processing.
"""
from dataclasses import dataclass
from typing import List
import numpy as np
import cv2


@dataclass
class Panel:
    """Single panel (subplot) region."""
    panel_id: int
    rect: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    image: np.ndarray


def detect_panel_boundaries(image: np.ndarray) -> List[Panel]:
    """
    Detect panel boundaries in multi-panel figures.

    Methods:
    1. Large whitespace gaps (vertical/horizontal splits)

    If no clear boundaries, return single panel (full image).
    Each candidate panel is validated for axis presence before accepting the split.
    """
    h, w = image.shape[:2]

    # Method 1: Large whitespace gaps (vertical/horizontal splits)
    all_candidates = _detect_whitespace_gaps(image)

    if not all_candidates:
        # No clear split: return full image as single panel
        return [Panel(0, (0, 0, w, h), image)]

    # Merge overlapping candidates
    merged = _merge_panel_candidates(all_candidates)

    if len(merged) < 2:
        return [Panel(0, (0, 0, w, h), image)]

    # Split into panels and validate each has plausible axis structure
    panels = []
    for i, rect in enumerate(merged):
        x0, y0, x1, y1 = rect
        panel_img = image[y0:y1, x0:x1]
        if _has_axis_structure(panel_img):
            panels.append(Panel(i, rect, panel_img))

    # If validation rejected panels, fall back to single panel
    if len(panels) < 2:
        return [Panel(0, (0, 0, w, h), image)]

    # Renumber panel IDs
    for i, panel in enumerate(panels):
        panel.panel_id = i

    return panels


def _has_axis_structure(panel_img: np.ndarray) -> bool:
    """Check if a panel image has at least 2 axis-like line structures."""
    if panel_img.size == 0:
        return False
    h, w = panel_img.shape[:2]
    if h < 50 or w < 50:
        return False
    gray = cv2.cvtColor(panel_img, cv2.COLOR_RGB2GRAY) if len(panel_img.shape) == 3 else panel_img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return False
    horiz = sum(1 for l in lines if abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 10 or
                abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) > 170)
    vert = sum(1 for l in lines if 80 < abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 100)
    return horiz >= 2 and vert >= 2


def _detect_whitespace_gaps(image: np.ndarray) -> List[tuple]:
    """Detect large whitespace gaps that indicate panel splits."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    # Threshold to find background (whitespace)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Vertical splits: find columns with very high white ratio (>98%)
    col_white_ratio = np.sum(binary == 255, axis=0) / h
    vertical_split_cols = np.where(col_white_ratio > 0.98)[0]

    # Horizontal splits: find rows with very high white ratio (>98%)
    row_white_ratio = np.sum(binary == 255, axis=1) / w
    horizontal_split_rows = np.where(row_white_ratio > 0.98)[0]

    candidates = []

    # Vertical split candidates
    if len(vertical_split_cols) > 0:
        # Find continuous regions of white columns (gaps)
        # Require minimum 50px gap to reduce false positives from matplotlib padding
        gaps_v = _find_continuous_gaps(vertical_split_cols, min_length=50)
        for gap in gaps_v:
            # Left panel
            left_rect = (0, 0, gap[0], h)
            # Right panel
            right_rect = (gap[1], 0, w, h)
            if _is_valid_panel_rect(left_rect, image):
                candidates.append(left_rect)
            if _is_valid_panel_rect(right_rect, image):
                candidates.append(right_rect)

    # Horizontal split candidates
    if len(horizontal_split_rows) > 0:
        gaps_h = _find_continuous_gaps(horizontal_split_rows, min_length=50)
        for gap in gaps_h:
            # Top panel
            top_rect = (0, 0, w, gap[0])
            # Bottom panel
            bottom_rect = (0, gap[1], w, h)
            if _is_valid_panel_rect(top_rect, image):
                candidates.append(top_rect)
            if _is_valid_panel_rect(bottom_rect, image):
                candidates.append(bottom_rect)

    return candidates


def _find_continuous_gaps(positions: np.ndarray, min_length: int = 10) -> List[tuple]:
    """Find continuous regions in sorted position array."""
    if len(positions) < min_length:
        return []

    gaps = []
    start = positions[0]
    prev = positions[0]

    for pos in positions[1:]:
        if pos - prev > 2:  # Gap in continuity
            if prev - start >= min_length:
                gaps.append((start, prev))
            start = pos
        prev = pos

    # Final gap
    if prev - start >= min_length:
        gaps.append((start, prev))

    return gaps


def _cluster_positions(positions: List[int], tolerance: int = 20) -> List[List[int]]:
    """Cluster nearby positions."""
    if not positions:
        return []

    clusters = []
    current_cluster = [positions[0]]

    for pos in positions[1:]:
        if pos - current_cluster[-1] <= tolerance:
            current_cluster.append(pos)
        else:
            clusters.append(current_cluster)
            current_cluster = [pos]

    clusters.append(current_cluster)
    return clusters


def _merge_panel_candidates(rects: List[tuple]) -> List[tuple]:
    """Merge overlapping panel candidates."""
    if not rects:
        return []

    # Sort by area (descending)
    sorted_rects = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)

    merged = []
    for rect in sorted_rects:
        # Check if this rect overlaps with any merged rect
        overlaps = False
        for merged_rect in merged:
            if _rects_overlap(rect, merged_rect, threshold=0.5):
                overlaps = True
                break
        if not overlaps:
            merged.append(rect)

    return merged


def _rects_overlap(r1: tuple, r2: tuple, threshold: float = 0.5) -> bool:
    """Check if two rectangles overlap significantly."""
    x0_1, y0_1, x1_1, y1_1 = r1
    x0_2, y0_2, x1_2, y1_2 = r2

    # Intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)

    if x1_i <= x0_i or y1_i <= y0_i:
        return False  # No intersection

    intersection_area = (x1_i - x0_i) * (y1_i - y0_i)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

    # Overlap ratio
    overlap_ratio = intersection_area / min(area1, area2)

    return overlap_ratio > threshold


def _is_valid_panel_rect(rect: tuple, image: np.ndarray) -> bool:
    """Check if panel rect is valid (non-trivial size)."""
    x0, y0, x1, y1 = rect
    h, w = image.shape[:2]

    width = x1 - x0
    height = y1 - y0

    # Must be at least 10% of image size
    min_width = w * 0.1
    min_height = h * 0.1

    return width >= min_width and height >= min_height
