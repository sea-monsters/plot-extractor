"""Detect chart axes, tick marks, and plot area."""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from plot_extractor.config import (
    HOUGH_THRESHOLD,
    MIN_LINE_LENGTH,
    MAX_LINE_GAP,
    AXIS_ANGLE_TOLERANCE,
)


@dataclass
class Axis:
    direction: str  # "x" or "y"
    position: int   # pixel coordinate of the axis line
    side: str       # "bottom", "top", "left", "right"
    plot_start: int # start of plot area along the orthogonal direction
    plot_end: int   # end of plot area along the orthogonal direction
    ticks: List[Tuple[int, float]] = None  # (pixel, raw_value_if_known)


def _angle_deg(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return abs(np.degrees(np.arctan2(dy, dx)))


def _is_horizontal(angle):
    return angle <= AXIS_ANGLE_TOLERANCE or angle >= 180 - AXIS_ANGLE_TOLERANCE


def _is_vertical(angle):
    return 90 - AXIS_ANGLE_TOLERANCE <= angle <= 90 + AXIS_ANGLE_TOLERANCE


def detect_axes(image_gray, edges=None):
    """Detect main x and y axes from the image."""
    h, w = image_gray.shape
    if edges is None:
        edges = cv2.Canny(image_gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    horiz_lines = []
    vert_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = _angle_deg(x1, y1, x2, y2)
            if _is_horizontal(angle):
                horiz_lines.append((x1, y1, x2, y2))
            elif _is_vertical(angle):
                vert_lines.append((x1, y1, x2, y2))

    axes = []
    # Detect X axes (horizontal lines near top or bottom)
    if horiz_lines:
        # Group by y coordinate
        ys = []
        for (x1, y1, x2, y2) in horiz_lines:
            ys.append((y1 + y2) / 2)
        ys = np.array(ys)
        # Bottom axis: y near bottom of image
        bottom_candidates = [y for y in ys if y > h * 0.8]
        if bottom_candidates:
            bottom_y = int(np.median(bottom_candidates))
            axes.append(Axis(
                direction="x",
                position=bottom_y,
                side="bottom",
                plot_start=0,
                plot_end=w,
                ticks=[]
            ))
        # Top axis (optional, e.g. for twin x) — must be very near top edge
        top_candidates = [y for y in ys if y < h * 0.15]
        if top_candidates:
            top_y = int(np.median(top_candidates))
            if not any(a.side == "bottom" for a in axes) or abs(top_y - bottom_y) > h * 0.3:
                axes.append(Axis(
                    direction="x",
                    position=top_y,
                    side="top",
                    plot_start=0,
                    plot_end=w,
                    ticks=[]
                ))

    # Detect Y axes (vertical lines near left or right)
    if vert_lines:
        xs = []
        for (x1, y1, x2, y2) in vert_lines:
            xs.append((x1 + x2) / 2)
        xs = np.array(xs)
        left_candidates = [x for x in xs if x < w * 0.2]
        if left_candidates:
            left_x = int(np.median(left_candidates))
            axes.append(Axis(
                direction="y",
                position=left_x,
                side="left",
                plot_start=0,
                plot_end=h,
                ticks=[]
            ))
        right_candidates = [x for x in xs if x > w * 0.8]
        if right_candidates:
            right_x = int(np.median(right_candidates))
            if not any(a.side == "left" for a in axes) or abs(right_x - left_x) > w * 0.3:
                axes.append(Axis(
                    direction="y",
                    position=right_x,
                    side="right",
                    plot_start=0,
                    plot_end=h,
                    ticks=[]
                ))

    return axes


def _filter_regular_ticks(candidates, min_ticks=4, tolerance=0.25):
    """From candidate tick positions, find the most regularly spaced subset."""
    if len(candidates) < min_ticks:
        return candidates
    candidates = sorted(set(candidates))
    best_subset = candidates[:]
    best_score = 0
    # Try different step sizes
    diffs = [candidates[i+1] - candidates[i] for i in range(len(candidates)-1)]
    if not diffs:
        return candidates
    # Use median diff as candidate step, and also common divisors
    test_steps = [np.median(diffs)]
    # Also try smaller steps that divide larger gaps evenly
    for d in diffs:
        if d > 0:
            for n in range(2, 6):
                test_steps.append(d / n)
    for step in test_steps:
        if step <= 1:
            continue
        # Try aligning to each candidate as start
        for start in candidates:
            subset = []
            for c in candidates:
                offset = abs(c - start)
                remainder = offset % step
                norm_rem = min(remainder, step - remainder)
                if norm_rem <= step * tolerance:
                    subset.append(c)
            if len(subset) >= min_ticks:
                score = len(subset) * 10 - np.std([candidates[i+1]-candidates[i] for i in range(len(candidates)-1)])
                if score > best_score:
                    best_score = score
                    best_subset = subset
    return sorted(best_subset)


def _find_peaks_1d(profile, min_height_ratio=0.3, min_distance=10):
    """Simple 1D peak detection without scipy."""
    if len(profile) == 0:
        return []
    smoothed = np.convolve(profile, np.ones(3) / 3, mode="same")
    max_val = np.max(smoothed)
    if max_val == 0:
        return []
    threshold = max_val * min_height_ratio
    # Find local maxima
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(i)
    # Enforce minimum distance
    if not peaks:
        return []
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
    return filtered


def detect_ticks(image_gray, axis: Axis, edges=None):
    """Detect tick marks along an axis using projection method."""
    h, w = image_gray.shape
    if edges is None:
        edges = cv2.Canny(image_gray, 50, 150)

    strip = 8  # pixel strip around axis to look for ticks

    if axis.direction == "x":
        y0 = max(0, axis.position - strip)
        y1 = min(h, axis.position + strip)
        strip_edges = edges[y0:y1, :]
        profile = np.sum(strip_edges, axis=0).astype(float)
        profile[:axis.plot_start] = 0
        profile[axis.plot_end:] = 0
        ticks = _find_peaks_1d(profile, min_height_ratio=0.3,
                               min_distance=max(10, w // 50))
    else:
        x0 = max(0, axis.position - strip)
        x1 = min(w, axis.position + strip)
        strip_edges = edges[:, x0:x1]
        profile = np.sum(strip_edges, axis=1).astype(float)
        profile[:axis.plot_start] = 0
        profile[axis.plot_end:] = 0
        ticks = _find_peaks_1d(profile, min_height_ratio=0.3,
                               min_distance=max(10, h // 50))

    ticks = sorted(set(ticks))
    # Filter to regular subset for primary axes
    if axis.side in ("bottom", "left"):
        ticks = _filter_regular_ticks(ticks, min_ticks=3, tolerance=0.3)
    axis.ticks = [(int(t), None) for t in ticks]
    return axis


def refine_plot_area(axes, image_shape):
    """Refine plot start/end for each axis based on crossing axes."""
    h, w = image_shape[:2]
    x_axes = [a for a in axes if a.direction == "x"]
    y_axes = [a for a in axes if a.direction == "y"]

    for xa in x_axes:
        lefts = [ya.position for ya in y_axes if ya.side == "left"]
        rights = [ya.position for ya in y_axes if ya.side == "right"]
        xa.plot_start = max(lefts) if lefts else 0
        xa.plot_end = min(rights) if rights else w

    for ya in y_axes:
        bottoms = [xa.position for xa in x_axes if xa.side == "bottom"]
        tops = [xa.position for xa in x_axes if xa.side == "top"]
        ya.plot_start = min(tops) if tops else 0
        ya.plot_end = max(bottoms) if bottoms else h

    return axes


def detect_all_axes(image_gray):
    """Full axis detection pipeline."""
    edges = cv2.Canny(image_gray, 50, 150)
    axes = detect_axes(image_gray, edges)
    axes = refine_plot_area(axes, image_gray.shape)
    for axis in axes:
        detect_ticks(image_gray, axis, edges)
    # Filter out secondary axes (top/right) that have too few ticks
    # or whose ticks match primary axis grid-line projections
    primary = [a for a in axes if a.side in ("bottom", "left")]
    secondary = [a for a in axes if a.side in ("top", "right")]

    def _tick_positions(axis):
        return [t[0] for t in axis.ticks]

    def _pattern_match(ticks_a, ticks_b, tol=5):
        if not ticks_a or not ticks_b:
            return False
        matches = 0
        for a in ticks_a:
            if any(abs(a - b) <= tol for b in ticks_b):
                matches += 1
        return matches >= max(3, len(ticks_a) * 0.6)

    h, w = image_gray.shape
    for sec in secondary:
        sec_ticks = _tick_positions(sec)
        # Determine if this secondary axis sits at an image edge (likely a spine)
        at_edge = False
        if sec.direction == "x" and sec.side == "top" and sec.position < h * 0.2:
            at_edge = True
        elif sec.direction == "x" and sec.side == "bottom" and sec.position > h * 0.8:
            at_edge = True
        elif sec.direction == "y" and sec.side == "left" and sec.position < w * 0.2:
            at_edge = True
        elif sec.direction == "y" and sec.side == "right" and sec.position > w * 0.8:
            at_edge = True

        # Check if this secondary axis is at a different position from primary axes
        # (i.e., it's a real secondary axis, not a duplicate line)
        position_differs = True
        for pri in primary:
            if pri.direction == sec.direction and abs(sec.position - pri.position) < 20:
                position_differs = False
                break

        if at_edge and position_differs:
            # Edge spine: keep it for plot bounds even if ticks match primary
            # (dual Y-axis charts have matching tick positions but different positions)
            primary.append(sec)
        elif len(sec_ticks) >= 4:
            # Reject if tick pattern matches a primary axis (likely grid lines)
            matched_primary = False
            for pri in primary:
                if pri.direction == sec.direction and _pattern_match(sec_ticks, _tick_positions(pri)):
                    matched_primary = True
                    break
            if not matched_primary:
                primary.append(sec)
    return primary
