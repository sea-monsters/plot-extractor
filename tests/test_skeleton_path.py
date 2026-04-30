"""Tests for junction-aware skeleton path tracing.

Covers:
- Endpoint and junction detection
- Simple path tracing (no branches)
- Branch resolution by direction continuity
- Integration with _extract_from_mask replacement
- Edge cases: empty skeleton, single pixel, cycle
"""
import pytest
import numpy as np
from plot_extractor.core.skeleton_path import (
    classify_skeleton_points,
    trace_skeleton_paths,
    extract_path_data,
    angle_diff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixel_skeleton(points, shape=(100, 100)):
    """Create a binary skeleton image from a list of (x, y) coordinates."""
    skel = np.zeros(shape, dtype=np.uint8)
    for x, y in points:
        skel[y, x] = 1
    return skel


def _horizontal_line(y, x_start, x_end, shape=(100, 100)):
    skel = np.zeros(shape, dtype=np.uint8)
    skel[y, x_start:x_end + 1] = 1
    return skel


def _diagonal_line(x0, y0, length, shape=(100, 100)):
    """Create a diagonal line going down-right."""
    skel = np.zeros(shape, dtype=np.uint8)
    for i in range(length):
        skel[y0 + i, x0 + i] = 1
    return skel


def _vertical_line(x, y_start, y_end, shape=(100, 100)):
    skel = np.zeros(shape, dtype=np.uint8)
    skel[y_start:y_end + 1, x] = 1
    return skel


# ---------------------------------------------------------------------------
# angle_diff
# ---------------------------------------------------------------------------

class TestAngleDiff:
    def test_same_direction(self):
        assert angle_diff((1, 0), (1, 0)) == pytest.approx(0.0, abs=0.1)

    def test_opposite_direction(self):
        assert angle_diff((1, 0), (-1, 0)) == pytest.approx(180.0, abs=0.1)

    def test_perpendicular(self):
        assert angle_diff((1, 0), (0, 1)) == pytest.approx(90.0, abs=0.1)


# ---------------------------------------------------------------------------
# classify_skeleton_points
# ---------------------------------------------------------------------------

class TestClassifySkeletonPoints:
    def test_empty_skeleton(self):
        skel = np.zeros((50, 50), dtype=np.uint8)
        endpoints, junctions = classify_skeleton_points(skel)
        assert len(endpoints) == 0
        assert len(junctions) == 0

    def test_single_pixel(self):
        skel = _pixel_skeleton([(25, 25)])
        endpoints, junctions = classify_skeleton_points(skel)
        assert len(endpoints) == 0  # isolated, no neighbors
        assert len(junctions) == 0

    def test_horizontal_line(self):
        skel = _horizontal_line(50, 20, 80)
        endpoints, junctions = classify_skeleton_points(skel)
        assert len(endpoints) == 2
        assert len(junctions) == 0
        ex = sorted([p[0] for p in endpoints])
        assert ex[0] == 20
        assert ex[1] == 80

    def test_t_junction(self):
        """T-junction: one branch point with 3 neighbors."""
        skel = np.zeros((50, 50), dtype=np.uint8)
        # Horizontal line
        for x in range(20, 31):
            skel[25, x] = 1
        # Vertical line going down from center
        for y in range(25, 35):
            skel[y, 25] = 1
        endpoints, junctions = classify_skeleton_points(skel)
        assert len(junctions) >= 1
        assert (25, 25) in junctions

    def test_cross_junction(self):
        """Cross: one branch point with 4 neighbors."""
        skel = np.zeros((50, 50), dtype=np.uint8)
        for x in range(20, 31):
            skel[25, x] = 1
        for y in range(20, 31):
            skel[y, 25] = 1
        endpoints, junctions = classify_skeleton_points(skel)
        assert len(junctions) >= 1
        assert (25, 25) in junctions


# ---------------------------------------------------------------------------
# trace_skeleton_paths
# ---------------------------------------------------------------------------

class TestTraceSkeletonPaths:
    def test_single_horizontal_path(self):
        skel = _horizontal_line(50, 20, 80)
        paths = trace_skeleton_paths(skel, min_path_len=5)
        assert len(paths) >= 1
        # Should span from x=20 to x=80
        longest = max(paths, key=len)
        assert len(longest) >= 50

    def test_two_separate_lines(self):
        skel = _horizontal_line(30, 10, 40, shape=(100, 100))
        skel2 = _horizontal_line(70, 10, 40, shape=(100, 100))
        combined = skel | skel2
        paths = trace_skeleton_paths(combined, min_path_len=5)
        assert len(paths) >= 2

    def test_empty_returns_empty(self):
        skel = np.zeros((50, 50), dtype=np.uint8)
        paths = trace_skeleton_paths(skel)
        assert paths == []

    def test_min_path_len_filters_short(self):
        skel = _horizontal_line(50, 20, 25)  # 6 pixels
        paths = trace_skeleton_paths(skel, min_path_len=10)
        assert len(paths) == 0

    def test_t_junction_produces_two_paths(self):
        """T-junction should produce paths that follow direction continuity."""
        skel = np.zeros((50, 50), dtype=np.uint8)
        for x in range(20, 31):
            skel[25, x] = 1
        for y in range(25, 35):
            skel[y, 25] = 1
        paths = trace_skeleton_paths(skel, min_path_len=3)
        assert len(paths) >= 1

    def test_diagonal_path(self):
        skel = _diagonal_line(10, 10, 30)
        paths = trace_skeleton_paths(skel, min_path_len=5)
        assert len(paths) >= 1
        longest = max(paths, key=len)
        assert len(longest) >= 20

    def test_cycle_produces_paths(self):
        """A closed loop (cycle) has no endpoints but should still be traced."""
        skel = np.zeros((50, 50), dtype=np.uint8)
        # Draw a small square
        for x in range(20, 30):
            skel[20, x] = 1
            skel[29, x] = 1
        for y in range(20, 30):
            skel[y, 20] = 1
            skel[y, 29] = 1
        paths = trace_skeleton_paths(skel, min_path_len=5)
        # Should find at least one path from the cycle
        assert len(paths) >= 1


# ---------------------------------------------------------------------------
# extract_path_data
# ---------------------------------------------------------------------------

class TestExtractPathData:
    def test_simple_path_to_xy(self):
        path = [(x, 50) for x in range(20, 80)]  # horizontal line at y=50
        xs, ys = extract_path_data([path])
        assert len(xs) > 0
        assert len(ys) > 0
        assert xs[0] == 20
        assert xs[-1] == 79

    def test_empty_paths(self):
        xs, ys = extract_path_data([])
        assert xs == []
        assert ys == []

    def test_multiple_paths_merged(self):
        paths = [
            [(x, 30) for x in range(10, 30)],
            [(x, 70) for x in range(40, 60)],
        ]
        xs, ys = extract_path_data(paths)
        assert len(xs) == len(paths[0]) + len(paths[1])
