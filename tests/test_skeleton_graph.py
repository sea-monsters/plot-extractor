"""Tests for skeleton graph traversal for multi-series separation.

Covers:
- build_skeleton_graph: convert binary skeleton to NetworkX graph
- extract_branches: continuous branches between endpoints/junctions
- assign_branches_to_series: group branches into n_series curves
- branches_to_mask: render branches back to binary mask
"""
import numpy as np
from plot_extractor.core.skeleton_graph import (
    build_skeleton_graph,
    extract_branches,
    assign_branches_to_series,
    branches_to_mask,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic skeletons
# ---------------------------------------------------------------------------

def _horizontal_line(length=20, y=10, x_start=5):
    """Create a horizontal line skeleton."""
    img = np.zeros((20, 40), dtype=np.uint8)
    img[y, x_start:x_start + length] = 1
    return img


def _two_parallel_lines(length=20, y1=8, y2=12, x_start=5):
    """Create two parallel horizontal lines."""
    img = np.zeros((20, 40), dtype=np.uint8)
    img[y1, x_start:x_start + length] = 1
    img[y2, x_start:x_start + length] = 1
    return img


def _x_shape(size=11):
    """Create an X-shaped skeleton (two crossing lines)."""
    img = np.zeros((size + 4, size + 4), dtype=np.uint8)
    c = (size + 4) // 2
    for i in range(size):
        offset = i - size // 2
        img[c + offset, c + offset] = 1
        img[c + offset, c - offset] = 1
    return img


def _cross_shape(arm=8):
    """Create a + (cross) shape."""
    img = np.zeros((2 * arm + 5, 2 * arm + 5), dtype=np.uint8)
    c = arm + 2
    img[c - arm:c + arm + 1, c] = 1
    img[c, c - arm:c + arm + 1] = 1
    return img


def _line_with_spur(length=15, spur_len=3, y=10, x_start=5, spur_y_offset=-3):
    """Create a line with a short spur branch."""
    img = np.zeros((25, 40), dtype=np.uint8)
    spur_x = x_start + length // 2
    img[y, x_start:x_start + length] = 1
    for i in range(spur_len):
        img[y + spur_y_offset + i, spur_x] = 1
    img[y, spur_x] = 1
    return img


# ---------------------------------------------------------------------------
# build_skeleton_graph
# ---------------------------------------------------------------------------

class TestBuildSkeletonGraph:
    def test_horizontal_line_has_two_endpoints(self):
        skel = _horizontal_line(length=20)
        graph = build_skeleton_graph(skel)
        endpoints = [n for n, d in graph.nodes(data=True)
                     if d.get("node_type") == "endpoint"]
        assert len(endpoints) == 2

    def test_cross_has_junctions(self):
        """Pixel-level cross creates a cluster of junction nodes."""
        skel = _cross_shape(arm=8)
        graph = build_skeleton_graph(skel)
        junctions = [n for n, d in graph.nodes(data=True)
                     if d.get("node_type") == "junction"]
        assert len(junctions) >= 1

    def test_x_shape_has_four_endpoints(self):
        skel = _x_shape(size=11)
        graph = build_skeleton_graph(skel)
        endpoints = [n for n, d in graph.nodes(data=True)
                     if d.get("node_type") == "endpoint"]
        assert len(endpoints) == 4

    def test_x_shape_has_one_junction(self):
        skel = _x_shape(size=11)
        graph = build_skeleton_graph(skel)
        junctions = [n for n, d in graph.nodes(data=True)
                     if d.get("node_type") == "junction"]
        assert len(junctions) == 1

    def test_empty_skeleton_returns_empty_graph(self):
        skel = np.zeros((20, 20), dtype=np.uint8)
        graph = build_skeleton_graph(skel)
        assert len(graph.nodes) == 0

    def test_single_pixel_is_endpoint(self):
        skel = np.zeros((10, 10), dtype=np.uint8)
        skel[5, 5] = 1
        graph = build_skeleton_graph(skel)
        endpoints = [n for n, d in graph.nodes(data=True)
                     if d.get("node_type") == "endpoint"]
        assert len(endpoints) == 1


# ---------------------------------------------------------------------------
# extract_branches
# ---------------------------------------------------------------------------

class TestExtractBranches:
    def test_horizontal_line_has_one_branch(self):
        skel = _horizontal_line(length=20)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        assert len(branches) == 1
        assert len(branches[0]) >= 20

    def test_x_shape_has_four_branches(self):
        skel = _x_shape(size=11)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        assert len(branches) == 4

    def test_cross_has_at_least_four_branches(self):
        """Pixel-level cross junction cluster produces multiple branches."""
        skel = _cross_shape(arm=8)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        assert len(branches) >= 4

    def test_empty_graph_returns_empty(self):
        graph = build_skeleton_graph(np.zeros((10, 10), dtype=np.uint8))
        branches = extract_branches(graph)
        assert not branches


# ---------------------------------------------------------------------------
# assign_branches_to_series
# ---------------------------------------------------------------------------

class TestAssignBranchesToSeries:
    def test_two_parallel_lines_assigns_two_series(self):
        skel = _two_parallel_lines(length=20)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        series = assign_branches_to_series(branches, n_series=2)
        assert len(series) == 2
        for s in series:
            assert len(s) > 0

    def test_crossing_lines_assigns_two_series(self):
        skel = _x_shape(size=11)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        series = assign_branches_to_series(branches, n_series=2)
        assert len(series) == 2
        total_branches = sum(len(s) for s in series)
        assert total_branches == len(branches)

    def test_single_series_gets_all_branches(self):
        skel = _horizontal_line(length=20)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        series = assign_branches_to_series(branches, n_series=1)
        assert len(series) == 1
        assert len(series[0]) == len(branches)

    def test_fewer_branches_than_series(self):
        """If n_series > branches, some series are empty."""
        skel = _horizontal_line(length=10)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        series = assign_branches_to_series(branches, n_series=3)
        assert len(series) == 3
        assert sum(len(s) for s in series) == len(branches)


# ---------------------------------------------------------------------------
# branches_to_mask
# ---------------------------------------------------------------------------

class TestBranchesToMask:
    def test_roundtrip_preserves_pixels(self):
        skel = _horizontal_line(length=15)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        reconstructed = branches_to_mask(branches, skel.shape)
        assert np.sum(reconstructed > 0) >= np.sum(skel > 0) * 0.8

    def test_empty_branches_produce_empty_mask(self):
        mask = branches_to_mask([], (20, 40))
        assert np.sum(mask) == 0

    def test_output_shape_matches_input(self):
        skel = _horizontal_line(length=15)
        graph = build_skeleton_graph(skel)
        branches = extract_branches(graph)
        reconstructed = branches_to_mask(branches, skel.shape)
        assert reconstructed.shape == skel.shape
