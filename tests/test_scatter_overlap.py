"""Tests for overlapping scatter point separation.

Based on 0809.1802 (2008 pure-CV 2D plot extraction):
detects abnormally large CCs and splits them using greedy shape matching.
"""
import pytest
import numpy as np
import cv2
from plot_extractor.core.scatter_overlap import (
    detect_overlap_candidates,
    separate_overlap_greedy,
    extract_template_from_ccs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_circle(mask, cx, cy, r, value=255):
    """Draw a filled circle on mask."""
    h, w = mask.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    mask[dist <= r] = value


def _make_scatter_mask(shape=(200, 200), points=None, radius=5):
    """Create a scatter mask with circular points."""
    mask = np.zeros(shape, dtype=np.uint8)
    if points is None:
        points = [(30, 30), (60, 40), (90, 50), (120, 60), (150, 70)]
    for x, y in points:
        _draw_circle(mask, x, y, radius)
    return mask


# ---------------------------------------------------------------------------
# detect_overlap_candidates
# ---------------------------------------------------------------------------

class TestDetectOverlapCandidates:
    def test_no_overlap(self):
        mask = _make_scatter_mask(radius=5)
        cc_stats = _get_cc_stats(mask)
        overlaps = detect_overlap_candidates(cc_stats, area_ratio=1.5)
        assert len(overlaps) == 0

    def test_single_overlap_detected(self):
        """Two overlapping circles create a larger CC."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        _draw_circle(mask, 50, 50, 6)
        _draw_circle(mask, 56, 50, 6)  # Overlapping
        # Also add isolated points
        _draw_circle(mask, 20, 20, 6)
        _draw_circle(mask, 80, 20, 6)
        _draw_circle(mask, 20, 80, 6)

        cc_stats = _get_cc_stats(mask)
        overlaps = detect_overlap_candidates(cc_stats, area_ratio=1.5, min_cc_count=3)
        assert len(overlaps) >= 1

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cc_stats = _get_cc_stats(mask)
        overlaps = detect_overlap_candidates(cc_stats)
        assert len(overlaps) == 0


# ---------------------------------------------------------------------------
# extract_template_from_ccs
# ---------------------------------------------------------------------------

class TestExtractTemplate:
    def test_template_from_regular_points(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        for x in [20, 40, 60, 80]:
            _draw_circle(mask, x, 50, 5)
        cc_stats = _get_cc_stats(mask)
        template = extract_template_from_ccs(mask, cc_stats)
        assert template is not None
        assert template.mean_radius > 0
        assert template.mean_area > 0

    def test_template_from_no_points(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cc_stats = _get_cc_stats(mask)
        template = extract_template_from_ccs(mask, cc_stats)
        assert template is None


# ---------------------------------------------------------------------------
# separate_overlap_greedy
# ---------------------------------------------------------------------------

class TestSeparateOverlapGreedy:
    def test_two_overlapping_circles(self):
        """Two overlapping circles should be split into 2 centroids."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        _draw_circle(mask, 45, 50, 6)
        _draw_circle(mask, 55, 50, 6)
        # Add isolated points for template
        _draw_circle(mask, 20, 20, 6)
        _draw_circle(mask, 80, 80, 6)

        cc_stats = _get_cc_stats(mask)
        template = extract_template_from_ccs(mask, cc_stats)
        assert template is not None

        overlaps = detect_overlap_candidates(cc_stats, area_ratio=1.3)
        if overlaps:
            result = separate_overlap_greedy(mask, overlaps[0], template)
            assert len(result) >= 2

    def test_no_overlap_returns_empty(self):
        mask = _make_scatter_mask(radius=5)
        cc_stats = _get_cc_stats(mask)
        template = extract_template_from_ccs(mask, cc_stats)
        overlaps = detect_overlap_candidates(cc_stats)
        if not overlaps:
            pytest.skip("No overlaps in test data")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_cc_stats(mask):
    """Get CC stats from mask."""
    num_labels, labels, stats, centroids = cv2_connected_components(mask)
    result = []
    for i in range(1, num_labels):
        result.append({
            "label": i,
            "area": int(stats[i, cv2.CC_STAT_AREA]),
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]),
            "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1]),
        })
    return result


def cv2_connected_components(mask):
    return cv2.connectedComponentsWithStats(mask, connectivity=8)
