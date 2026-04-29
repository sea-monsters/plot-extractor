"""Tests for candidate_map priority table: fixed priority ordering.

TDD tests for the build_candidate_maps function that replaces the
scattered conditional branching in calibrate_all_axes.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_anchor(tick, text="10", value=None, formula_value=None, formula_latex=None):
    return SimpleNamespace(
        tick_pixel=tick,
        tesseract_text=text,
        tesseract_value=value,
        formula_value=formula_value,
        formula_latex=formula_latex,
        source="tesseract" if value is not None else "missing",
    )


# --- Test 1: Formula-generated log ticks are highest priority ---

def test_formula_generated_is_highest_priority():
    """When formula can generate log ticks, it should be the sole candidate."""
    from plot_extractor.core.label_crop_planner import build_candidate_maps
    from plot_extractor.core.axis_detector import Axis

    # Y-axis: values increase upward, pixels decrease
    # pixel 160 (bottom) → 10^0, pixel 100 → 10^2, pixel 40 (top) → 10^4
    axis = Axis(direction="y", side="left", position=60, plot_start=30, plot_end=170,
                ticks=[(40, None), (100, None), (160, None)])
    anchors = [
        _make_anchor(40, value=10000.0, formula_value=10000.0, formula_latex="10^{4}"),
        _make_anchor(100, value=100.0, formula_value=100.0, formula_latex="10^{2}"),
        _make_anchor(160, value=1.0, formula_value=1.0, formula_latex="10^0"),
    ]
    result = build_candidate_maps(
        formula_log_score=0.85,
        formula_value_count=3,
        formula_anchor_count=3,
        tesseract_count=3,
        label_count=3,
        anchors=anchors,
        formula_labeled=[(40, 10000.0), (100, 100.0), (160, 1.0)],
        fused_labeled=[(40, 10000.0), (100, 100.0), (160, 1.0)],
        tesseract_labeled=[(40, 10000.0), (100, 100.0), (160, 1.0)],
        axis=axis,
    )
    assert len(result) >= 1
    assert result[0][0] == "formula_generated"


# --- Test 2: Fused takes priority over tesseract when formula evidence exists ---

def test_fused_over_tesseract_when_formula_partial():
    """Fused (tesseract + formula correction) > raw tesseract."""
    from plot_extractor.core.label_crop_planner import build_candidate_maps

    anchors = [
        _make_anchor(40, value=1.0),
        _make_anchor(100, value=10.0, formula_value=100.0),
        _make_anchor(160, value=100.0),
    ]
    result = build_candidate_maps(
        formula_log_score=0.2,
        formula_value_count=1,
        formula_anchor_count=1,
        tesseract_count=3,
        label_count=3,
        anchors=anchors,
        formula_labeled=[(100, 100.0)],
        fused_labeled=[(100, 100.0), (40, 1.0), (160, 100.0)],
        tesseract_labeled=[(40, 1.0), (100, 10.0), (160, 100.0)],
        axis=None,
    )
    sources = [src for src, _ in result]
    assert "tesseract" in sources
    if "fused" in sources:
        assert sources.index("fused") < sources.index("tesseract")


# --- Test 3: Heuristic is always last resort ---

def test_heuristic_is_last_resort():
    """When no OCR is available, heuristic should be the only candidate."""
    from plot_extractor.core.label_crop_planner import build_candidate_maps

    result = build_candidate_maps(
        formula_log_score=0.0,
        formula_value_count=0,
        formula_anchor_count=0,
        tesseract_count=0,
        label_count=0,
        anchors=[],
        formula_labeled=[],
        fused_labeled=[],
        tesseract_labeled=[],
        axis=None,
        tick_pixels=[50, 100, 150],
    )
    assert len(result) == 1
    assert result[0][0] == "heuristic"


# --- Test 4: Priority order is formula_generated > formula > fused > tesseract > heuristic ---

def test_priority_ordering():
    """Candidate maps should follow the fixed priority table."""
    from plot_extractor.core.label_crop_planner import CANDIDATE_PRIORITY

    expected_order = ["formula_generated", "formula", "fused", "tesseract", "heuristic"]
    assert list(CANDIDATE_PRIORITY.keys()) == expected_order


# --- Test 5: Formula label values used when tesseract is sparse ---

def test_formula_labels_when_tesseract_sparse():
    """When tesseract count is low and formula has values, formula should appear."""
    from plot_extractor.core.label_crop_planner import build_candidate_maps

    anchors = [
        _make_anchor(40, text="", value=None, formula_value=100.0, formula_latex="10^{2}"),
        _make_anchor(100, text="", value=None, formula_value=1000.0, formula_latex="10^{3}"),
        _make_anchor(160, text="", value=None),
    ]
    result = build_candidate_maps(
        formula_log_score=0.35,
        formula_value_count=2,
        formula_anchor_count=2,
        tesseract_count=0,
        label_count=2,
        anchors=anchors,
        formula_labeled=[(40, 100.0), (100, 1000.0)],
        fused_labeled=[(40, 100.0), (100, 1000.0)],
        tesseract_labeled=[(40, None), (100, None), (160, None)],
        axis=None,
    )
    sources = [src for src, _ in result]
    assert "formula" in sources


if __name__ == "__main__":
    test_formula_generated_is_highest_priority()
    print("[PASS] test_formula_generated_is_highest_priority")
    test_fused_over_tesseract_when_formula_partial()
    print("[PASS] test_fused_over_tesseract_when_formula_partial")
    test_heuristic_is_last_resort()
    print("[PASS] test_heuristic_is_last_resort")
    test_priority_ordering()
    print("[PASS] test_priority_ordering")
    test_formula_labels_when_tesseract_sparse()
    print("[PASS] test_formula_labels_when_tesseract_sparse")
    print("[SUCCESS] candidate priority tests passed")
