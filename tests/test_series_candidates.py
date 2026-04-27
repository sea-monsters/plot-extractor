"""Unit tests for series_candidates module."""
import numpy as np
import cv2
from pathlib import Path

from plot_extractor.core.series_candidates import (
    extract_series_multi_candidate,
    _score_series_candidate,
    SeriesMappingCandidate,
)
from plot_extractor.core.axis_calibrator import CalibratedAxis, Axis
from plot_extractor.core.axis_candidates import AxisMappingCandidate


def test_series_candidate_scoring():
    """Test series confidence scoring."""
    # Mock calibrated axis
    axis = Axis(
        direction="x",
        side="bottom",
        position=400,
        plot_start=100,
        plot_end=500,
        ticks=[(100, 0.0), (200, 2.0), (300, 4.0), (400, 6.0), (500, 8.0)],
    )

    candidate = AxisMappingCandidate(
        scale="linear",
        a=50.0,
        b=100.0,
        residual=100.0,
        confidence=80.0,
        source="ocr",
        tick_map=[(100, 0.0), (200, 2.0), (300, 4.0), (400, 6.0), (500, 8.0)],
    )

    cal_axis = CalibratedAxis(
        axis=axis,
        axis_type="linear",
        a=50.0,
        b=100.0,
        residual=100.0,
        inverted=False,
        tick_map=[(100, 0.0), (200, 2.0), (300, 4.0), (400, 6.0), (500, 8.0)],
    )

    # Test: good continuity, good coverage
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ys = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    result = _score_series_candidate("test_series", xs, ys, cal_axis, "test")

    print(f"Continuity score: {result.continuity_score:.2f}")
    print(f"Coverage score: {result.coverage_score:.2f}")
    print(f"Point count: {result.point_count}")
    print(f"Confidence: {result.confidence:.2f}")

    assert result.continuity_score > 80.0, "Expected high continuity for uniform spacing"
    assert result.coverage_score > 50.0, "Expected reasonable coverage"
    assert result.confidence > 60.0, "Expected good overall confidence"

    print("[PASS] Series candidate scoring test")


def test_empty_series():
    """Test handling of empty series."""
    axis = Axis(
        direction="x",
        side="bottom",
        position=400,
        plot_start=100,
        plot_end=500,
        ticks=[],
    )

    cal_axis = CalibratedAxis(
        axis=axis,
        axis_type="linear",
        a=50.0,
        b=100.0,
        residual=0.0,
        inverted=False,
        tick_map=[],
    )

    result = _score_series_candidate("empty", [], [], cal_axis, "test")

    assert result.confidence == 0.0, "Empty series should have zero confidence"
    assert result.point_count == 0, "Empty series should have zero points"

    print("[PASS] Empty series test")


def test_cc_centroid_extraction():
    """Test connected component centroid extraction."""
    # Create mock mask with 3 blobs
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (20, 50), 5, 255, -1)
    cv2.circle(mask, (40, 60), 5, 255, -1)
    cv2.circle(mask, (60, 40), 5, 255, -1)

    # Mock calibrated axes
    x_axis = Axis(
        direction="x",
        side="bottom",
        position=100,
        plot_start=0,
        plot_end=100,
        ticks=[],
    )

    y_axis = Axis(
        direction="y",
        side="left",
        position=0,
        plot_start=0,
        plot_end=100,
        ticks=[],
    )

    x_cal = CalibratedAxis(
        axis=x_axis,
        axis_type="linear",
        a=1.0,
        b=0.0,
        residual=0.0,
        inverted=False,
        tick_map=[],
    )

    y_cal = CalibratedAxis(
        axis=y_axis,
        axis_type="linear",
        a=1.0,
        b=0.0,
        residual=0.0,
        inverted=False,
        tick_map=[],
    )

    # Test extraction
    from plot_extractor.core.series_candidates import _extract_cc_centroid

    result = _extract_cc_centroid(mask, x_cal, y_cal)

    print(f"CC centroid extracted {len(result)} series")
    if result:
        name, xs, ys = result[0]
        print(f"  Series name: {name}")
        print(f"  Points: {len(xs)}")

    print("[PASS] CC centroid extraction test")


if __name__ == "__main__":
    test_series_candidate_scoring()
    test_empty_series()
    test_cc_centroid_extraction()
    print("\n[SUCCESS] All series_candidates tests passed")