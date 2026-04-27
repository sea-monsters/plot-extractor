"""Test multi-candidate axis solving."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.axis_candidates import (
    solve_axis_multi_candidate,
    AxisCandidatesResult,
)
from plot_extractor.core.axis_detector import Axis


def create_mock_axis(direction="x", position=100, plot_start=50, plot_end=450):
    """Create mock Axis for testing."""
    axis = Axis(
        direction=direction,
        side="bottom" if direction == "x" else "left",
        position=position,
        plot_start=plot_start,
        plot_end=plot_end,
        ticks=[(100, None), (200, None), (300, None), (400, None)],
    )
    return axis


def test_solve_from_ocr_linear():
    """Test OCR-derived linear candidate."""
    axis = create_mock_axis()

    # Good linear OCR ticks
    ocr_ticks = [(100, 0.0), (200, 5.0), (300, 10.0), (400, 15.0)]

    result = solve_axis_multi_candidate(axis, ocr_ticks)

    assert result.best is not None
    assert result.best.scale == "linear"
    assert result.best.source == "ocr"
    assert result.best.confidence > 50
    print("[PASS] OCR linear candidate test passed")


def test_solve_from_ocr_log():
    """Test OCR-derived log candidate."""
    axis = create_mock_axis()

    # Good log OCR ticks
    ocr_ticks = [(100, 1.0), (200, 10.0), (300, 100.0), (400, 1000.0)]

    result = solve_axis_multi_candidate(axis, ocr_ticks, preferred_type="log")

    assert result.best is not None
    assert result.best.scale == "log"
    assert result.best.source == "ocr"
    assert result.best.confidence > 50
    print("[PASS] OCR log candidate test passed")


def test_fallback_heuristic():
    """Test heuristic fallback when OCR unavailable."""
    axis = create_mock_axis()

    # No OCR labels
    ocr_ticks = [(100, None), (200, None)]
    result = solve_axis_multi_candidate(axis, ocr_ticks)

    # Should have heuristic candidate
    heuristic = [c for c in result.all if c.source == "heuristic"]
    assert len(heuristic) > 0
    print("[PASS] Heuristic fallback test passed")


def test_multi_candidate_sorting():
    """Test candidates are sorted by confidence."""
    axis = create_mock_axis()

    # Good OCR ticks
    ocr_ticks = [(100, 0.0), (200, 5.0), (300, 10.0), (400, 15.0)]

    result = solve_axis_multi_candidate(axis, ocr_ticks)

    # Should have multiple candidates
    assert len(result.all) >= 2

    # Check sorted by confidence (descending)
    for i in range(len(result.all) - 1):
        assert result.all[i].confidence >= result.all[i + 1].confidence

    print(f"[PASS] Candidate sorting test passed (found {len(result.all)} candidates)")
    print(f"  Top candidate: {result.best.source} (confidence={result.best.confidence:.1f})")


if __name__ == "__main__":
    print("Running axis candidates tests...")
    test_solve_from_ocr_linear()
    test_solve_from_ocr_log()
    test_fallback_heuristic()
    test_multi_candidate_sorting()
    print("\n[SUCCESS] All tests passed")