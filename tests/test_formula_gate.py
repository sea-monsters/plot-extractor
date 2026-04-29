"""Tests for is_processable_for_formula: single-point FormulaOCR gate.

TDD tests written BEFORE implementation.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_anchor_meta(
    tesseract_count: int = 0,
    label_count: int = 0,
    formula_log_score: float = 0.0,
) -> dict:
    return {
        "anchor_count": tesseract_count + 2,
        "tesseract_count": tesseract_count,
        "label_count": label_count,
        "geometry_label_count": label_count,
        "selected_count": 0,
        "formula_log_score": formula_log_score,
        "selected_indices": [],
    }


def _make_anchor(text="10^{2}", value=None, confidence=0.8, source="synthetic_ocr", is_far=False):
    anchor = SimpleNamespace(
        tick_pixel=100,
        crop=__import__("numpy").full((16, 16, 3), 255, dtype=__import__("numpy").uint8),
        label_bbox=(96, 96, 104, 104),
        tesseract_text=text,
        tesseract_value=value,
        confidence=confidence,
        source=source,
        formula_latex=None,
        formula_value=None,
    )
    return anchor


# --- Test 1: Log axis always triggers ---

def test_log_axis_triggers_formula():
    """Axis already detected as log should always be processable."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    assert is_processable_for_formula(
        axis_is_log=True,
        tesseract_count=3,
        label_count=3,
        anchors=[_make_anchor(value=100.0) for _ in range(3)],
    ) is True


# --- Test 2: Sparse tesseract triggers ---

def test_sparse_tesseract_triggers():
    """When tesseract labels are sparse (< 3), trigger FormulaOCR."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=1,
        label_count=3,
        anchors=[_make_anchor(text="", value=None) for _ in range(3)],
    ) is True


# --- Test 3: Well-populated linear axis does NOT trigger ---

def test_well_populated_linear_does_not_trigger():
    """Linear axis with enough tesseract values should not trigger FormulaOCR."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    anchors = [_make_anchor(text=str(i), value=float(i), source="tesseract") for i in range(5)]
    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=4,
        label_count=5,
        anchors=anchors,
    ) is False


# --- Test 4: Suspected exponential format triggers ---

def test_suspected_exponential_triggers():
    """Text containing ^ or scientific notation should trigger."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    anchors = [_make_anchor(text="10^2", value=None)]
    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=0,
        label_count=1,
        anchors=anchors,
    ) is True


# --- Test 5: All-empty anchors do NOT trigger ---

def test_all_empty_anchors_do_not_trigger():
    """No labels at all should not trigger FormulaOCR."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    anchors = [_make_anchor(text="", value=None) for _ in range(3)]
    for a in anchors:
        a.tesseract_text = None
    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=0,
        label_count=0,
        anchors=anchors,
    ) is False


# --- Test 6: 100-110 range (OCR superscript misread) triggers ---

def test_hundreds_range_triggers():
    """Values in 100-110 range (superscript misread) should trigger."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    anchors = [_make_anchor(value=102.0), _make_anchor(value=104.0)]
    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=2,
        label_count=2,
        anchors=anchors,
    ) is True


# --- Test 7: 10-19 range (missing superscript) triggers ---

def test_teens_range_triggers():
    """Values in 10-19 range (missing superscript) should trigger."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    anchors = [_make_anchor(value=12.0), _make_anchor(value=14.0)]
    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=2,
        label_count=2,
        anchors=anchors,
    ) is True


# --- Test 8: Suppress on minor tick blank ---

def test_suppress_on_minor_tick_only():
    """All anchors are blank minor ticks → suppress."""
    from plot_extractor.core.label_crop_planner import is_processable_for_formula

    anchors = []
    for _ in range(4):
        a = _make_anchor(text="", value=None)
        a.tesseract_text = None
        a.crop = __import__("numpy").empty((0, 0), dtype=__import__("numpy").uint8)
        anchors.append(a)
    assert is_processable_for_formula(
        axis_is_log=False,
        tesseract_count=0,
        label_count=0,
        anchors=anchors,
    ) is False


if __name__ == "__main__":
    test_log_axis_triggers_formula()
    print("[PASS] test_log_axis_triggers_formula")
    test_sparse_tesseract_triggers()
    print("[PASS] test_sparse_tesseract_triggers")
    test_well_populated_linear_does_not_trigger()
    print("[PASS] test_well_populated_linear_does_not_trigger")
    test_suspected_exponential_triggers()
    print("[PASS] test_suspected_exponential_triggers")
    test_all_empty_anchors_do_not_trigger()
    print("[PASS] test_all_empty_anchors_do_not_trigger")
    test_hundreds_range_triggers()
    print("[PASS] test_hundreds_range_triggers")
    test_teens_range_triggers()
    print("[PASS] test_teens_range_triggers")
    test_suppress_on_minor_tick_only()
    print("[PASS] test_suppress_on_minor_tick_only")
    print("[SUCCESS] is_processable_for_formula tests passed")
