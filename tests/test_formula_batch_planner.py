"""Tests for FormulaOCR batch planning."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.axis_detector import Axis
from plot_extractor.core.axis_calibrator import (
    _plan_formula_batch_requests,
    _should_use_formula_label_values,
)


def _make_axis(direction: str, side: str, position: int, ticks: list[int]) -> Axis:
    return Axis(
        direction=direction,
        side=side,
        position=position,
        plot_start=min(ticks) - 10,
        plot_end=max(ticks) + 10,
        ticks=[(tick, None) for tick in ticks],
    )


def _make_anchor(tick: int, text: str = "10^{2}", source: str = "synthetic_ocr", value=None):
    return SimpleNamespace(
        tick_pixel=tick,
        crop=np.full((16, 16, 3), 255, dtype=np.uint8),
        label_bbox=(tick - 4, tick - 4, tick + 4, tick + 4),
        tesseract_text=text,
        tesseract_value=value,
        confidence=0.8,
        source=source,
    )


def test_formula_batch_planner_caps_total_requests():
    """Planner should keep the batch compact and retain both log axes."""
    axis_y = _make_axis("y", "left", 60, [40, 100, 160])
    axis_x = _make_axis("x", "bottom", 120, [50, 120, 190])

    axis_anchor_map = {
        id(axis_y): [_make_anchor(40), _make_anchor(100), _make_anchor(160)],
        id(axis_x): [_make_anchor(50), _make_anchor(120), _make_anchor(190)],
    }
    axis_is_log = {id(axis_y): True, id(axis_x): True}

    plan = _plan_formula_batch_requests([axis_y, axis_x], axis_anchor_map, axis_is_log, max_total_crops=4)

    assert plan.requested_count == 4
    assert plan.kept_count == 4
    assert plan.dropped_count == 0
    assert plan.batch_size_hint == 4
    assert len(plan.requests) == 4
    assert any(req.axis_id == id(axis_y) for req in plan.requests)
    assert any(req.axis_id == id(axis_x) for req in plan.requests)


def test_formula_label_override_requires_multiple_formula_values():
    """A single formula read should not override a healthy tesseract axis."""
    assert not _should_use_formula_label_values(1, 1, 0.75)
    assert not _should_use_formula_label_values(2, 1, 0.75)
    assert not _should_use_formula_label_values(1, 2, 0.75)
    assert _should_use_formula_label_values(2, 2, 0.75)


def test_formula_batch_planner_skips_plain_axis_with_tesseract_values():
    """Planner should stay sparse when tesseract already reads a plain axis."""
    axis_x = _make_axis("x", "bottom", 120, [50, 120, 190])
    axis_anchor_map = {
        id(axis_x): [
            _make_anchor(50, text="1", source="tesseract", value=1.0),
            _make_anchor(120, text="2", source="tesseract", value=2.0),
            _make_anchor(190, text="3", source="tesseract", value=3.0),
        ]
    }
    axis_is_log = {id(axis_x): False}

    plan = _plan_formula_batch_requests([axis_x], axis_anchor_map, axis_is_log)

    assert plan.requested_count == 0
    assert plan.kept_count == 0
    assert plan.batch_size_hint == 0
    assert plan.requests == []


if __name__ == "__main__":
    test_formula_batch_planner_caps_total_requests()
    test_formula_label_override_requires_multiple_formula_values()
    test_formula_batch_planner_skips_plain_axis_with_tesseract_values()
    print("[SUCCESS] Formula batch planner tests passed")
