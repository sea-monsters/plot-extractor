"""Unit tests for absolute log tick solver."""
from plot_extractor.core.axis_calibrator import (
    _canonical_log_value_score,
    _solve_absolute_log_ticks,
)
from plot_extractor.core.axis_detector import Axis


def _axis(ticks):
    return Axis(
        direction="x",
        position=100,
        side="bottom",
        plot_start=min(ticks),
        plot_end=max(ticks),
        ticks=[(t, None) for t in ticks],
    )


def test_superscript_loss_values_are_not_canonical_bonus():
    """OCR-flattened values like 103 should not outrank real powers."""
    assert _canonical_log_value_score(100.0) == 1.0
    assert _canonical_log_value_score(103.0) < 0.5


def test_absolute_solver_decodes_103_as_power_candidate():
    """A right-edge 103 read can be reinterpreted instead of accepted literally."""
    ticks = [77, 89, 101, 121, 183, 214, 236, 254, 268, 280, 299, 361, 393, 415, 447, 459, 478]
    solved, diagnostics = _solve_absolute_log_ticks(
        _axis(ticks),
        ticks,
        anchors=[(478.0, 103.0)],
        preferred_type="log",
    )
    assert solved is not None
    assert diagnostics["candidate_count"] > 1
    assert diagnostics["best_candidate_value"] != 103.0
    values = [v for _p, v in solved]
    assert max(values) / min(values) >= 100.0
