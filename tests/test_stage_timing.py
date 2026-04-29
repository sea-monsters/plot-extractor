"""Tests for per-stage timing telemetry.

TDD tests for StageTiming dataclass and timing integration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Test 1: StageTiming dataclass has required fields ---

def test_stage_timing_dataclass_fields():
    """StageTiming must carry all pipeline stage durations."""
    from plot_extractor.core.label_crop_planner import StageTiming

    timing = StageTiming(
        crop_planning_ms=12.5,
        formula_infer_ms=45.3,
        calibrate_ms=33.1,
        total_ms=90.9,
    )
    assert timing.crop_planning_ms == 12.5
    assert timing.formula_infer_ms == 45.3
    assert timing.calibrate_ms == 33.1
    assert timing.total_ms == 90.9


# --- Test 2: StageTiming default values are zero ---

def test_stage_timing_defaults():
    """Default StageTiming should have all zeros."""
    from plot_extractor.core.label_crop_planner import StageTiming

    timing = StageTiming()
    assert timing.crop_planning_ms == 0.0
    assert timing.formula_infer_ms == 0.0
    assert timing.calibrate_ms == 0.0
    assert timing.total_ms == 0.0


# --- Test 3: StageTiming to_dict produces serializable output ---

def test_stage_timing_to_dict():
    """StageTiming should serialize to a plain dict."""
    from plot_extractor.core.label_crop_planner import StageTiming

    timing = StageTiming(
        crop_planning_ms=12.5,
        formula_infer_ms=45.3,
        calibrate_ms=33.1,
        total_ms=90.9,
    )
    d = timing.to_dict()
    assert isinstance(d, dict)
    assert d["crop_planning_ms"] == 12.5
    assert d["formula_infer_ms"] == 45.3
    assert d["calibrate_ms"] == 33.1
    assert d["total_ms"] == 90.9


if __name__ == "__main__":
    test_stage_timing_dataclass_fields()
    print("[PASS] test_stage_timing_dataclass_fields")
    test_stage_timing_defaults()
    print("[PASS] test_stage_timing_defaults")
    test_stage_timing_to_dict()
    print("[PASS] test_stage_timing_to_dict")
    print("[SUCCESS] stage timing tests passed")
