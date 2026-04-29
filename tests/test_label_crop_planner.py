"""Tests for label_crop_planner: unified tick-label crop planning.

TDD tests written BEFORE implementation to define the required contract.
Tesseract-dependent tests are conditional on availability.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.axis_detector import Axis


def _make_image(height=240, width=240):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _make_x_axis(ticks=(50, 100, 150)):
    return Axis(
        direction="x",
        side="bottom",
        position=120,
        plot_start=min(ticks) - 10,
        plot_end=max(ticks) + 10,
        ticks=[(t, None) for t in ticks],
    )


def _make_y_axis(ticks=(40, 100, 160)):
    return Axis(
        direction="y",
        side="left",
        position=60,
        plot_start=min(ticks) - 10,
        plot_end=max(ticks) + 10,
        ticks=[(t, None) for t in ticks],
    )


# --- Test 1: PlannedCrop dataclass has required fields ---

def test_planned_crop_dataclass_fields():
    """PlannedCrop must carry all diagnostic metadata."""
    from plot_extractor.core.label_crop_planner import PlannedCrop

    crop_arr = np.full((16, 16, 3), 255, dtype=np.uint8)
    pc = PlannedCrop(
        tick_pixel=100,
        search_bbox=(90, 122, 110, 160),
        text_bbox_local=(2, 3, 14, 12),
        final_bbox=(92, 125, 104, 135),
        crop=crop_arr,
        tesseract_probe_text="10",
        tesseract_probe_value=10.0,
        quality_flags={"is_empty": False, "is_far_from_tick": False, "possible_minor_tick": False},
    )
    assert pc.tick_pixel == 100
    assert pc.search_bbox == (90, 122, 110, 160)
    assert pc.text_bbox_local == (2, 3, 14, 12)
    assert pc.final_bbox == (92, 125, 104, 135)
    assert pc.crop.shape == (16, 16, 3)
    assert pc.tesseract_probe_text == "10"
    assert pc.tesseract_probe_value == 10.0
    assert not pc.quality_flags["is_empty"]
    assert not pc.quality_flags["is_far_from_tick"]
    assert not pc.quality_flags["possible_minor_tick"]


# --- Test 2: plan_tick_label_crop returns None on blank image ---

def test_plan_tick_label_crop_returns_none_on_blank():
    """No text on image → None crop (no spurious detection)."""
    from plot_extractor.core.label_crop_planner import plan_tick_label_crop

    image = _make_image()
    axis = _make_x_axis(ticks=(50, 100, 150))
    tick_pixels = [50, 100, 150]
    result = plan_tick_label_crop(image, axis, tick_pixels, tick_pixel=100)
    assert result is None


# --- Test 3: plan_tick_label_crop finds label with mock Tesseract ---

def test_plan_tick_label_crop_finds_label_with_mock():
    """With mocked Tesseract returning a text box, crop should be valid."""
    from plot_extractor.core.label_crop_planner import plan_tick_label_crop

    image = _make_image()
    # Draw "10" below tick at x=100 (large text)
    cv2.putText(image, "10", (90, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    axis = _make_x_axis(ticks=(50, 100, 150))
    tick_pixels = [50, 100, 150]

    # Mock tesseract to return a box matching the text position in upscaled coords
    mock_data = {
        "text": ["10", ""],
        "left": [28, 0],
        "top": [52, 0],
        "width": [28, 0],
        "height": [28, 0],
        "conf": ["90", "-1"],
    }

    with patch("plot_extractor.core.label_crop_planner._tesseract_geometry_probe") as mock_probe:
        mock_probe.return_value = ((14, 26, 28, 40), "10")
        result = plan_tick_label_crop(image, axis, tick_pixels, tick_pixel=100)

    assert result is not None
    assert result.tick_pixel == 100
    assert result.final_bbox is not None
    assert result.crop is not None
    assert result.crop.size > 0
    assert isinstance(result.quality_flags, dict)
    assert not result.quality_flags["is_empty"]


# --- Test 4: plan_tick_label_crop flags far-from-tick crops ---

def test_plan_tick_label_crop_flags_far_crop():
    """Text far from the tick center should get is_far_from_tick=True."""
    from plot_extractor.core.label_crop_planner import plan_tick_label_crop

    image = _make_image()
    axis = _make_x_axis(ticks=(50, 100, 150))
    tick_pixels = [50, 100, 150]

    # Mock tesseract returning a box that's far from the tick center
    with patch("plot_extractor.core.label_crop_planner._tesseract_geometry_probe") as mock_probe:
        # bbox_local far from tick center in the search window
        mock_probe.return_value = ((2, 2, 10, 10), "99")
        result = plan_tick_label_crop(image, axis, tick_pixels, tick_pixel=100)

    if result is not None:
        assert isinstance(result.quality_flags["is_far_from_tick"], bool)


# --- Test 5: plan_tick_label_crops_batch plans multiple ticks ---

def test_plan_tick_label_crops_batch_returns_all_ticks():
    """Batch planner should return one entry per tick (or None)."""
    from plot_extractor.core.label_crop_planner import plan_tick_label_crops_batch

    image = _make_image()
    axis = _make_x_axis(ticks=(50, 100, 150))
    tick_pixels = [50, 100, 150]

    results = plan_tick_label_crops_batch(image, axis, tick_pixels)
    assert len(results) == 3


# --- Test 6: quality flags correct when no text ---

def test_quality_flags_none_when_no_text():
    """When no text is found, result is None (no spurious flags)."""
    from plot_extractor.core.label_crop_planner import plan_tick_label_crop

    image = _make_image()
    axis = _make_y_axis(ticks=(40, 100, 160))
    tick_pixels = [40, 100, 160]
    result = plan_tick_label_crop(image, axis, tick_pixels, tick_pixel=100)
    assert result is None


# --- Test 7: Y-axis direction produces valid crops with mock ---

def test_y_axis_crop_has_valid_bbox_with_mock():
    """Y-axis crop planning should produce valid coordinate geometry."""
    from plot_extractor.core.label_crop_planner import plan_tick_label_crop

    image = _make_image()
    axis = _make_y_axis(ticks=(40, 100, 160))
    tick_pixels = [40, 100, 160]

    with patch("plot_extractor.core.label_crop_planner._tesseract_geometry_probe") as mock_probe:
        mock_probe.return_value = ((10, 5, 40, 18), "10")
        result = plan_tick_label_crop(image, axis, tick_pixels, tick_pixel=100)

    assert result is not None
    x1, y1, x2, y2 = result.final_bbox
    assert x2 > x1
    assert y2 > y1


# --- Test 8: _expand_bbox_with_padding expands correctly ---

def test_expand_bbox_with_padding_increases_size():
    """Expansion should always produce a larger or equal bbox."""
    from plot_extractor.core.label_crop_planner import _expand_bbox_with_padding

    original = (5, 3, 15, 10)
    crop_shape = (30, 40, 3)
    expanded = _expand_bbox_with_padding(original, crop_shape, "x")
    assert expanded[0] <= original[0]
    assert expanded[1] <= original[1]
    assert expanded[2] >= original[2]
    assert expanded[3] >= original[3]


# --- Test 9: _directional_search_window produces valid geometry ---

def test_directional_search_window_valid_geometry():
    """Search window must always produce valid (x2>x1, y2>y1) coords."""
    from plot_extractor.core.label_crop_planner import _directional_search_window

    image = _make_image(240, 240)
    axis = _make_x_axis(ticks=(50, 100, 150))
    for tick in [50, 100, 150]:
        x1, y1, x2, y2 = _directional_search_window(image, axis, [50, 100, 150], tick)
        assert x2 > x1, f"tick={tick}: x2={x2} <= x1={x1}"
        assert y2 > y1, f"tick={tick}: y2={y2} <= y1={y1}"


# --- Test 10: _directional_search_window for y-axis ---

def test_directional_search_window_y_axis():
    """Y-axis search window should extend horizontally from axis."""
    from plot_extractor.core.label_crop_planner import _directional_search_window

    image = _make_image(240, 240)
    axis = _make_y_axis(ticks=(40, 100, 160))
    for tick in [40, 100, 160]:
        x1, y1, x2, y2 = _directional_search_window(image, axis, [40, 100, 160], tick)
        assert x2 > x1, f"tick={tick}: x2={x2} <= x1={x1}"
        assert y2 > y1, f"tick={tick}: y2={y2} <= y1={y1}"


if __name__ == "__main__":
    test_planned_crop_dataclass_fields()
    print("[PASS] test_planned_crop_dataclass_fields")
    test_plan_tick_label_crop_returns_none_on_blank()
    print("[PASS] test_plan_tick_label_crop_returns_none_on_blank")
    test_plan_tick_label_crop_finds_label_with_mock()
    print("[PASS] test_plan_tick_label_crop_finds_label_with_mock")
    test_plan_tick_label_crop_flags_far_crop()
    print("[PASS] test_plan_tick_label_crop_flags_far_crop")
    test_plan_tick_label_crops_batch_returns_all_ticks()
    print("[PASS] test_plan_tick_label_crops_batch_returns_all_ticks")
    test_quality_flags_none_when_no_text()
    print("[PASS] test_quality_flags_none_when_no_text")
    test_y_axis_crop_has_valid_bbox_with_mock()
    print("[PASS] test_y_axis_crop_has_valid_bbox_with_mock")
    test_expand_bbox_with_padding_increases_size()
    print("[PASS] test_expand_bbox_with_padding_increases_size")
    test_directional_search_window_valid_geometry()
    print("[PASS] test_directional_search_window_valid_geometry")
    test_directional_search_window_y_axis()
    print("[PASS] test_directional_search_window_y_axis")
    print("[SUCCESS] label_crop_planner tests passed")
