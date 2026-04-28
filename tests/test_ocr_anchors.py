"""Unit tests for tick-label anchor detection."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.axis_detector import Axis
from plot_extractor.core.ocr_reader import detect_tick_label_anchors, read_all_tick_labels


def _make_blank_image(height=240, width=240):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_detect_tick_label_anchors_x_axis():
    """X-axis label blobs should match the nearest x tick positions."""
    image = _make_blank_image()
    axis = Axis(
        direction="x",
        side="bottom",
        position=120,
        plot_start=20,
        plot_end=220,
        ticks=[(40, None), (100, None), (170, None)],
    )

    for center_x in (40, 100, 170):
        cv2.rectangle(image, (center_x - 7, 132), (center_x + 7, 148), (0, 0, 0), -1)

    anchors = detect_tick_label_anchors(image, axis, [40, 100, 170])

    assert [a.tick_pixel for a in anchors] == [40, 100, 170]
    assert all(a.label_bbox != (0, 0, 0, 0) for a in anchors)
    assert all(abs(a.label_center - tick) <= 12 for a, tick in zip(anchors, [40, 100, 170]))

    legacy = read_all_tick_labels(image, axis, [40, 100, 170])
    assert [pix for pix, _ in legacy] == [40, 100, 170]


def test_detect_tick_label_anchors_y_axis():
    """Y-axis label blobs should match the nearest y tick positions."""
    image = _make_blank_image()
    axis = Axis(
        direction="y",
        side="left",
        position=70,
        plot_start=20,
        plot_end=220,
        ticks=[(40, None), (110, None), (180, None)],
    )

    for center_y in (40, 110, 180):
        cv2.rectangle(image, (18, center_y - 7), (36, center_y + 7), (0, 0, 0), -1)

    anchors = detect_tick_label_anchors(image, axis, [40, 110, 180])

    assert [a.tick_pixel for a in anchors] == [40, 110, 180]
    assert all(a.label_bbox != (0, 0, 0, 0) for a in anchors)
    assert all(abs(a.label_center - tick) <= 12 for a, tick in zip(anchors, [40, 110, 180]))

    legacy = read_all_tick_labels(image, axis, [40, 110, 180])
    assert [pix for pix, _ in legacy] == [40, 110, 180]


if __name__ == "__main__":
    test_detect_tick_label_anchors_x_axis()
    test_detect_tick_label_anchors_y_axis()
    print("[SUCCESS] OCR anchor tests passed")
