"""Unit tests for axis text-instance localization."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.axis_detector import Axis
from plot_extractor.core.text_instance_locator import detect_axis_label_instances


def _make_blank_image(height=240, width=240):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_detect_axis_label_instances_merges_superscript_like_glyphs():
    """A superscript fragment should stay attached to its base label."""
    image = _make_blank_image()
    axis = Axis(
        direction="x",
        side="bottom",
        position=120,
        plot_start=20,
        plot_end=220,
        ticks=[(60, None), (120, None), (180, None)],
    )

    # Base digits for one log-style label.
    cv2.rectangle(image, (52, 134), (60, 150), (0, 0, 0), -1)
    cv2.rectangle(image, (62, 134), (70, 150), (0, 0, 0), -1)
    # Superscript-like fragment, slightly above/right of the base.
    cv2.rectangle(image, (68, 126), (74, 132), (0, 0, 0), -1)

    # A second label should remain separate.
    cv2.rectangle(image, (118, 134), (126, 150), (0, 0, 0), -1)
    cv2.rectangle(image, (128, 134), (136, 150), (0, 0, 0), -1)

    instances = detect_axis_label_instances(image, axis)
    assert len(instances) >= 2

    merged = min(instances, key=lambda item: abs(item.center - 62))
    assert merged.bbox[0] <= 52
    assert merged.bbox[1] <= 126
    assert merged.bbox[2] >= 74
    assert merged.bbox[3] >= 150
    assert len(merged.components) >= 1

    other = min(instances, key=lambda item: abs(item.center - 128))
    assert other.bbox[0] >= 112
    assert other.bbox[2] <= 146


if __name__ == "__main__":
    test_detect_axis_label_instances_merges_superscript_like_glyphs()
    print("[SUCCESS] text instance locator tests passed")
