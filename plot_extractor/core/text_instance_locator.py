"""Locate text-like label instances inside axis label bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from plot_extractor.core.axis_detector import Axis


@dataclass
class TextInstance:
    """A localized text instance with a crop suitable for OCR."""

    bbox: tuple[int, int, int, int]
    center: int
    crop: np.ndarray
    components: list[tuple[int, int, int, int]]
    confidence: float = 0.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _axis_label_band(image: np.ndarray, axis: Axis) -> tuple[int, int, int, int]:
    """Return a search band around the likely label region for one axis."""
    h, w = image.shape[:2]
    if axis.direction == "x":
        x1 = max(0, int(axis.plot_start) - 8)
        x2 = min(w, int(axis.plot_end) + 8)
        y1 = max(0, int(axis.position) + 2)
        y2 = min(h, int(axis.position) + 78)
    else:
        y1 = max(0, int(axis.plot_start) - 8)
        y2 = min(h, int(axis.plot_end) + 8)
        if axis.side == "right":
            x1 = max(0, int(axis.position) + 2)
            x2 = min(w, int(axis.position) + 112)
        else:
            x1 = max(0, int(axis.position) - 112)
            x2 = max(0, int(axis.position) - 2)
    if x2 <= x1:
        x2 = min(w, x1 + 10)
    if y2 <= y1:
        y2 = min(h, y1 + 10)
    return x1, y1, x2, y2


def _build_text_mask(region: np.ndarray) -> np.ndarray:
    """Build a compact foreground mask for text-like components."""
    if region.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = _to_gray(region)
    if gray.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive threshold helps when the label is slightly faded or anti-aliased.
    bh, bw = gray.shape[:2]
    block_size = min(31, max(11, int(min(bh, bw) * 0.25) | 1))
    try:
        adapt = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            8,
        )
        binary = cv2.bitwise_or(otsu, adapt)
    except cv2.error:
        binary = otsu

    # Small closing links glyph fragments without merging neighboring labels.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((3, 2), dtype=np.uint8),
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )
    return binary


def _component_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return connected-component boxes from a text mask."""
    if binary.size == 0:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    h, w = binary.shape[:2]
    band_area = max(1, h * w)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < 10:
            continue
        if area > band_area * 0.35:
            continue
        if bw <= 0 or bh <= 0:
            continue

        # Ignore long axis spines and large background fragments.
        if (bw > 32 and bh <= 4) or (bh > 32 and bw <= 4):
            continue
        aspect = bw / max(bh, 1)
        if aspect > 16.0 and area > 80:
            continue

        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    return boxes


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_gap = max(0, max(bx1 - ax2, ax1 - bx2))
    y_gap = max(0, max(by1 - ay2, ay1 - by2))
    return x_gap, y_gap


def _should_merge_boxes(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Merge character boxes that likely belong to one label instance."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw = ax2 - ax1
    ah = ay2 - ay1
    bw = bx2 - bx1
    bh = by2 - by1
    x_gap, y_gap = _bbox_gap(a, b)

    # Base rule: neighboring glyphs on the same label line should be close
    # in both axes.  This keeps adjacent tick labels separate but joins
    # superscripts and tightly spaced numeric groups.
    close_h = x_gap <= max(4, int(min(aw, bw) * 0.9))
    close_v = y_gap <= max(4, int(max(ah, bh) * 0.75))
    if close_h and close_v:
        return True

    # Superscript-style clusters sit slightly above/right of the base glyph.
    is_superscript = (
        x_gap <= max(5, int(max(aw, bw) * 0.6))
        and y_gap <= max(6, int(max(ah, bh) * 1.1))
        and (by2 <= ay2 or ay1 <= by2)
    )
    if is_superscript:
        return True

    return False


def _cluster_boxes(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    if len(boxes) <= 1:
        return [boxes] if boxes else []

    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(boxes)):
        a = boxes[i]
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if _should_merge_boxes(a, b):
                union(i, j)

    clusters: dict[int, list[tuple[int, int, int, int]]] = {}
    for i, box in enumerate(boxes):
        root = find(i)
        clusters.setdefault(root, []).append(box)
    return list(clusters.values())


def detect_axis_label_instances(image: np.ndarray, axis: Axis) -> List[TextInstance]:
    """Return candidate text instances inside the axis label band."""
    x1, y1, x2, y2 = _axis_label_band(image, axis)
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return []

    mask = _build_text_mask(region)
    if mask.size == 0:
        return []

    boxes = _component_boxes(mask)
    if not boxes:
        return []

    clusters = _cluster_boxes(boxes)
    instances: list[TextInstance] = []
    region_h, region_w = region.shape[:2]

    for cluster in clusters:
        xs = [bx1 for bx1, _, _, _ in cluster]
        ys = [by1 for _, by1, _, _ in cluster]
        xe = [bx2 for _, _, bx2, _ in cluster]
        ye = [by2 for _, _, _, by2 in cluster]
        bx1 = max(0, min(xs))
        by1 = max(0, min(ys))
        bx2 = min(region_w, max(xe))
        by2 = min(region_h, max(ye))

        width = bx2 - bx1
        height = by2 - by1
        # Expand the localized box so OCR sees the complete token, including
        # nearby whitespace and superscript fragments.
        pad_x = max(4, int(width * 0.35))
        pad_y = max(4, int(height * 0.35))
        if axis.direction == "x":
            pad_y = max(pad_y, 5)
            pad_x = max(pad_x, 3)
        else:
            pad_x = max(pad_x, 5)
            pad_y = max(pad_y, 3)

        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(region_w, bx2 + pad_x)
        cy2 = min(region_h, by2 + pad_y)
        crop = region[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        full_bbox = (x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2)
        if axis.direction == "x":
            center = int((full_bbox[0] + full_bbox[2]) / 2)
        else:
            center = int((full_bbox[1] + full_bbox[3]) / 2)
        area = max(1, (full_bbox[2] - full_bbox[0]) * (full_bbox[3] - full_bbox[1]))
        confidence = min(1.0, 0.25 + 0.12 * len(cluster) + min(0.35, area / 4000.0))

        instances.append(
            TextInstance(
                bbox=full_bbox,
                center=center,
                crop=crop,
                components=[(x1 + bx1, y1 + by1, x1 + bx2, y1 + by2) for bx1, by1, bx2, by2 in cluster],
                confidence=float(confidence),
            )
        )

    if axis.direction == "x":
        instances.sort(key=lambda item: item.center)
    else:
        instances.sort(key=lambda item: item.center)
    return instances
