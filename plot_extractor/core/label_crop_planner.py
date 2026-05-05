"""Unified tick-label crop planner (inspired by Docling's prepare_element pattern).

Consolidates the three previously scattered crop stages into a single
strategy layer:

  1. _directional_tick_search_window  →  search window
  2. _tesseract_text_bbox             →  text geometry probe
  3. _crop_tick_label_from_tesseract_bbox  →  final crop with expansion

Every crop now carries structured metadata (PlannedCrop) so that
FormulaOCR diagnostics can trace "was the window wrong, bbox wrong,
or recognition wrong?".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np

from plot_extractor.core.axis_detector import Axis


@dataclass
class PlannedCrop:
    """Structured result from planning a single tick-label crop."""

    tick_pixel: int
    search_bbox: tuple[int, int, int, int]
    text_bbox_local: Optional[tuple[int, int, int, int]]
    final_bbox: Optional[tuple[int, int, int, int]]
    crop: Optional[np.ndarray]
    tesseract_probe_text: Optional[str]
    tesseract_probe_value: Optional[float]
    quality_flags: dict[str, bool] = field(default_factory=dict)


@dataclass
class StageTiming:
    """Per-stage timing telemetry for the extraction pipeline.

    Separates accuracy-related metrics from throughput metrics, following
    Docling's threaded-stage timing pattern.
    """

    crop_planning_ms: float = 0.0
    formula_infer_ms: float = 0.0
    calibrate_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "crop_planning_ms": round(self.crop_planning_ms, 2),
            "formula_infer_ms": round(self.formula_infer_ms, 2),
            "calibrate_ms": round(self.calibrate_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


def _local_tick_gap(tick_pixels: List[int], tick: int) -> int:
    """Median gap between *tick* and its immediate neighbours."""
    if len(tick_pixels) < 2:
        return 30
    diffs = [abs(tick_pixels[i + 1] - tick_pixels[i])
             for i in range(len(tick_pixels) - 1)]
    return max(10, int(sorted(diffs)[len(diffs) // 2]))


def _directional_search_window(
    image: np.ndarray,
    axis: Axis,
    tick_pixels: List[int],
    tick_pixel: int,
) -> tuple[int, int, int, int]:
    """Build a tick-anchored search window in the label direction."""
    h, w = image.shape[:2]
    tick = int(tick_pixel)
    axis_pos = int(axis.position)
    gap = _local_tick_gap(tick_pixels, tick)

    if axis.direction == "x":
        half_width = int(min(max(gap * 0.48, 18), 70))
        vertical_extent = int(min(max(gap * 0.75, 38), 76))
        x1 = max(0, tick - half_width)
        x2 = min(w, tick + half_width)
        if axis.side == "top":
            y1 = max(0, axis_pos - vertical_extent)
            y2 = max(0, axis_pos - 2)
        else:
            y1 = min(h, axis_pos + 2)
            y2 = min(h, axis_pos + vertical_extent)
    else:
        half_height = int(min(max(gap * 0.45, 12), 36))
        horizontal_extent = int(min(max(gap * 0.95, 60), 130))
        y1 = max(0, tick - half_height)
        y2 = min(h, tick + half_height)
        if axis.side == "right":
            x1 = min(w, axis_pos + 2)
            x2 = min(w, axis_pos + horizontal_extent)
        else:
            x1 = max(0, axis_pos - horizontal_extent)
            x2 = max(0, axis_pos - 2)
            if x2 <= x1:
                x1 = min(w, axis_pos + 2)
                x2 = min(w, axis_pos + horizontal_extent)

    if x2 <= x1:
        x2 = min(w, x1 + 10)
    if y2 <= y1:
        y2 = min(h, y1 + 10)
    return int(x1), int(y1), int(x2), int(y2)


def _is_empty_or_uniform_crop(crop: np.ndarray) -> bool:
    """Cheap pre-filter: skip tesseract for blank/uniform search windows.

    Tesseract subprocess setup costs ~80-120 ms per invocation; eliminating
    obviously empty crops removes a substantial share of probe calls on
    the v3 log_x profile (16 probe calls/image, ~100 ms each).
    """
    if crop.size == 0:
        return True
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    if float(np.std(gray)) < 3.5:
        return True
    if int(np.count_nonzero(gray < 128)) < 8:
        return True
    return False


def _tesseract_geometry_probe_impl(
    crop: np.ndarray,
) -> tuple[Optional[tuple[int, int, int, int]], Optional[str]]:
    """Run pytesseract geometry probe on a non-empty crop."""
    try:
        import pytesseract  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None, None

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop.copy()
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(
            binary,
            output_type=pytesseract.Output.DICT,
            config='--psm 7 -c tessedit_char_whitelist=0123456789eE^xX×-+.(){}',
        )
    except Exception:  # pylint: disable=broad-except
        return None, None

    boxes = []
    texts = []
    n = len(data.get("text", []))
    for idx in range(n):
        raw_text = (data["text"][idx] or "").strip()
        if not raw_text:
            continue
        if not any(ch.isdigit() for ch in raw_text):
            continue
        try:
            conf = float(data.get("conf", ["-1"] * n)[idx])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < -0.5:
            continue
        x = int(data["left"][idx] / 2.0)
        y = int(data["top"][idx] / 2.0)
        bw = int(data["width"][idx] / 2.0)
        bh = int(data["height"][idx] / 2.0)
        if bw < 1 or bh < 1:
            continue
        boxes.append((x, y, x + bw, y + bh))
        texts.append(raw_text)

    if not boxes:
        return None, None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2), " ".join(texts)


@lru_cache(maxsize=2048)
def _tesseract_geometry_probe_cached(
    crop_bytes: bytes,
    shape: tuple,
    dtype_str: str,
) -> tuple[Optional[tuple[int, int, int, int]], Optional[str]]:
    """Bytes-keyed cache around the tesseract geometry probe."""
    arr = np.frombuffer(crop_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
    return _tesseract_geometry_probe_impl(arr.copy())


def _tesseract_geometry_probe(
    crop: np.ndarray,
) -> tuple[Optional[tuple[int, int, int, int]], Optional[str]]:
    """Use Tesseract for geometry only: union word boxes in one label window.

    Layered fast paths:

    1. Reject crops that are empty or uniform (no measurable text contrast)
       without paying the subprocess cost.
    2. Bytes-keyed ``lru_cache`` deduplicates exact repeat crops across
       the validation batch.
    """
    if crop is None or crop.size == 0:
        return None, None
    if _is_empty_or_uniform_crop(crop):
        return None, None
    return _tesseract_geometry_probe_cached(
        crop.tobytes(),
        tuple(crop.shape),
        str(crop.dtype),
    )


def _expand_bbox_with_padding(
    bbox_local: tuple[int, int, int, int],
    search_crop_shape: tuple[int, ...],
    axis_direction: str,
    expansion_factor: float = 0.30,
) -> tuple[int, int, int, int]:
    """Expand text bbox with asymmetric padding (Docling's expansion_factor pattern)."""
    lx1, ly1, lx2, ly2 = bbox_local
    ch, cw = search_crop_shape[:2]

    if axis_direction == "x":
        pad_x = max(6, int((lx2 - lx1) * expansion_factor))
        pad_y = max(6, int((ly2 - ly1) * 0.45))
    else:
        pad_x = max(5, int((lx2 - lx1) * 0.22))
        pad_y = max(4, int((ly2 - ly1) * expansion_factor))

    return (
        max(0, lx1 - pad_x),
        max(0, ly1 - pad_y),
        min(cw, lx2 + pad_x + 2),
        min(ch, ly2 + pad_y),
    )


def plan_tick_label_crop(
    image: np.ndarray,
    axis: Axis,
    tick_pixels: List[int],
    tick_pixel: int,
    policy=None,
    force_geometry_fallback: bool = False,
) -> Optional[PlannedCrop]:
    """Plan a single tick-label crop with full diagnostic metadata.

    Returns PlannedCrop with all intermediate results, or None when
    no text is found in the search window.
    """
    search_bbox = _directional_search_window(image, axis, tick_pixels, tick_pixel)
    sx1, sy1, sx2, sy2 = search_bbox
    search_crop = image[sy1:sy2, sx1:sx2]

    if search_crop.size == 0:
        return None

    bbox_local, probe_text = _tesseract_geometry_probe(search_crop)

    if bbox_local is None:
        if not force_geometry_fallback:
            return None
        # Drop ticks whose search window has no text signal at all. Phantom
        # anchors on uniform windows feed noise into FormulaOCR / calibration
        # without rescuing any real label, so they're a net loss for both
        # throughput (wasted second-pass OCR) and accuracy (RANSAC outliers).
        if _is_empty_or_uniform_crop(search_crop):
            return None
        bbox_local = (0, 0, search_crop.shape[1], search_crop.shape[0])

    expanded_local = _expand_bbox_with_padding(
        bbox_local, search_crop.shape, axis.direction,
    )
    elx1, ely1, elx2, ely2 = expanded_local

    final_x1 = max(0, sx1 + elx1)
    final_y1 = max(0, sy1 + ely1)
    final_x2 = min(image.shape[1], sx1 + elx2)
    final_y2 = min(image.shape[0], sy1 + ely2)
    final_bbox = (final_x1, final_y1, final_x2, final_y2)

    if final_x2 <= final_x1 or final_y2 <= final_y1:
        return None

    crop = image[final_y1:final_y2, final_x1:final_x2]

    from plot_extractor.utils.math_utils import parse_numeric  # pylint: disable=import-outside-toplevel
    probe_value = None
    if probe_text:
        probe_value = parse_numeric(probe_text)

    center = int((final_x1 + final_x2) / 2) if axis.direction == "x" else int((final_y1 + final_y2) / 2)
    gap = _local_tick_gap(tick_pixels, tick_pixel)
    is_far = abs(center - int(tick_pixel)) > max(12, gap * 0.55)

    quality_flags = {
        "is_empty": False,
        "is_far_from_tick": is_far,
        "possible_minor_tick": False,
        "geometry_fallback": probe_text is None,
    }

    return PlannedCrop(
        tick_pixel=int(tick_pixel),
        search_bbox=search_bbox,
        text_bbox_local=bbox_local,
        final_bbox=final_bbox,
        crop=crop,
        tesseract_probe_text=probe_text,
        tesseract_probe_value=probe_value,
        quality_flags=quality_flags,
    )


def is_processable_for_formula(
    axis_is_log: bool,
    tesseract_count: int,
    label_count: int,
    anchors: list,
) -> bool:
    """Single-point gate: should FormulaOCR be triggered for this axis?

    Inspired by Docling's ``is_processable`` pattern (granite_vision.py).

    Trigger conditions (any one is sufficient):
      - Axis already detected as log.
      - Tesseract labels are sparse (< 3).
      - Tesseract values suggest superscript misread (100-110 or 10-19).
      - Anchor text contains exponential notation characters.

    Suppress conditions (override triggers):
      - All anchors are blank (minor ticks only, no text geometry).
      - No labels at all and axis is not log.
    """
    # Suppress: no anchors at all
    if not anchors:
        return False

    # Suppress: all anchors are empty (minor ticks)
    has_any_text = False
    has_any_value = False
    for anchor in anchors:
        text = getattr(anchor, "tesseract_text", None) or ""
        value = getattr(anchor, "tesseract_value", None)
        crop_size = getattr(anchor, "crop", None)
        if crop_size is not None:
            try:
                if crop_size.size > 0:
                    has_any_text = has_any_text or bool(text.strip())
            except AttributeError:
                pass
        if text and any(ch.isdigit() for ch in text):
            has_any_text = True
        if value is not None:
            has_any_value = True

    # Suppress only when there are truly no labels at all
    if not has_any_text and not has_any_value and label_count == 0 and not axis_is_log:
        return False

    # Trigger: log axis
    if axis_is_log:
        return True

    # Trigger: sparse tesseract labels
    if tesseract_count < 3:
        return True

    # Trigger: no tesseract values but some labels exist
    if tesseract_count == 0 and label_count > 0:
        return True

    # Trigger: superscript misread patterns (100-110 or 10-19)
    for anchor in anchors:
        value = getattr(anchor, "tesseract_value", None)
        if value is not None and (100 <= value <= 110 or 10 <= value <= 19):
            return True

    # Trigger: text with exponential notation characters
    for anchor in anchors:
        text = getattr(anchor, "tesseract_text", "") or ""
        if any(ch in text for ch in ("^", "e", "E", "×", "x", "X", "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")):
            return True

    return False


# ---------------------------------------------------------------------------
# Candidate map priority table (Docling-inspired fixed priority)
# ---------------------------------------------------------------------------

CANDIDATE_PRIORITY: dict[str, int] = {
    "formula_generated": 0,
    "formula": 1,
    "fused": 2,
    "tesseract": 3,
    "heuristic": 4,
}


def build_candidate_maps(
    formula_log_score: float,
    formula_value_count: int,
    formula_anchor_count: int,
    tesseract_count: int,
    label_count: int,
    anchors: list,
    formula_labeled: list,
    fused_labeled: list,
    tesseract_labeled: list,
    axis=None,
    tick_pixels: list | None = None,
) -> list[tuple[str, list]]:
    """Build candidate maps using a fixed priority table.

    Replaces the scattered conditional branching in calibrate_all_axes
    with a single, ordered evaluation of each candidate source.

    Priority order (highest first):
      1. formula_generated — formula can extrapolate a complete log axis
      2. formula — direct formula anchor values
      3. fused — tesseract corrected by formula where available
      4. tesseract — raw tesseract reads
      5. heuristic — synthetic tick generation (always last resort)
    """
    candidates: list[tuple[str, int, list]] = []

    # 1. Formula-generated: full log axis extrapolation
    formula_generated_built = False
    if formula_log_score >= 0.3 and formula_value_count >= 1 and axis is not None:
        from plot_extractor.core.axis_calibrator import _build_formula_generated_log_ticks  # pylint: disable=import-outside-toplevel
        formula_generated = _build_formula_generated_log_ticks(axis, anchors)
        if formula_generated:
            candidates.append(("formula_generated", CANDIDATE_PRIORITY["formula_generated"], formula_generated))
            formula_generated_built = True

    # When formula_generated was not built, evaluate remaining candidates
    if not formula_generated_built:
        # 2. Formula direct values
        tesseract_sparse = tesseract_count < 2
        has_formula_evidence = formula_anchor_count >= 2
        if formula_labeled and (
            (formula_log_score < 0.3 and tesseract_sparse and has_formula_evidence)
            or (formula_value_count >= 2)
        ):
            candidates.append(("formula", CANDIDATE_PRIORITY["formula"], formula_labeled))

    if formula_log_score >= 0.3 and formula_value_count >= 2 and candidates:
        # Multiple FormulaOCR exponent reads are authoritative for log axes.
        # Do not let noisier tesseract/fused maps win only because they have
        # more labels; missing tick values can be filled by calibration fallback.
        candidates.sort(key=lambda item: item[1])
        return [(name, labeled) for name, _priority, labeled in candidates]

    # 3. Fused: tesseract corrected by formula
    if formula_value_count >= 1 and tesseract_count > 0 and fused_labeled:
        candidates.append(("fused", CANDIDATE_PRIORITY["fused"], fused_labeled))

    # 4. Tesseract raw
    if tesseract_count > 0 and tesseract_labeled:
        candidates.append(("tesseract", CANDIDATE_PRIORITY["tesseract"], tesseract_labeled))

    # 5. Heuristic: always available as fallback
    if not candidates:
        heuristic_ticks = tick_pixels or []
        candidates.append(("heuristic", CANDIDATE_PRIORITY["heuristic"], [(p, None) for p in heuristic_ticks]))

    # Sort by priority and return (name, labeled) pairs
    candidates.sort(key=lambda item: item[1])
    return [(name, labeled) for name, _, labeled in candidates]


def plan_tick_label_crops_batch(
    image: np.ndarray,
    axis: Axis,
    tick_pixels: List[int],
    policy=None,
    force_geometry_fallback: bool = False,
) -> list[Optional[PlannedCrop]]:
    """Plan crops for all ticks on one axis."""
    return [
        plan_tick_label_crop(
            image,
            axis,
            tick_pixels,
            tp,
            policy=policy,
            force_geometry_fallback=force_geometry_fallback,
        )
        for tp in tick_pixels
    ]
