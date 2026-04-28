"""OCR and tick label reading with fallback mechanisms."""
import glob
from functools import lru_cache
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except ImportError:
    _HAS_PYTESSERACT = False


def _find_tesseract_binary() -> str | None:
    """Locate tesseract executable without hard-coding paths.

    1. Check PATH via shutil.which (cross-platform).
    2. Search common installation directories on Windows/macOS/Linux.
    3. If found outside PATH, return the absolute path so the caller
       can set pytesseract.pytesseract.tesseract_cmd.
    4. If not found anywhere, return None.
    """
    # 1. PATH lookup
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd

    # 2. Common install locations (platform-aware, no hard-coded defaults)
    candidates = []
    if sys.platform == "win32":
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        for pf in program_files:
            if pf:
                candidates.extend(glob.glob(os.path.join(pf, "Tesseract-OCR", "tesseract.exe")))
    elif sys.platform == "darwin":
        candidates.extend(glob.glob("/usr/local/Cellar/tesseract/*/bin/tesseract"))
        candidates.extend(glob.glob("/opt/homebrew/Cellar/tesseract/*/bin/tesseract"))
    else:
        candidates.extend(glob.glob("/usr/bin/tesseract"))
        candidates.extend(glob.glob("/usr/local/bin/tesseract"))

    for c in candidates:
        if Path(c).exists():
            return c
    return None


_TESSERACT_CMD: str | None = None
TESSERACT_AVAILABLE = False


def init_tesseract() -> str | None:
    """Initialize tesseract path. Call once before OCR operations.

    Returns the resolved binary path, or None if unavailable.
    When None, callers should print an installation prompt.
    """
    global TESSERACT_AVAILABLE, _TESSERACT_CMD
    if not _HAS_PYTESSERACT:
        return None
    if _TESSERACT_CMD is not None:
        return _TESSERACT_CMD

    _TESSERACT_CMD = _find_tesseract_binary()
    if _TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
        TESSERACT_AVAILABLE = True
    return _TESSERACT_CMD

from plot_extractor.utils.math_utils import parse_numeric


@dataclass
class AxisLabelAnchor:
    """Axis tick anchor plus the label crop used to read it."""

    tick_pixel: int
    label_bbox: tuple[int, int, int, int]
    label_center: int
    crop: np.ndarray
    tesseract_text: Optional[str] = None
    tesseract_value: Optional[float] = None
    formula_latex: Optional[str] = None
    formula_value: Optional[float] = None
    confidence: float = 0.0
    source: str = "missing"


def _clean_ocr_text(text: str) -> str:
    """Clean OCR output to extract likely numeric text."""
    text = text.strip()
    # Remove common noise
    text = text.replace("|", "").replace("—", "-").replace("–", "-")
    return text


def _ocr_tick_label_text_impl(
    image_crop,
    block_size: int | None = None,
    c_val: int | None = None,
) -> tuple[Optional[str], Optional[float], int]:
    """Read raw OCR text and numeric value from a tick-label crop."""
    init_tesseract()
    if not TESSERACT_AVAILABLE:
        return None, None, 0
    try:
        preprocessed, contours_len = _preprocess_tick_crop(
            image_crop,
            policy=None if block_size is None or c_val is None else type("OCRPolicy", (), {"ocr_block_size": block_size, "ocr_C": c_val})(),
        )
        from PIL import Image

        img = Image.fromarray(preprocessed)

        # Try multiple PSM modes; single-character/number modes work best for ticks
        configs = [
            "--psm 8 -c tessedit_char_whitelist=0123456789.,-eE+kKmMbB%",
            "--psm 7 -c tessedit_char_whitelist=0123456789.,-eE+kKmMbB%",
        ]
        fallback_text = None
        for cfg in configs:
            text = pytesseract.image_to_string(img, config=cfg)
            text = _clean_ocr_text(text)
            if text and fallback_text is None:
                fallback_text = text
            val = parse_numeric(text)
            # Keep negative labels (e.g. axes spanning below zero).
            if val is not None:
                return text, val, contours_len

        # Scatteract negative-sign heuristic:
        # If OTSU shows 2 top-level contours (minus + digit) but OCR only
        # returned a single digit, prepend '-' and retry.
        if contours_len == 2:
            for cfg in configs:
                text = pytesseract.image_to_string(img, config=cfg)
                text = _clean_ocr_text(text)
                if len(text) == 1 and text.isdigit():
                    val = parse_numeric("-" + text)
                    if val is not None:
                        return "-" + text, val, contours_len

        return fallback_text, None, contours_len
    except OSError:
        return None, None, 0


def _ocr_tick_label_text(image_crop, policy=None) -> tuple[Optional[str], Optional[float], int]:
    """Read raw OCR text and numeric value from a tick-label crop."""
    block_size = None
    c_val = None
    if policy is not None:
        block_size = int(policy.ocr_block_size) | 1
        c_val = int(policy.ocr_C)
    return _ocr_tick_label_text_impl(image_crop, block_size=block_size, c_val=c_val)


@lru_cache(maxsize=4096)
def _ocr_tick_label_text_cached(
    crop_bytes: bytes,
    shape: tuple[int, ...],
    dtype_str: str,
    block_size: int | None,
    c_val: int | None,
) -> tuple[Optional[str], Optional[float], int]:
    crop = np.frombuffer(crop_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
    return _ocr_tick_label_text_impl(crop, block_size=block_size, c_val=c_val)


def _ocr_tick_label_text_cached_for_crop(image_crop, policy=None) -> tuple[Optional[str], Optional[float], int]:
    block_size = None
    c_val = None
    if policy is not None:
        block_size = int(policy.ocr_block_size) | 1
        c_val = int(policy.ocr_C)
    crop = np.ascontiguousarray(image_crop)
    return _ocr_tick_label_text_cached(
        crop.tobytes(),
        crop.shape,
        crop.dtype.str,
        block_size,
        c_val,
    )


def _deskew_crop(crop, max_angle=10.0):
    """Deskew a text crop and count independent contours.

    Based on Scatteract's tesseract.py compute_skew / deskew.
    Returns (rotated_gray, contours_len) where contours_len is the number
    of top-level contours in an OTSU-thresholded version of the image.
    This count is used by the negative-sign heuristic.
    """
    gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # OTSU threshold for contour counting (Scatteract-style)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None and len(hierarchy) > 0:
        contours_len = len([j for j in hierarchy[0] if j[-1] == -1])
    else:
        contours_len = len(contours)

    # Compute skew angle from dark pixels
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return gray, contours_len

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > max_angle:
        return gray, contours_len

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return rotated, contours_len


def _preprocess_tick_crop(crop: np.ndarray, policy=None) -> tuple[np.ndarray, int]:
    """Enhance a tick-label crop for OCR.

    Operates only on the crop; never mutates the global preprocessed image.
    Returns (preprocessed_image, contours_len) where contours_len is used
    for the negative-sign heuristic.
    """
    if crop.size == 0:
        return crop, 0

    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop.copy()

    # Deskew for rotated labels + count contours (Scatteract-style)
    gray, contours_len = _deskew_crop(gray)

    # Upscale small crops for better OCR accuracy
    h, w = gray.shape[:2]
    if h < 30 or w < 60:
        scale = 3
    elif h < 50 or w < 100:
        scale = 2
    else:
        scale = 1
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Light denoising on crop only
    gray = cv2.medianBlur(gray, 3)

    # Adaptive threshold for local contrast enhancement
    bh, bw = gray.shape[:2]
    if policy is not None:
        block_size = policy.ocr_block_size | 1  # ensure odd
        c_val = policy.ocr_C
    else:
        block_size = min(25, max(11, int(min(bh, bw) * 0.3) | 1))  # odd, clamped
        c_val = 10
    try:
        preprocessed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )
    except cv2.error:
        # Fallback to original grayscale if adaptive threshold fails
        preprocessed = gray

    return preprocessed, contours_len


def read_tick_label(image_crop, policy=None) -> Optional[float]:
    """Attempt OCR on a small crop around a tick label."""
    _, value, _ = _ocr_tick_label_text_cached_for_crop(image_crop, policy=policy)
    return value


def _detect_label_blobs(image: np.ndarray, axis) -> List[Tuple[int, int, int, int]]:
    """Find text-like connected components in the axis label region.

    Returns list of (x, y, w, h) bounding boxes for candidate label blobs.
    """
    h, w = image.shape[:2]
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Define label search region based on axis direction/side
    if axis.direction == "x":
        # Labels below axis
        y1 = min(h, int(axis.position) + 3)
        y2 = min(h, int(axis.position) + 55)
        x1 = max(0, int(axis.plot_start) - 5)
        x2 = min(w, int(axis.plot_end) + 5)
    else:
        if axis.side == "right":
            x1 = min(w, int(axis.position) + 3)
            x2 = min(w, int(axis.position) + 90)
        else:
            x1 = max(0, int(axis.position) - 90)
            x2 = max(0, int(axis.position) - 3)
        y1 = max(0, int(axis.plot_start) - 5)
        y2 = min(h, int(axis.plot_end) + 5)

    if x2 <= x1 or y2 <= y1:
        return []

    region = gray[y1:y2, x1:x2]
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove tiny noise with small opening
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    blobs = []
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        if area < 15:
            continue
        if area > 5000:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.1 or aspect > 8.0:
            continue
        # Map back to image coordinates
        blobs.append((x1 + bx, y1 + by, bw, bh))

    return blobs


@dataclass
class _LabelCandidate:
    bbox: tuple[int, int, int, int]
    center: int
    crop: np.ndarray
    tesseract_text: Optional[str]
    tesseract_value: Optional[float]
    confidence: float


def _match_ticks_labels_bidirectional(
    tick_pixels: List[int],
    labels: List[Tuple[int, float]],
) -> List[Tuple[int, Optional[float]]]:
    """Scatteract-style bidirectional nearest-neighbor matching.

    Only keeps (tick, label) pairs that are mutual nearest neighbors.
    This rejects spurious detections on both sides.
    """
    if not labels or not tick_pixels:
        return [(p, None) for p in tick_pixels]

    tick_pixels = sorted(tick_pixels)
    label_positions = [pos for pos, _ in labels]

    # Forward: nearest tick for each label
    label_to_tick = {}
    for lpos, lval in labels:
        nearest = min(tick_pixels, key=lambda t, _lpos=lpos: abs(t - _lpos))
        label_to_tick[lpos] = nearest

    # Backward: nearest label for each tick
    tick_to_label = {}
    for t in tick_pixels:
        nearest = min(label_positions, key=lambda l, _t=t: abs(l - _t))
        tick_to_label[t] = nearest

    # Keep only mutual nearest neighbors
    matched = {}
    for lpos, lval in labels:
        nearest_tick = label_to_tick[lpos]
        if tick_to_label.get(nearest_tick) == lpos:
            # If multiple labels map to same tick, prefer the closer one
            if nearest_tick in matched:
                old_lpos = None
                for ll, _ in labels:
                    is_mutual = (
                        label_to_tick.get(ll) == nearest_tick
                        and tick_to_label.get(nearest_tick) == ll
                    )
                    if is_mutual:
                        old_lpos = ll
                        break
                if old_lpos is not None and abs(lpos - nearest_tick) < abs(old_lpos - nearest_tick):
                    matched[nearest_tick] = lval
            else:
                matched[nearest_tick] = lval

    return [(p, matched.get(p)) for p in tick_pixels]


def _match_tick_label_candidates(
    tick_pixels: List[int],
    candidates: List[_LabelCandidate],
) -> List[tuple[int, Optional[_LabelCandidate]]]:
    """Mutual nearest-neighbor match from ticks to label candidates."""
    if not candidates or not tick_pixels:
        return [(p, None) for p in tick_pixels]

    tick_pixels = sorted(tick_pixels)
    candidate_positions = [cand.center for cand in candidates]

    candidate_to_tick = {}
    for idx, cand in enumerate(candidates):
        nearest = min(tick_pixels, key=lambda t, _cand=cand: abs(t - _cand.center))
        candidate_to_tick[idx] = nearest

    tick_to_candidate = {}
    for tick in tick_pixels:
        nearest_idx = min(
            range(len(candidates)),
            key=lambda i, _tick=tick: abs(candidate_positions[i] - _tick),
        )
        tick_to_candidate[tick] = nearest_idx

    matched = {}
    for idx, cand in enumerate(candidates):
        nearest_tick = candidate_to_tick[idx]
        if tick_to_candidate.get(nearest_tick) != idx:
            continue
        existing = matched.get(nearest_tick)
        if existing is None:
            matched[nearest_tick] = cand
            continue
        current_dist = abs(existing.center - nearest_tick)
        new_dist = abs(cand.center - nearest_tick)
        if new_dist < current_dist or (new_dist == current_dist and cand.confidence > existing.confidence):
            matched[nearest_tick] = cand

    return [(tick, matched.get(tick)) for tick in tick_pixels]


def _crop_tick_label_region(
    image: np.ndarray,
    axis,
    tick_pixel: int,
    padding: int = 15,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the expected tick-label region around one tick position."""
    h, w = image.shape[:2]
    if axis.direction == "x":
        y1 = min(h, int(axis.position) + 3)
        y2 = min(h, int(axis.position) + padding * 2 + 5)
        x1 = max(0, int(tick_pixel) - padding)
        x2 = min(w, int(tick_pixel) + padding)
    else:
        prev_pix = int(tick_pixel) - padding * 2
        next_pix = int(tick_pixel) + padding * 2
        half_gap = min(abs(int(tick_pixel) - prev_pix), abs(next_pix - int(tick_pixel))) / 2.0
        v_pad = int(min(max(half_gap - 1, 10), padding * 1.5))
        y1 = max(0, int(tick_pixel) - v_pad)
        y2 = min(h, int(tick_pixel) + v_pad)
        h_pad = padding * 3
        if axis.side == "right":
            x1 = int(axis.position) + 3
            x2 = min(w, int(axis.position) + h_pad)
        else:
            x1 = max(0, int(axis.position) - h_pad)
            x2 = max(0, int(axis.position) - 3)
            if x2 <= x1:
                x1, x2 = int(axis.position) + 3, min(w, int(axis.position) + h_pad)

    if x2 <= x1:
        x2 = min(w, x1 + 10)
    if y2 <= y1:
        y2 = min(h, y1 + 10)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def detect_tick_label_anchors(
    image: np.ndarray,
    axis,
    tick_pixels: List[int],
    policy=None,
) -> List[AxisLabelAnchor]:
    """Detect tick anchors and preserve the label crop used to read them."""
    init_tesseract()
    tick_pixels = sorted(tick_pixels)
    if not tick_pixels:
        return []

    blobs = _detect_label_blobs(image, axis)
    candidates: List[_LabelCandidate] = []
    h, w = image.shape[:2]
    for bx, by, bw, bh in blobs:
        cx1 = max(0, bx - 2)
        cy1 = max(0, by - 2)
        cx2 = min(w, bx + bw + 2)
        cy2 = min(h, by + bh + 2)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        text, value, contours_len = _ocr_tick_label_text_cached_for_crop(crop, policy=policy)
        if axis.direction == "x":
            center = int((cx1 + cx2) / 2)
        else:
            center = int((cy1 + cy2) / 2)
        label_area = max((cx2 - cx1) * (cy2 - cy1), 1)
        confidence = 0.15
        if value is not None:
            confidence = 0.75
        elif text:
            confidence = 0.35
        if contours_len == 2:
            confidence = min(1.0, confidence + 0.05)
        candidates.append(_LabelCandidate(
            bbox=(cx1, cy1, cx2, cy2),
            center=center,
            crop=crop,
            tesseract_text=text,
            tesseract_value=value,
            confidence=min(1.0, confidence + min(label_area / 4000.0, 0.2)),
        ))

    matched = _match_tick_label_candidates(tick_pixels, candidates)
    anchors: List[AxisLabelAnchor] = []
    empty_crop = np.empty((0, 0), dtype=image.dtype)
    for tick_pixel, candidate in matched:
        if candidate is None:
            crop, bbox = _crop_tick_label_region(image, axis, tick_pixel, padding=15)
            text, value, contours_len = _ocr_tick_label_text_cached_for_crop(crop, policy=policy) if crop.size > 0 else ("", None, 0)
            confidence = 0.0
            if value is not None:
                confidence = 0.45
            elif text:
                confidence = 0.2
            if contours_len == 2:
                confidence = min(1.0, confidence + 0.05)
            anchors.append(AxisLabelAnchor(
                tick_pixel=int(tick_pixel),
                label_bbox=bbox,
                label_center=int(tick_pixel),
                crop=crop if crop.size > 0 else empty_crop,
                tesseract_text=text or None,
                tesseract_value=value,
                confidence=float(confidence),
                source=(
                    "synthetic_ocr"
                    if (value is not None or text)
                    else "synthetic"
                ),
            ))
            continue
        anchors.append(AxisLabelAnchor(
            tick_pixel=int(tick_pixel),
            label_bbox=candidate.bbox,
            label_center=int(candidate.center),
            crop=candidate.crop,
            tesseract_text=candidate.tesseract_text,
            tesseract_value=candidate.tesseract_value,
            confidence=float(candidate.confidence),
            source=(
                "tesseract"
                if candidate.tesseract_value is not None
                else "ocr"
                if candidate.tesseract_text
                else "missing"
            ),
        ))
    return anchors


def read_all_tick_labels(
    image, axis, tick_pixels: List[int], padding=15, policy=None
) -> List[Tuple[int, Optional[float]]]:
    """Read labels for all detected ticks using bidirectional matching.

    Scatteract-style: detect label blobs independently, OCR each blob,
    then match to ticks via bidirectional nearest-neighbor validation.
    Falls back to direct per-tick OCR if bidirectional yields <2 matches.
    """
    init_tesseract()
    if not TESSERACT_AVAILABLE:
        return [(p, None) for p in tick_pixels]

    tick_pixels = sorted(tick_pixels)
    h, w = image.shape[:2]

    anchors = detect_tick_label_anchors(image, axis, tick_pixels, policy=policy)
    anchor_results = [(anchor.tick_pixel, anchor.tesseract_value) for anchor in anchors]
    anchor_valid = sum(1 for _, v in anchor_results if v is not None)
    min_matches = max(3, len(tick_pixels) * 2 // 3)
    if anchor_valid >= min_matches:
        return anchor_results

    # Fallback to direct per-tick OCR if anchor matching is too weak.
    direct_results = []
    for idx, pix in enumerate(tick_pixels):
        if axis.direction == "x":
            y1 = min(h, axis.position + 3)
            y2 = min(h, axis.position + padding * 2 + 5)
            x1 = max(0, pix - padding)
            x2 = min(w, pix + padding)
            crop = image[y1:y2, x1:x2]
        else:
            prev_pix = tick_pixels[idx - 1] if idx > 0 else pix - padding * 2
            next_pix = tick_pixels[idx + 1] if idx < len(tick_pixels) - 1 else pix + padding * 2
            half_gap = min(abs(pix - prev_pix), abs(next_pix - pix)) / 2.0
            v_pad = int(min(max(half_gap - 1, 10), padding * 1.5))
            y1 = max(0, pix - v_pad)
            y2 = min(h, pix + v_pad)
            h_pad = padding * 3
            if axis.side == "right":
                x1 = axis.position + 3
                x2 = min(w, axis.position + h_pad)
            else:
                x1 = max(0, axis.position - h_pad)
                x2 = max(0, axis.position - 3)
                if x2 <= x1:
                    x1, x2 = axis.position + 3, min(w, axis.position + h_pad)
            crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            direct_results.append((pix, None))
            continue
        val = read_tick_label(crop, policy=policy)
        direct_results.append((pix, val))

    direct_valid = sum(1 for _, v in direct_results if v is not None)
    if direct_valid >= anchor_valid:
        return direct_results
    return anchor_results
