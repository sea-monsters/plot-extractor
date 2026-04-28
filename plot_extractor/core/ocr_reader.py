"""OCR and tick label reading with fallback mechanisms."""
import glob
import os
import shutil
import sys
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


def _clean_ocr_text(text: str) -> str:
    """Clean OCR output to extract likely numeric text."""
    text = text.strip()
    # Remove common noise
    text = text.replace("|", "").replace("—", "-").replace("–", "-")
    return text


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
    init_tesseract()
    if not TESSERACT_AVAILABLE:
        return None
    try:
        preprocessed, contours_len = _preprocess_tick_crop(image_crop, policy=policy)
        from PIL import Image
        img = Image.fromarray(preprocessed)

        # Try multiple PSM modes; single-character/number modes work best for ticks
        configs = [
            "--psm 8 -c tessedit_char_whitelist=0123456789.,-eE+kKmMbB%",
            "--psm 7 -c tessedit_char_whitelist=0123456789.,-eE+kKmMbB%",
        ]
        for cfg in configs:
            text = pytesseract.image_to_string(img, config=cfg)
            text = _clean_ocr_text(text)
            val = parse_numeric(text)
            # Keep negative labels (e.g. axes spanning below zero).
            if val is not None:
                return val

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
                        return val

        return None
    except OSError:
        return None


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

    # --- Phase 1: Bidirectional matching (Scatteract-style) ---
    blobs = _detect_label_blobs(image, axis)
    labels = []
    for bx, by, bw, bh in blobs:
        # Expand slightly for OCR
        cx1 = max(0, bx - 2)
        cy1 = max(0, by - 2)
        cx2 = min(w, bx + bw + 2)
        cy2 = min(h, by + bh + 2)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        val = read_tick_label(crop, policy=policy)
        if val is not None:
            # Label position = center along the axis direction
            if axis.direction == "x":
                label_pos = int((cx1 + cx2) / 2)
            else:
                label_pos = int((cy1 + cy2) / 2)
            labels.append((label_pos, val))

    # Remove duplicate positions (keep first / best)
    seen_pos = set()
    unique_labels = []
    for pos, val in sorted(labels, key=lambda x: x[1] is not None, reverse=True):
        # Allow small tolerance for duplicate detection
        is_dup = any(abs(pos - sp) < 5 for sp in seen_pos)
        if not is_dup:
            seen_pos.add(pos)
            unique_labels.append((pos, val))

    bidirectional_results = _match_ticks_labels_bidirectional(tick_pixels, unique_labels)
    bidirectional_valid = sum(1 for _, v in bidirectional_results if v is not None)

    # --- Phase 2: Fallback to direct per-tick OCR if bidirectional too weak ---
    # Only trust bidirectional if it matches a strong majority of ticks
    # and has at least 3 valid labels.
    min_matches = max(3, len(tick_pixels) * 2 // 3)
    if bidirectional_valid >= min_matches:
        return bidirectional_results

    # Direct OCR at each tick position (original method)
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

    # Prefer whichever method yielded more valid labels
    direct_valid = sum(1 for _, v in direct_results if v is not None)
    if direct_valid >= bidirectional_valid:
        return direct_results
    return bidirectional_results
