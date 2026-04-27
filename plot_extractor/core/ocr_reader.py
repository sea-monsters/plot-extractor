"""OCR and tick label reading with fallback mechanisms."""
import glob
import json
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


def _deskew_crop(crop, max_angle=5.0):
    """Deskew a text crop using moments."""
    gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return crop
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > max_angle:
        return crop
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return rotated


def _preprocess_tick_crop(crop: np.ndarray, policy=None) -> np.ndarray:
    """Enhance a tick-label crop for OCR.

    Operates only on the crop; never mutates the global preprocessed image.
    """
    if crop.size == 0:
        return crop

    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop.copy()

    # Deskew for rotated labels
    gray = _deskew_crop(gray)

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

    return preprocessed


def read_tick_label(image_crop, policy=None) -> Optional[float]:
    """Attempt OCR on a small crop around a tick label."""
    init_tesseract()
    if not TESSERACT_AVAILABLE:
        return None
    try:
        preprocessed = _preprocess_tick_crop(image_crop, policy=policy)
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
        return None
    except OSError:
        return None


def read_all_tick_labels(image, axis, tick_pixels: List[int], padding=15, policy=None) -> List[Tuple[int, Optional[float]]]:
    """Read labels for all detected ticks. Returns list of (pixel, value_or_None)."""
    init_tesseract()
    if not TESSERACT_AVAILABLE:
        return [(p, None) for p in tick_pixels]

    h, w = image.shape[:2]
    results = []
    tick_pixels = sorted(tick_pixels)

    for idx, pix in enumerate(tick_pixels):
        if axis.direction == "x":
            # Label is below the tick; allow modest height for multi-line labels
            y1 = min(h, axis.position + 3)
            y2 = min(h, axis.position + padding * 2 + 5)
            x1 = max(0, pix - padding)
            x2 = min(w, pix + padding)
            crop = image[y1:y2, x1:x2]
        else:
            # Label is left or right of the tick
            # Compute vertical bounds: use neighbour distance to avoid overlap,
            # but enforce a minimum height so OCR has enough pixels to work with
            prev_pix = tick_pixels[idx - 1] if idx > 0 else pix - padding * 2
            next_pix = tick_pixels[idx + 1] if idx < len(tick_pixels) - 1 else pix + padding * 2
            half_gap = min(abs(pix - prev_pix), abs(next_pix - pix)) / 2.0
            v_pad = int(min(max(half_gap - 1, 10), padding * 1.5))
            y1 = max(0, pix - v_pad)
            y2 = min(h, pix + v_pad)

            # Horizontal: left-side labels are common; right-side for inverted axes
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
            results.append((pix, None))
            continue
        val = read_tick_label(crop, policy=policy)
        results.append((pix, val))
    return results

