"""OCR and tick label reading with fallback mechanisms."""
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from plot_extractor.utils.math_utils import parse_numeric


def _clean_ocr_text(text: str) -> str:
    """Clean OCR output to extract likely numeric text."""
    text = text.strip()
    # Remove common noise
    text = text.replace("|", "").replace("—", "-").replace("–", "-")
    return text


def read_tick_label(image_crop) -> Optional[float]:
    """Attempt OCR on a small crop around a tick label."""
    if not TESSERACT_AVAILABLE:
        return None
    try:
        # Convert numpy RGB to PIL if needed
        if len(image_crop.shape) == 3:
            from PIL import Image
            img = Image.fromarray(image_crop)
        else:
            from PIL import Image
            img = Image.fromarray(image_crop)
        text = pytesseract.image_to_string(img, config="--psm 7 -c tessedit_char_whitelist=0123456789.,-eE+kKmMbB%")
        text = _clean_ocr_text(text)
        val = parse_numeric(text)
        return val
    except Exception:
        return None


def read_all_tick_labels(image, axis, tick_pixels: List[int], padding=15) -> List[Tuple[int, Optional[float]]]:
    """Read labels for all detected ticks. Returns list of (pixel, value_or_None)."""
    h, w = image.shape[:2]
    results = []
    for pix in tick_pixels:
        if axis.direction == "x":
            # Label is below the tick
            y1 = min(h, axis.position + 5)
            y2 = min(h, axis.position + padding * 2)
            x1 = max(0, pix - padding)
            x2 = min(w, pix + padding)
            crop = image[y1:y2, x1:x2]
        else:
            # Label is left of the tick
            y1 = max(0, pix - padding)
            y2 = min(h, pix + padding)
            x1 = max(0, axis.position - padding * 3)
            x2 = max(0, axis.position - 5)
            if x2 <= x1:
                x1, x2 = axis.position + 5, min(w, axis.position + padding * 3)
            crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            results.append((pix, None))
            continue
        val = read_tick_label(crop)
        results.append((pix, val))
    return results


def load_meta_labels(image_path: Path) -> Optional[dict]:
    """Load ground-truth tick labels from meta JSON if available."""
    meta_path = image_path.parent / f"{image_path.stem}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None
