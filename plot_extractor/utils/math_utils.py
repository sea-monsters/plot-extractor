"""Math utilities for parsing numbers and fitting axes."""
import re
import numpy as np


_RE_NUM = re.compile(
    r"^\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|"  # 1,000.50
    r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?|"          # 1.2e3
    r"[+-]?\d+)\s*"                              # 42
    r"([kKmMbB%]?)$"                             # suffix
)


_SUFFIX_MUL = {
    "k": 1e3, "K": 1e3,
    "m": 1e6, "M": 1e6,
    "b": 1e9, "B": 1e9,
    "%": 0.01,
}


def parse_numeric(text: str) -> float | None:
    """Parse a numeric string like '1,000', '1.5e3', '10k', '50%'."""
    text = text.strip().replace(",", "")
    # Handle superscripts like 10² → 10^2
    text = text.replace("²", "^2").replace("³", "^3")
    if "^" in text:
        try:
            base, exp = text.split("^", 1)
            return float(base.strip()) ** float(exp.strip())
        except (ValueError, IndexError):
            pass
    m = _RE_NUM.match(text)
    if not m:
        return None
    num_str, suffix = m.groups()
    try:
        val = float(num_str)
    except ValueError:
        return None
    if suffix in _SUFFIX_MUL:
        val *= _SUFFIX_MUL[suffix]
    return val


def fit_linear(pixels, values):
    """Fit pixel = a * value + b. Returns (a, b, residuals)."""
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(pixels) < 2:
        return None, None, np.inf
    coeffs = np.polyfit(values, pixels, 1)
    a, b = coeffs
    pred = a * values + b
    residuals = np.mean((pixels - pred) ** 2)
    return a, b, residuals


def fit_log(pixels, values):
    """Fit pixel = a * log10(value) + b. Returns (a, b, residuals)."""
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        return None, None, np.inf
    log_vals = np.log10(values)
    if len(pixels) < 2:
        return None, None, np.inf
    coeffs = np.polyfit(log_vals, pixels, 1)
    a, b = coeffs
    pred = a * log_vals + b
    residuals = np.mean((pixels - pred) ** 2)
    return a, b, residuals


def _r_squared(actual, predicted):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def classify_axis(pixels, values):
    """Classify axis as linear or log based on R² comparison."""
    a_lin, b_lin, _ = fit_linear(pixels, values)
    a_log, b_log, _ = fit_log(pixels, values)

    if a_lin is not None:
        pred_lin = a_lin * values + b_lin
        r2_lin = _r_squared(pixels, pred_lin)
    else:
        r2_lin = -np.inf

    if a_log is not None:
        log_vals = np.log10(values)
        pred_log = a_log * log_vals + b_log
        r2_log = _r_squared(pixels, pred_log)
    else:
        r2_log = -np.inf

    if r2_lin >= r2_log:
        _, _, res_lin = fit_linear(pixels, values)
        return "linear", (a_lin, b_lin), res_lin
    _, _, res_log = fit_log(pixels, values)
    return "log", (a_log, b_log), res_log


def pixel_to_data(pixel, a, b, axis_type, inverted=False):
    """Convert pixel coordinate to data value."""
    if a is None or b is None:
        return None
    if axis_type == "log":
        if a == 0:
            return None
        if inverted:
            return 10 ** ((-pixel - b) / a)
        return 10 ** ((pixel - b) / a)
    if a == 0:
        return None
    if inverted:
        return (-pixel - b) / a
    return (pixel - b) / a


def data_to_pixel(value, a, b, axis_type):
    """Convert data value to pixel coordinate."""
    if a is None or b is None:
        return None
    if axis_type == "log":
        if value <= 0:
            return None
        return a * np.log10(value) + b
    return a * value + b
