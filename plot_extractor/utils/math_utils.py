"""Math utilities for parsing numbers and fitting axes."""
import re
import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


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
    try:
        coeffs = np.polyfit(values, pixels, 1)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None, None, np.inf
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
    try:
        coeffs = np.polyfit(log_vals, pixels, 1)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None, None, np.inf
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


def _is_geometric_sequence(values, tol=0.25):
    """Check if values form an approximate geometric sequence (log indicator).

    Uses log-space differences: true geometric sequences have constant
    log-differences, whereas arithmetic sequences have decreasing log-diffs.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return False
    if np.any(values <= 0):
        return False
    log_vals = np.log10(values)
    diffs = np.diff(log_vals)
    mean_diff = float(np.mean(np.abs(diffs)))
    if mean_diff == 0:
        return True
    if mean_diff < 0.01:
        # Too small to distinguish from noise
        return False
    cv = float(np.std(diffs) / mean_diff)
    return cv < tol


def _is_arithmetic_sequence(values, tol=0.25):
    """Check if values form an approximate arithmetic sequence (linear indicator)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return False
    diffs = np.diff(values)
    mean_diff = float(np.mean(np.abs(diffs)))
    if mean_diff == 0:
        return True
    cv = float(np.std(diffs) / mean_diff)
    return cv < tol


def classify_axis(pixels, values, preferred_type=None):
    """Classify axis as linear or log based on value sequence + R² comparison."""
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)

    a_lin, b_lin, res_lin = fit_linear(pixels, values)
    a_log, b_log, res_log = fit_log(pixels, values)

    # Determine fits for R² comparison
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

    # Use value-sequence pattern as primary signal (robust to bad tick pixels)
    is_geom = _is_geometric_sequence(values)
    is_arith = _is_arithmetic_sequence(values)

    if is_geom and not is_arith:
        axis_type = "log"
    elif is_arith and not is_geom:
        axis_type = "linear"
    else:
        # Ambiguous or too few points: fall back to R² comparison
        if r2_log > r2_lin + 0.05:
            axis_type = "log"
        else:
            axis_type = "linear"

    # Preferred type override: strong signal from policy ensemble
    if preferred_type == "log" and a_log is not None:
        axis_type = "log"
    elif preferred_type == "linear" and a_lin is not None:
        axis_type = "linear"

    if axis_type == "log" and a_log is not None:
        return "log", (a_log, b_log), res_log
    if axis_type == "linear" and a_lin is not None:
        return "linear", (a_lin, b_lin), res_lin

    # Last resort
    if a_log is not None:
        return "log", (a_log, b_log), res_log
    return "linear", (a_lin, b_lin), res_lin


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
