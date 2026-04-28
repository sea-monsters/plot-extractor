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


def _compute_ransac_threshold(pixels):
    """Auto-compute RANSAC residual threshold from tick spacing.

    Uses 15% of median tick spacing, following Scatteract's robust
    approach where inliers are points within a fraction of the
    natural scale of the data.
    """
    pixels = np.asarray(pixels, dtype=float)
    if len(pixels) < 2:
        return 5.0
    sorted_pixels = np.sort(pixels)
    spacings = np.diff(sorted_pixels)
    if len(spacings) == 0:
        return 5.0
    median_spacing = np.median(spacings)
    return max(median_spacing * 0.15, 1.0)


def fit_linear_ransac(
    pixels, values, residual_threshold=None, max_trials=100, random_state=42
):
    """Fit pixel = a * value + b using RANSAC robust regression.

    Scatteract-style: iteratively sample minimal subsets, fit model,
    count inliers, then refit with best inlier set.

    Returns (a, b, residual, inlier_indices).
    """
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(pixels)
    if n < 2:
        return None, None, np.inf, []

    if residual_threshold is None:
        residual_threshold = _compute_ransac_threshold(pixels)

    if n == 2:
        a, b, res = fit_linear(pixels, values)
        return a, b, res, [0, 1]

    rng = np.random.default_rng(random_state)
    best_inliers = []
    best_a, best_b = None, None

    for _ in range(max_trials):
        idx = rng.choice(n, 2, replace=False)
        sample_pixels = pixels[idx]
        sample_values = values[idx]

        a, b, _ = fit_linear(sample_pixels, sample_values)
        if a is None:
            continue

        pred = a * values + b
        residuals = np.abs(pixels - pred)
        inliers = np.where(residuals < residual_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_a, best_b = a, b

    if len(best_inliers) < 2:
        a, b, res = fit_linear(pixels, values)
        return a, b, res, list(range(n))

    inlier_pixels = pixels[best_inliers]
    inlier_values = values[best_inliers]
    a, b, res = fit_linear(inlier_pixels, inlier_values)
    return a, b, res, list(best_inliers)


def fit_log_ransac(
    pixels, values, residual_threshold=None, max_trials=100, random_state=42
):
    """Fit pixel = a * log10(value) + b using RANSAC robust regression.

    Non-positive values are filtered out before fitting rather than causing
    total failure — this prevents a single OCR misread (e.g. "0") from
    disabling log calibration entirely.

    Returns (a, b, residual, inlier_indices) where inlier_indices refer to
    the *filtered* positive subset, not the original array.
    """
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)

    # Filter non-positive values (log domain requirement)
    positive_mask = values > 0
    if np.sum(positive_mask) < 2:
        return None, None, np.inf, []
    pixels = pixels[positive_mask]
    values = values[positive_mask]

    n = len(pixels)
    if n < 2:
        return None, None, np.inf, []

    log_vals = np.log10(values)

    if residual_threshold is None:
        residual_threshold = _compute_ransac_threshold(pixels)

    if n == 2:
        a, b, res = fit_log(pixels, values)
        return a, b, res, [0, 1]

    rng = np.random.default_rng(random_state)
    best_inliers = []
    best_a, best_b = None, None

    for _ in range(max_trials):
        idx = rng.choice(n, 2, replace=False)
        sample_pixels = pixels[idx]
        sample_values = log_vals[idx]

        a, b, _ = fit_linear(sample_pixels, sample_values)
        if a is None:
            continue

        pred = a * log_vals + b
        residuals = np.abs(pixels - pred)
        inliers = np.where(residuals < residual_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_a, best_b = a, b

    if len(best_inliers) < 2:
        a, b, res = fit_log(pixels, values)
        return a, b, res, list(range(n))

    inlier_pixels = pixels[best_inliers]
    inlier_values = log_vals[best_inliers]
    a, b, res = fit_linear(inlier_pixels, inlier_values)
    return a, b, res, list(best_inliers)


def fit_log(pixels, values):
    """Fit pixel = a * log10(value) + b. Returns (a, b, residuals)."""
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)
    positive_mask = values > 0
    if np.sum(positive_mask) < 2:
        return None, None, np.inf
    pixels = pixels[positive_mask]
    values = values[positive_mask]
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
    """Classify axis as linear or log based on RANSAC robust regression.

    Scatteract-style: use RANSAC to handle OCR outliers, then compare
    fit quality between linear and log models.
    """
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)

    # Use RANSAC when enough points to benefit from outlier rejection
    if len(pixels) >= 3:
        a_lin, b_lin, res_lin, inliers_lin = fit_linear_ransac(pixels, values)
        a_log, b_log, res_log, inliers_log = fit_log_ransac(pixels, values)
    else:
        a_lin, b_lin, res_lin = fit_linear(pixels, values)
        a_log, b_log, res_log = fit_log(pixels, values)
        inliers_lin = inliers_log = list(range(len(pixels)))

    # Compute R² on inliers for comparison
    if a_lin is not None and inliers_lin:
        inlier_pixels_lin = pixels[inliers_lin]
        inlier_values_lin = values[inliers_lin]
        pred_lin = a_lin * inlier_values_lin + b_lin
        r2_lin = _r_squared(inlier_pixels_lin, pred_lin)
    else:
        r2_lin = -np.inf

    if a_log is not None and inliers_log:
        inlier_pixels_log = pixels[inliers_log]
        inlier_values_log = values[inliers_log]
        log_vals = np.log10(inlier_values_log)
        pred_log = a_log * log_vals + b_log
        r2_log = _r_squared(inlier_pixels_log, pred_log)
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
