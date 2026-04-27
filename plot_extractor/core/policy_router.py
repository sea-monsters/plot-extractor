"""Policy router: convert chart-type probabilities into extraction config."""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from plot_extractor.core.chart_type_guesser import ImageFeatures
from plot_extractor.utils.math_utils import sigmoid as _sigmoid


@dataclass
class ExtractionPolicy:
    """Concrete extraction configuration produced by the policy ensemble."""

    # Preprocessing
    noise_strategy: str = "clean"          # "clean", "salt_pepper", "jpeg", "blur", "rotation_noise"
    median_ksize: int = 3
    bilateral_d: int = 9
    unsharp_amount: float = 0.0

    # Rotation
    rotation_correct: bool = False
    rotation_angle: float = 0.0

    # Color / extraction path
    color_strategy: str = "hue_only"       # "hue_only", "hsv3d", "layered", "none"
    min_clusters: int = 1
    density_strategy: str = "standard"     # "standard", "thinning", "scatter"
    thinning_quality_gate: float = 0.3

    # OCR
    ocr_block_size: int = 25
    ocr_C: int = 10
    ocr_deskew: bool = True

    # Axis detection
    hough_threshold: int = 100
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10

    # Calibration
    cal_residual_threshold_linear: float = 1e4
    cal_residual_threshold_log: float = 5e4


# ---------------------------------------------------------------------------
# Policy weight matrix
# Rows = strategies, Columns = chart types
# Negative weights suppress harmful strategies; positive weights activate.
# ---------------------------------------------------------------------------

CHART_TYPES = [
    "simple_linear", "log_y", "log_x", "loglog",
    "scatter", "multi_series", "dense", "dual_y",
    "inverted_y", "no_grid",
]

POLICY_WEIGHTS: Dict[str, Dict[str, float]] = {
    # ---- Noise / preprocessing ----
    "noise_median": {
        "simple_linear": 0.1, "log_y": 0.1, "log_x": 0.1, "loglog": 0.1,
        "scatter": -0.3, "multi_series": 0.0, "dense": 0.2,
        "dual_y": 0.1, "inverted_y": 0.1, "no_grid": 0.1,
    },
    "noise_bilateral": {
        "simple_linear": 0.0, "log_y": 0.0, "log_x": 0.0, "loglog": 0.0,
        "scatter": 0.0, "multi_series": 0.1, "dense": 0.0,
        "dual_y": 0.0, "inverted_y": 0.0, "no_grid": 0.0,
    },
    "noise_unsharp": {
        "simple_linear": -0.2, "log_y": -0.2, "log_x": -0.2, "loglog": -0.2,
        "scatter": 0.5, "multi_series": -0.1, "dense": -0.1,
        "dual_y": -0.2, "inverted_y": -0.2, "no_grid": -0.2,
    },
    # ---- OCR ----
    "ocr_adaptive": {
        "simple_linear": 0.2, "log_y": 0.2, "log_x": 0.2, "loglog": 0.2,
        "scatter": 0.0, "multi_series": 0.1, "dense": 0.1,
        "dual_y": 0.2, "inverted_y": 0.2, "no_grid": 0.2,
    },
    "ocr_deskew": {
        "simple_linear": 0.0, "log_y": 0.0, "log_x": 0.0, "loglog": 0.0,
        "scatter": 0.0, "multi_series": 0.0, "dense": 0.0,
        "dual_y": 0.0, "inverted_y": 0.0, "no_grid": 0.0,
    },
    # ---- Color / series separation ----
    "color_hsv3d": {
        "simple_linear": 0.0, "log_y": 0.0, "log_x": 0.0, "loglog": 0.0,
        "scatter": 0.0, "multi_series": 0.8, "dense": 0.0,
        "dual_y": 0.0, "inverted_y": 0.0, "no_grid": 0.0,
    },
    "color_layered": {
        "simple_linear": 0.0, "log_y": 0.0, "log_x": 0.0, "loglog": 0.0,
        "scatter": 0.0, "multi_series": 0.3, "dense": 0.2,
        "dual_y": 0.0, "inverted_y": 0.0, "no_grid": 0.0,
    },
    # ---- Extraction strategy ----
    "extract_thinning": {
        "simple_linear": -0.3, "log_y": -0.3, "log_x": -0.3, "loglog": -0.3,
        "scatter": -0.5, "multi_series": -0.2, "dense": 1.0,
        "dual_y": -0.3, "inverted_y": -0.3, "no_grid": -0.3,
    },
    "extract_scatter": {
        "simple_linear": 0.0, "log_y": 0.0, "log_x": 0.0, "loglog": 0.0,
        "scatter": 1.0, "multi_series": 0.0, "dense": -0.3,
        "dual_y": 0.0, "inverted_y": 0.0, "no_grid": 0.0,
    },
    "extract_cc": {
        "simple_linear": 0.2, "log_y": 0.2, "log_x": 0.2, "loglog": 0.2,
        "scatter": -0.3, "multi_series": 0.1, "dense": -0.5,
        "dual_y": 0.2, "inverted_y": 0.2, "no_grid": 0.2,
    },
    # ---- Rotation ----
    "rotation_correct": {
        "simple_linear": 0.0, "log_y": 0.0, "log_x": 0.0, "loglog": 0.0,
        "scatter": 0.0, "multi_series": 0.0, "dense": 0.0,
        "dual_y": 0.0, "inverted_y": 0.0, "no_grid": 0.0,
    },
}



def _strategy_activation(
    strategy_weights: Dict[str, float],
    type_probs: Dict[str, float],
) -> float:
    """Compute activation strength for a strategy given type probabilities."""
    score = 0.0
    for ctype, prob in type_probs.items():
        w = strategy_weights.get(ctype, 0.0)
        score += prob * w
    # Shift so that score == 0 maps to ~0.5 activation
    return float(_sigmoid(score * 3.0))


def _apply_noise_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Feature-gated noise strategy with type-weighted fine-tuning."""
    # Base routing from image quality features
    if features.laplacian_variance > 80 and features.extreme_pixel_ratio > 0.005:
        base_strategy = "salt_pepper"
    elif features.block_variance_ratio > 2.0:
        base_strategy = "jpeg"
    elif features.fft_high_freq_ratio < 0.08:
        base_strategy = "blur"
    elif abs(features.rotation_estimate) > 0.3:
        base_strategy = "rotation_noise"
    else:
        base_strategy = "clean"

    # Type-probability modulation
    if activation > 0.6:
        policy.noise_strategy = base_strategy
    elif activation < 0.3 and base_strategy != "clean":
        # Strong negative weight: force clean path
        policy.noise_strategy = "clean"
    else:
        policy.noise_strategy = base_strategy

    # Parameter tuning based on strategy
    if policy.noise_strategy == "salt_pepper":
        policy.median_ksize = 3
    elif policy.noise_strategy == "jpeg":
        policy.bilateral_d = 9
    elif policy.noise_strategy == "blur":
        policy.unsharp_amount = 1.5
    elif policy.noise_strategy == "rotation_noise":
        policy.unsharp_amount = 1.0


def _apply_rotation_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Rotation correction based on feature estimate, modulated by type weights."""
    if abs(features.rotation_estimate) >= 0.3:
        policy.rotation_correct = True
        policy.rotation_angle = features.rotation_estimate
    else:
        policy.rotation_correct = False
        policy.rotation_angle = 0.0


def _apply_color_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Color separation strategy."""
    if activation > 0.6:
        policy.color_strategy = "hsv3d"
        policy.min_clusters = max(2, features.hue_peak_count)
    elif activation > 0.4:
        policy.color_strategy = "layered"
        policy.min_clusters = max(2, features.hue_peak_count)
    else:
        policy.color_strategy = "hue_only"
        policy.min_clusters = max(1, features.hue_peak_count)


def _apply_thinning_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Activate thinning for dense charts."""
    if activation > 0.6:
        policy.density_strategy = "thinning"
    elif activation < 0.25 and policy.density_strategy == "thinning":
        policy.density_strategy = "standard"


def _apply_scatter_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Activate scatter extraction for scatter charts."""
    if activation > 0.6:
        policy.density_strategy = "scatter"
    elif activation < 0.25 and policy.density_strategy == "scatter":
        policy.density_strategy = "standard"


def _apply_cc_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Standard/CC extraction is the fallback; strong activation reinforces it."""
    if activation > 0.7 and policy.density_strategy not in ("thinning", "scatter"):
        policy.density_strategy = "standard"


def _apply_ocr_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """OCR preprocessing parameters."""
    # Adaptive block size based on noise type
    if features.block_variance_ratio > 2.0:
        policy.ocr_block_size = 31  # Larger block for JPEG artifacts
    elif features.laplacian_variance > 80:
        policy.ocr_block_size = 21  # Medium block for salt & pepper
    else:
        policy.ocr_block_size = 25

    # C parameter: higher for noisier images
    if features.laplacian_variance > 100:
        policy.ocr_C = 15
    elif features.laplacian_variance > 60:
        policy.ocr_C = 12
    else:
        policy.ocr_C = 10

    policy.ocr_deskew = abs(features.rotation_estimate) > 0.3


def _apply_hough_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Axis detection Hough parameter tuning."""
    # Lower threshold for low-edge-density images (no_grid, faint axes)
    if features.edge_density < 0.05:
        policy.hough_threshold = max(50, int(100 - 50 * (0.05 - features.edge_density) * 20))
    else:
        policy.hough_threshold = 100

    policy.hough_min_line_length = int(min(features.aspect_ratio, 1.0) * 50)


def _apply_cal_strategy(policy: ExtractionPolicy, activation: float, features: ImageFeatures):
    """Calibration fallback strategy."""
    # Tighter thresholds for clean images (use texture features as proxy)
    is_clean = (
        features.laplacian_variance < 60
        and features.block_variance_ratio < 1.5
        and features.fft_high_freq_ratio > 0.1
    )
    if is_clean:
        policy.cal_residual_threshold_linear = 5e3
        policy.cal_residual_threshold_log = 2e4
    else:
        policy.cal_residual_threshold_linear = 1e4
        policy.cal_residual_threshold_log = 5e4


STRATEGY_APPLIERS = {
    "noise_median": _apply_noise_strategy,
    "noise_bilateral": _apply_noise_strategy,
    "noise_unsharp": _apply_noise_strategy,
    "ocr_adaptive": _apply_ocr_strategy,
    "ocr_deskew": _apply_ocr_strategy,
    "color_hsv3d": _apply_color_strategy,
    "color_layered": _apply_color_strategy,
    "extract_thinning": _apply_thinning_strategy,
    "extract_scatter": _apply_scatter_strategy,
    "extract_cc": _apply_cc_strategy,
    "rotation_correct": _apply_rotation_strategy,
}


def compute_policy(
    features: ImageFeatures,
    type_probs: Dict[str, float],
) -> ExtractionPolicy:
    """Compute extraction policy from image features and type probabilities.

    The policy is built by evaluating each strategy's activation strength
    (type-probability weighted) and applying feature-gated parameter choices.
    """
    policy = ExtractionPolicy()

    # Collect all activations for conflict resolution
    activations = {}
    for strategy_name, weights in POLICY_WEIGHTS.items():
        activations[strategy_name] = _strategy_activation(weights, type_probs)

    # Resolve density_strategy collision: pick highest activation among competitors
    density_candidates = {
        "thinning": activations.get("extract_thinning", 0.0),
        "scatter": activations.get("extract_scatter", 0.0),
        "standard": activations.get("extract_cc", 0.0),
    }
    best_density = max(density_candidates, key=density_candidates.get)
    if density_candidates[best_density] > 0.5:
        policy.density_strategy = best_density

    # Apply non-density strategies
    for strategy_name, activation in activations.items():
        if strategy_name in ("extract_thinning", "extract_scatter", "extract_cc"):
            continue  # Already resolved above
        applier = STRATEGY_APPLIERS.get(strategy_name)
        if applier:
            applier(policy, activation, features)

    # Final feature-gated overrides (independent of type weights)
    _apply_hough_strategy(policy, 0.5, features)

    return policy
