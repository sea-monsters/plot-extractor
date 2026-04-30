"""Adaptive strategy selector for extraction policy routing.

Replaces fixed POLICY_WEIGHTS matrix with a decision tree that uses
diagnostic signals from ImageFeatures. Based on CACHED multi-context
fusion concept (2305.04151) and 7 failed threshold-tuning rounds.

Decision tree:
  1. Grid → morphological grid removal parameters
  2. n_colors > 1 → HSV clustering separation + CC continuity verification
  3. is_dense → thinning → path tracing
  4. is_scatter → CC centroid extraction
  5. has_log_axis → disable scatter fallback
  6. noise_level → OCR preprocessing parameters
"""
from plot_extractor.core.chart_type_guesser import ImageFeatures
from plot_extractor.core.policy_router import ExtractionPolicy
from typing import Dict


def compute_adaptive_policy(
    features: ImageFeatures,
    type_probs: Dict[str, float],
) -> ExtractionPolicy:
    """Compute extraction policy using a decision tree.

    Uses diagnostic signals (features) and type probabilities together,
    but relies primarily on measurable image properties for routing,
    with type probabilities as soft confirmation.

    Args:
        features: Extracted image features.
        type_probs: Chart type probability dict.

    Returns:
        ExtractionPolicy configured for the detected signal combination.
    """
    policy = ExtractionPolicy()

    # --- Density strategy: the most impactful routing decision ---
    is_dense = features.fg_col_density > 0.6 and features.cc_area_mean < 50
    is_scatter = (features.saturation_mean > 200
                  and features.cc_area_mean > 20
                  and features.cc_area_mean < 60
                  and features.fg_col_density < 0.5)
    has_log_axis = max(
        type_probs.get("log_y", 0), type_probs.get("log_x", 0),
        type_probs.get("loglog", 0),
    ) > 0.3

    # Type probability reinforcement
    dense_prob = type_probs.get("dense", 0)
    scatter_prob = type_probs.get("scatter", 0)

    if dense_prob > 0.5 or (is_dense and dense_prob > 0.2):
        policy.density_strategy = "thinning"
    elif scatter_prob > 0.5 and not has_log_axis:
        policy.density_strategy = "scatter"
    elif is_scatter and not has_log_axis and scatter_prob > 0.2:
        policy.density_strategy = "scatter"
    elif is_dense:
        policy.density_strategy = "thinning"
    else:
        policy.density_strategy = "standard"

    # --- Color strategy: multi-series separation ---
    n_colors = max(1, features.hue_peak_count)
    multi_prob = type_probs.get("multi_series", 0)
    dual_prob = type_probs.get("dual_y", 0)

    if n_colors >= 3 and (multi_prob > 0.3 or dual_prob > 0.3):
        policy.color_strategy = "hsv3d"
        policy.min_clusters = n_colors
    elif n_colors >= 2 and multi_prob > 0.2:
        policy.color_strategy = "layered"
        policy.min_clusters = n_colors
    else:
        policy.color_strategy = "hue_only"
        policy.min_clusters = 1

    # --- Noise strategy: preprocessing ---
    if features.laplacian_variance > 80 and features.extreme_pixel_ratio > 0.005:
        policy.noise_strategy = "salt_pepper"
        policy.median_ksize = 3
    elif features.block_variance_ratio > 2.0:
        policy.noise_strategy = "jpeg"
        policy.bilateral_d = 9
    elif features.fft_high_freq_ratio < 0.08:
        policy.noise_strategy = "blur"
        policy.unsharp_amount = 1.5
    else:
        policy.noise_strategy = "clean"

    # --- Rotation correction ---
    if abs(features.rotation_estimate) >= 0.3:
        policy.rotation_correct = True
        policy.rotation_angle = features.rotation_estimate
    else:
        policy.rotation_correct = False
        policy.rotation_angle = 0.0

    # --- OCR preprocessing ---
    if features.block_variance_ratio > 2.0:
        policy.ocr_block_size = 31
    elif features.laplacian_variance > 80:
        policy.ocr_block_size = 21
    else:
        policy.ocr_block_size = 25

    if features.laplacian_variance > 100:
        policy.ocr_C = 15
    elif features.laplacian_variance > 60:
        policy.ocr_C = 12
    else:
        policy.ocr_C = 10

    policy.ocr_deskew = abs(features.rotation_estimate) > 0.3

    # --- Axis detection Hough parameters ---
    if features.edge_density < 0.05:
        policy.hough_threshold = max(50, int(100 - 50 * (0.05 - features.edge_density) * 20))
    else:
        policy.hough_threshold = 100

    policy.hough_min_line_length = int(min(features.aspect_ratio, 1.0) * 50)

    # --- Calibration thresholds ---
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

    # --- Thinning quality gate ---
    if policy.density_strategy == "thinning":
        policy.thinning_quality_gate = 0.2 if is_dense else 0.3

    return policy
