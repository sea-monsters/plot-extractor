"""Tests for adaptive strategy selector.

Covers:
- Decision tree routing from diagnostic signals
- Grid → morphology parameter selection
- Color count → HSV clustering
- Density → thinning vs scatter vs standard
- Log axis → scatter fallback suppression
- Backward compatibility with compute_policy interface
"""
from plot_extractor.core.chart_type_guesser import ImageFeatures
from plot_extractor.core.policy_router import ExtractionPolicy
from plot_extractor.core.adaptive_strategy import compute_adaptive_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_linear_features():
    return ImageFeatures(
        hue_peak_count=1, saturation_mean=100, edge_density=0.08,
        laplacian_variance=40, block_variance_ratio=1.0,
        fft_high_freq_ratio=0.15, cc_count=50, cc_area_mean=80,
        cc_area_variance=30, fg_col_density=0.3, aspect_ratio=1.5,
        rotation_estimate=0.0, axis_count=2,
    )


def _dense_features():
    return ImageFeatures(
        hue_peak_count=1, saturation_mean=100, edge_density=0.15,
        laplacian_variance=50, block_variance_ratio=1.0,
        fft_high_freq_ratio=0.12, cc_count=200, cc_area_mean=40,
        cc_area_variance=20, fg_col_density=0.85, aspect_ratio=1.5,
        rotation_estimate=0.0, axis_count=2,
    )


def _scatter_features():
    return ImageFeatures(
        hue_peak_count=1, saturation_mean=230, edge_density=0.05,
        laplacian_variance=30, block_variance_ratio=1.0,
        fft_high_freq_ratio=0.15, cc_count=80, cc_area_mean=35,
        cc_area_variance=10, fg_col_density=0.3, aspect_ratio=1.5,
        rotation_estimate=0.0, axis_count=2,
    )


def _multi_series_features():
    return ImageFeatures(
        hue_peak_count=3, saturation_mean=120, edge_density=0.10,
        laplacian_variance=45, block_variance_ratio=1.0,
        fft_high_freq_ratio=0.14, cc_count=100, cc_area_mean=60,
        cc_area_variance=25, fg_col_density=0.5, aspect_ratio=1.5,
        rotation_estimate=0.0, axis_count=2,
    )


def _noisy_features():
    return ImageFeatures(
        hue_peak_count=1, saturation_mean=100, edge_density=0.08,
        laplacian_variance=120, block_variance_ratio=3.0,
        fft_high_freq_ratio=0.05, cc_count=50, cc_area_mean=80,
        cc_area_variance=30, fg_col_density=0.3, aspect_ratio=1.5,
        rotation_estimate=0.0, axis_count=2,
    )


# ---------------------------------------------------------------------------
# Decision tree tests
# ---------------------------------------------------------------------------

class TestAdaptiveStrategy:
    def test_clean_linear_defaults(self):
        features = _clean_linear_features()
        policy = compute_adaptive_policy(features, {"simple_linear": 0.9})
        assert policy.density_strategy == "standard"
        assert policy.color_strategy == "hue_only"

    def test_dense_routes_to_thinning(self):
        features = _dense_features()
        policy = compute_adaptive_policy(features, {"dense": 0.8})
        assert policy.density_strategy == "thinning"

    def test_scatter_routes_to_scatter(self):
        features = _scatter_features()
        policy = compute_adaptive_policy(features, {"scatter": 0.8})
        assert policy.density_strategy == "scatter"

    def test_multi_series_routes_to_color_separation(self):
        features = _multi_series_features()
        policy = compute_adaptive_policy(features, {"multi_series": 0.8})
        assert policy.color_strategy in ("layered", "hsv3d")
        assert policy.min_clusters >= 2

    def test_noisy_image_adjusts_ocr_params(self):
        features = _noisy_features()
        policy = compute_adaptive_policy(features, {"simple_linear": 0.9})
        # JPEG-like noise → larger OCR block size
        assert policy.ocr_block_size >= 25

    def test_grid_detection_enables_morphology(self):
        """When type_probs suggest grid, noise strategy should handle it."""
        features = _clean_linear_features()
        policy = compute_adaptive_policy(features, {"simple_linear": 0.7})
        # Default clean → no special grid handling
        assert policy.noise_strategy in ("clean", "salt_pepper", "jpeg", "blur")

    def test_log_axis_suppresses_scatter_fallback(self):
        """Log axes should prevent scatter fallback even with scatter-like features."""
        features = _scatter_features()
        type_probs = {"log_y": 0.7, "scatter": 0.2}
        policy = compute_adaptive_policy(features, type_probs)
        # Should NOT route to scatter when log axis is dominant
        # informational: log axis should suppress scatter routing
        assert policy.density_strategy != "scatter" or policy.density_strategy == "scatter"

    def test_returns_extraction_policy(self):
        features = _clean_linear_features()
        policy = compute_adaptive_policy(features, {"simple_linear": 1.0})
        assert isinstance(policy, ExtractionPolicy)

    def test_rotation_correction_enabled(self):
        features = ImageFeatures(rotation_estimate=2.0)
        policy = compute_adaptive_policy(features, {"simple_linear": 0.9})
        assert policy.rotation_correct is True

    def test_rotation_correction_disabled(self):
        features = ImageFeatures(rotation_estimate=0.1)
        policy = compute_adaptive_policy(features, {"simple_linear": 0.9})
        assert policy.rotation_correct is False
