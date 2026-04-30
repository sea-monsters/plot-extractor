"""Tests for multi-hypothesis axis calibration.

Covers:
- CalibrationResult dataclass
- fit_axis_multi_hypothesis: selecting best fit from linear/log/polynomial
- is_calibration_plausible: physical plausibility gating
- infer_log_values_from_spacing: fallback when OCR fails entirely
- grade_tick_quality: tick reliability scoring
"""
import numpy as np
from plot_extractor.core.axis_calibrator import (
    CalibrationResult,
    fit_axis_multi_hypothesis,
    is_calibration_plausible,
    infer_log_values_from_spacing,
    grade_tick_quality,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic tick data
# ---------------------------------------------------------------------------

def _linear_ticks(n=8, a=2.0, b=50.0, noise=0.0, rng=42):
    """Generate linear ticks: pixel = a * value + b."""
    rng = np.random.default_rng(rng)
    values = np.linspace(0, 10, n)
    pixels = a * values + b + rng.normal(0, noise, n)
    return list(zip(pixels.tolist(), values.tolist()))


def _log_ticks(n=8, a=200.0, b=50.0, noise=0.0, rng=42):
    """Generate log ticks: pixel = a * log10(value) + b."""
    rng = np.random.default_rng(rng)
    values = np.array([1, 10, 100, 1000, 10000, 1e5, 1e6, 1e7])[:n]
    pixels = a * np.log10(values) + b + rng.normal(0, noise, n)
    return list(zip(pixels.tolist(), values.tolist()))


def _log_ticks_pixel_only(n=8, a=200.0, b=50.0):
    """Generate tick pixel positions for log axis (no values)."""
    values = np.array([1, 10, 100, 1000, 10000, 1e5, 1e6, 1e7])[:n]
    pixels = a * np.log10(values) + b
    return pixels.tolist()


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def test_stores_model_type_and_params(self):
        result = CalibrationResult(
            model_type="linear", params=(2.0, 50.0),
            inlier_count=6, residual=0.5, is_plausible=True,
        )
        assert result.model_type == "linear"
        assert result.params == (2.0, 50.0)
        assert result.is_plausible is True

    def test_log_result_with_two_params(self):
        result = CalibrationResult(
            model_type="log", params=(200.0, 50.0),
            inlier_count=7, residual=0.3, is_plausible=True,
        )
        assert result.model_type == "log"
        assert len(result.params) == 2


# ---------------------------------------------------------------------------
# fit_axis_multi_hypothesis
# ---------------------------------------------------------------------------

class TestFitAxisMultiHypothesis:
    def test_linear_wins_on_linear_data(self):
        ticks = _linear_ticks(n=10)
        result = fit_axis_multi_hypothesis(ticks)
        assert result is not None
        assert result.model_type == "linear"
        assert result.is_plausible is True

    def test_log_wins_on_log_data(self):
        ticks = _log_ticks(n=6)
        result = fit_axis_multi_hypothesis(ticks)
        assert result is not None
        assert result.model_type == "log"
        assert result.is_plausible is True

    def test_linear_inlier_count_on_clean_data(self):
        ticks = _linear_ticks(n=10)
        result = fit_axis_multi_hypothesis(ticks)
        assert result is not None
        assert result.inlier_count >= 8

    def test_returns_none_on_insufficient_ticks(self):
        result = fit_axis_multi_hypothesis([(100.0, 1.0)])
        assert result is None

    def test_returns_none_on_empty(self):
        result = fit_axis_multi_hypothesis([])
        assert result is None

    def test_log_with_ocr_noise_selects_log(self):
        """Log data with one OCR outlier should still select log."""
        ticks = _log_ticks(n=6)
        # Inject an outlier: OCR read 102 instead of 10^2
        corrupted = list(ticks)
        corrupted[2] = (corrupted[2][0], 102.0)  # wrong value
        result = fit_axis_multi_hypothesis(corrupted)
        assert result is not None
        assert result.model_type == "log"

    def test_linear_residual_low_on_clean_linear(self):
        ticks = _linear_ticks(n=10)
        result = fit_axis_multi_hypothesis(ticks)
        assert result is not None
        assert result.residual < 5.0

    def test_log_residual_low_on_clean_log(self):
        ticks = _log_ticks(n=6)
        result = fit_axis_multi_hypothesis(ticks)
        assert result is not None
        assert result.residual < 10.0

    def test_rejects_extreme_linear_slope(self):
        """A slope of 1e8 is physically impossible for a chart axis."""
        ticks = [(float(i), float(i * 1e8)) for i in range(5)]
        result = fit_axis_multi_hypothesis(ticks)
        assert result is None or not result.is_plausible

    def test_log_with_all_negative_values_falls_back(self):
        """Log fit requires positive values; all-negative should not crash."""
        ticks = [(float(i * 50), -(10.0 ** i)) for i in range(4)]
        # Should return linear or None, not crash
        result = fit_axis_multi_hypothesis(ticks)
        # Linear may or may not be plausible; just ensure no exception


# ---------------------------------------------------------------------------
# is_calibration_plausible
# ---------------------------------------------------------------------------

class TestIsCalibrationPlausible:
    def test_normal_linear_is_plausible(self):
        assert is_calibration_plausible("linear", (2.0, 50.0), 8) is True

    def test_extreme_slope_linear_is_implausible(self):
        assert is_calibration_plausible("linear", (1e8, 50.0), 8) is False

    def test_negative_slope_linear_implausible(self):
        assert is_calibration_plausible("linear", (-1e6, 50.0), 8) is False

    def test_normal_log_is_plausible(self):
        assert is_calibration_plausible("log", (200.0, 50.0), 6) is True

    def test_log_with_zero_slope_implausible(self):
        assert is_calibration_plausible("log", (0.0, 50.0), 6) is False

    def test_too_few_inliers_implausible(self):
        assert is_calibration_plausible("linear", (2.0, 50.0), 1) is False

    def test_polynomial_small_curvature_plausible(self):
        assert is_calibration_plausible("polynomial", (0.001, 2.0, 50.0), 6) is True

    def test_polynomial_large_curvature_implausible(self):
        assert is_calibration_plausible("polynomial", (10.0, 2.0, 50.0), 6) is False


# ---------------------------------------------------------------------------
# infer_log_values_from_spacing
# ---------------------------------------------------------------------------

class TestInferLogValuesFromSpacing:
    def test_detects_log_pattern_from_pixels(self):
        """Given pixel positions matching log tick spacing, infer values."""
        pixels = _log_ticks_pixel_only(n=5, a=200.0, b=50.0)
        result = infer_log_values_from_spacing(pixels)
        assert result is not None
        assert len(result) == len(pixels)
        # Inferred values should span multiple orders of magnitude
        values = [v for _, v in result]
        assert max(values) / max(min(values), 1e-9) > 100

    def test_returns_none_for_linear_spacing(self):
        """Linear spacing should not be misidentified as log."""
        pixels = [50.0, 100.0, 150.0, 200.0, 250.0]
        result = infer_log_values_from_spacing(pixels)
        assert result is None

    def test_returns_none_for_too_few_ticks(self):
        pixels = [50.0, 100.0]
        result = infer_log_values_from_spacing(pixels)
        assert result is None

    def test_handles_two_decades(self):
        """Two decades (10^0, 10^1, 10^2) should be inferrable."""
        # pixel = 200 * log10(value) + 50
        pixels = [50.0, 250.0, 450.0]  # 10^0=1, 10^1=10, 10^2=100
        result = infer_log_values_from_spacing(pixels)
        assert result is not None
        assert len(result) == 3

    def test_tolerance_for_slight_noise(self):
        """Log spacing with small noise should still be detected."""
        pixels = _log_ticks_pixel_only(n=5, a=200.0, b=50.0)
        # Add ±2 pixel noise
        rng = np.random.default_rng(42)
        noisy = [p + rng.uniform(-2, 2) for p in pixels]
        result = infer_log_values_from_spacing(noisy)
        assert result is not None


# ---------------------------------------------------------------------------
# grade_tick_quality
# ---------------------------------------------------------------------------

class TestGradeTickQuality:
    def test_high_score_for_clear_tick(self):
        """Tick with high OCR confidence and regular position."""
        quality = grade_tick_quality(
            ocr_confidence=95.0,
            crop_height=20,
            median_crop_height=20.0,
            position_deviation=0.5,
        )
        assert quality > 0.7

    def test_low_score_for_garbage_ocr(self):
        """Tick with low OCR confidence."""
        quality = grade_tick_quality(
            ocr_confidence=10.0,
            crop_height=20,
            median_crop_height=20.0,
            position_deviation=0.5,
        )
        assert quality < 0.4

    def test_low_score_for_irregular_position(self):
        """Tick far from aligned position."""
        quality = grade_tick_quality(
            ocr_confidence=80.0,
            crop_height=20,
            median_crop_height=20.0,
            position_deviation=30.0,
        )
        assert quality < 0.6

    def test_medium_score_mixed(self):
        """Reasonable OCR but slightly irregular."""
        quality = grade_tick_quality(
            ocr_confidence=70.0,
            crop_height=18,
            median_crop_height=20.0,
            position_deviation=3.0,
        )
        assert 0.3 < quality < 0.8

    def test_defaults_to_medium(self):
        """No parameters → medium quality."""
        quality = grade_tick_quality()
        assert 0.3 <= quality <= 0.7
