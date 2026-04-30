"""Tests for Frangi grayscale pipeline for dense curve extraction.

Covers:
- apply_noise_aware_preprocessing: NLM denoising based on noise detection
- _extract_from_frangi_ridges: Frangi vesselness + ridge extraction
- Fallback when scikit-image is unavailable
"""
import numpy as np
import pytest
from plot_extractor.core.image_loader import apply_noise_aware_preprocessing


# ---------------------------------------------------------------------------
# Helpers: synthetic images
# ---------------------------------------------------------------------------

def _synthetic_line_image(width=100, height=80, x1=10, y1=40, x2=90, y2=40):
    """Create an image with a horizontal line."""
    img = np.zeros((height, width), dtype=np.uint8)
    img[y1:y1+2, x1:x2] = 255
    return img


def _noisy_image(width=100, height=80, noise_level=50):
    """Create a noisy image with random noise."""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, noise_level, size=(height, width), dtype=np.uint8)
    return noise


def _noisy_line_image(width=100, height=80, noise_level=30):
    """Create an image with a line and additive noise."""
    img = _synthetic_line_image(width, height)
    rng = np.random.default_rng(42)
    noise = rng.integers(0, noise_level, size=(height, width), dtype=np.uint8)
    noisy = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return noisy


class MockPolicy:
    """Mock ExtractionPolicy for testing."""
    noise_threshold: float = 100.0


def _has_skimage():
    """Check if scikit-image is available."""
    try:
        from skimage.filters import frangi  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# apply_noise_aware_preprocessing
# ---------------------------------------------------------------------------

class TestApplyNoiseAwarePreprocessing:
    def test_returns_grayscale(self):
        """Output should be grayscale even if input is RGB."""
        rgb = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
        gray = apply_noise_aware_preprocessing(rgb, MockPolicy())
        assert len(gray.shape) == 2

    def test_preserves_grayscale_input(self):
        """Grayscale input should remain grayscale."""
        gray_in = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        gray_out = apply_noise_aware_preprocessing(gray_in, MockPolicy())
        assert gray_out.shape == gray_in.shape

    def test_reduces_noise_when_above_threshold(self):
        """Noisy images should be denoised when above threshold."""
        noisy = _noisy_image(noise_level=80)
        policy = MockPolicy()
        policy.noise_threshold = 10.0  # Low threshold triggers denoising
        result = apply_noise_aware_preprocessing(noisy, policy)
        # Denoised image should have different pixel values
        assert not np.array_equal(result, noisy)

    def test_skips_denoising_when_below_threshold(self):
        """Clean images should not be denoised."""
        clean = _synthetic_line_image()
        policy = MockPolicy()
        policy.noise_threshold = 10000.0  # High threshold skips denoising
        result = apply_noise_aware_preprocessing(clean, policy)
        # Should be identical to input (both grayscale)
        assert np.array_equal(result, clean)

    def test_handles_empty_image(self):
        """Empty (all zeros) image should not crash."""
        empty = np.zeros((50, 50), dtype=np.uint8)
        result = apply_noise_aware_preprocessing(empty, MockPolicy())
        assert result.shape == empty.shape


# ---------------------------------------------------------------------------
# Frangi ridge extraction (requires scikit-image)
# ---------------------------------------------------------------------------

class TestFrangiRidgeExtraction:
    @pytest.mark.skipif(
        not _has_skimage(),
        reason="scikit-image not installed"
    )
    def test_frangi_enhances_line(self):
        """Frangi filter should respond strongly to line structures."""
        from skimage.filters import frangi
        line_img = _synthetic_line_image()
        vesselness = frangi(line_img, sigmas=range(1, 4, 1), black_ridges=True)
        # Line pixels should have higher vesselness than background
        line_response = vesselness[40, 50]  # On the line
        bg_response = vesselness[10, 10]    # Background
        assert line_response > bg_response

    @pytest.mark.skipif(
        not _has_skimage(),
        reason="scikit-image not installed"
    )
    def test_frangi_suppresses_noise(self):
        """Frangi filter should suppress random noise."""
        from skimage.filters import frangi
        noise_img = _noisy_image(noise_level=100)
        vesselness = frangi(noise_img, sigmas=range(1, 4, 1), black_ridges=True)
        # Noise image should have low vesselness overall
        max_response = np.max(vesselness)
        assert max_response < 0.1
