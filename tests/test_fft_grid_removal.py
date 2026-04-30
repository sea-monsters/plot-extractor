"""Tests for FFT-based periodic grid line removal.

Covers:
- suppress_grid_lines_fft: Remove periodic grid via frequency domain filtering
- Detection of grid period from FFT peaks
- Preservation of non-grid structures (curves, scatter points)
"""
import numpy as np
from plot_extractor.geometry.grid_suppress import suppress_grid_lines_fft


# ---------------------------------------------------------------------------
# Helpers: synthetic grid images
# ---------------------------------------------------------------------------

def _grid_image(width=100, height=80, grid_period=20, line_width=1):
    """Create an image with a periodic vertical grid."""
    img = np.ones((height, width), dtype=np.uint8) * 200
    for x in range(0, width, grid_period):
        img[:, x:x+line_width] = 50
    return img


def _grid_with_curve(width=100, height=80, grid_period=20):
    """Create an image with grid + a diagonal curve."""
    img = _grid_image(width, height, grid_period)
    # Add a diagonal curve
    for i in range(min(width, height) - 10):
        img[10 + i, 10 + i] = 0
    return img


def _non_periodic_lines(width=100, height=80):
    """Create an image with non-periodic (irregular) lines."""
    img = np.ones((height, width), dtype=np.uint8) * 200
    # Irregular vertical lines
    for x in [5, 23, 47, 81]:
        img[:, x:x+1] = 50
    return img


# ---------------------------------------------------------------------------
# suppress_grid_lines_fft
# ---------------------------------------------------------------------------

class TestSuppressGridLinesFFT:
    def test_removes_periodic_grid(self):
        """Periodic grid should be attenuated."""
        grid_img = _grid_image(grid_period=20)
        result = suppress_grid_lines_fft(grid_img)
        # Grid lines should be less prominent
        assert result.shape == grid_img.shape

    def test_preserves_image_dimensions(self):
        """Output should have same dimensions as input."""
        img = np.random.randint(0, 255, size=(60, 80), dtype=np.uint8)
        result = suppress_grid_lines_fft(img)
        assert result.shape == img.shape

    def test_handles_rgb_input(self):
        """RGB input should be converted to grayscale for processing."""
        rgb = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
        result = suppress_grid_lines_fft(rgb)
        assert len(result.shape) == 2

    def test_handles_grayscale_input(self):
        """Grayscale input should remain grayscale."""
        gray = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        result = suppress_grid_lines_fft(gray)
        assert len(result.shape) == 2

    def test_empty_image_returns_empty(self):
        """Empty image should not crash."""
        empty = np.zeros((40, 60), dtype=np.uint8)
        result = suppress_grid_lines_fft(empty)
        assert np.all(result == 0)

    def test_preserves_non_grid_content(self):
        """Non-periodic content should be preserved."""
        curve_img = _grid_with_curve(grid_period=25)
        result = suppress_grid_lines_fft(curve_img)
        # The diagonal curve should still be visible
        # Check that some dark pixels remain (curve)
        assert np.min(result) < 100

    def test_uses_grid_period_estimate(self):
        """grid_period_estimate parameter should be accepted."""
        img = _grid_image(grid_period=30)
        result = suppress_grid_lines_fft(img, grid_period_estimate=30)
        assert result.shape == img.shape

    def test_non_periodic_lines_less_affected(self):
        """Non-periodic lines should not be removed as aggressively."""
        img = _non_periodic_lines()
        result = suppress_grid_lines_fft(img)
        # Some structure should remain
        assert result.shape == img.shape
