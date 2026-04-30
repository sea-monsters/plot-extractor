"""Grid line suppression via directional morphology.

Removes horizontal/vertical grid lines while preserving data curves and scatter points.
"""
import numpy as np
import cv2


def suppress_grid_lines(image: np.ndarray) -> np.ndarray:
    """
    Suppress grid lines using directional morphology.

    Strategy:
    - Horizontal lines: wide horizontal kernel (MORPH_OPEN)
    - Vertical lines: tall vertical kernel (MORPH_OPEN)
    - Subtract detected grid from original

    Args:
        image: RGB or grayscale image

    Returns:
        Grid-suppressed image (same color space as input)
    """
    is_rgb = len(image.shape) == 3
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_rgb else image

    # Detect horizontal grid lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical grid lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)

    # Combine grid lines
    grid_mask = cv2.add(h_lines, v_lines)

    # Suppress grid from original
    suppressed = cv2.subtract(gray, grid_mask)

    if is_rgb:
        # Convert back to RGB
        result = cv2.cvtColor(suppressed, cv2.COLOR_GRAY2RGB)
        return result

    return suppressed


def suppress_grid_lines_fft(
    image: np.ndarray, grid_period_estimate: int = 50
) -> np.ndarray:
    """Remove periodic grid lines using 2D FFT.

    Detects and attenuates frequency peaks corresponding to periodic
    grid structures. Preserves non-periodic content like curves and
    scatter points.

    Steps:
    1. Compute 2D FFT of grayscale image
    2. Detect peaks at grid-spacing frequencies
    3. Mask those peaks (attenuate)
    4. Inverse FFT to reconstruct image without grid

    Args:
        image: RGB or grayscale image.
        grid_period_estimate: Approximate pixel distance between grid lines.

    Returns:
        Grayscale image with periodic grid lines attenuated.
    """
    is_rgb = len(image.shape) == 3
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_rgb else image

    if gray.size == 0 or gray.max() == 0:
        return gray

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    threshold = np.percentile(magnitude, 99)
    peaks = magnitude > threshold

    fshift[peaks] *= 0.1

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return img_back
