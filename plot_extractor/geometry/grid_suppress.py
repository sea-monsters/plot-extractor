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
