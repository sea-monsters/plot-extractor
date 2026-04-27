"""Image loading and preprocessing."""
import cv2
import numpy as np
from pathlib import Path


def load_image(path, target_dpi=100):
    """Load image from path and preprocess."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess(image, denoise=True):
    """Apply denoising and enhancement."""
    if denoise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return image


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate image around its center with white padding.

    Uses cv2.warpAffine with BORDER_CONSTANT fill so that rotated areas
    are white (background), avoiding false foreground artifacts.
    """
    if abs(angle_degrees) < 0.05:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Compute output size to avoid clipping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust translation to keep image centered
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    if len(image.shape) == 3:
        border_value = (255, 255, 255)
    else:
        border_value = 255

    return cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
