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
