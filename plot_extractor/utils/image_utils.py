"""Image utility functions."""
import cv2
import numpy as np


def detect_background_color(image, border=20):
    """Detect background color by sampling corners and edges."""
    h, w = image.shape[:2]
    samples = []
    # corners
    samples.append(image[0:border, 0:border].reshape(-1, 3))
    samples.append(image[0:border, w - border:w].reshape(-1, 3))
    samples.append(image[h - border:h, 0:border].reshape(-1, 3))
    samples.append(image[h - border:h, w - border:w].reshape(-1, 3))
    # top/bottom edges
    samples.append(image[0:border, border:w - border].reshape(-1, 3))
    samples.append(image[h - border:h, border:w - border].reshape(-1, 3))
    all_samples = np.vstack(samples)
    # Use k-means with k=2, pick the more frequent as background
    all_samples = np.float32(all_samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(all_samples, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    bg_idx = int(np.argmax(counts))
    bg_color = centers[bg_idx]
    return np.clip(bg_color, 0, 255).astype(np.uint8)


def make_foreground_mask(image, bg_color, threshold=30):
    """Create binary mask where pixels differ from background."""
    bg_img = np.full_like(image, bg_color)
    diff = cv2.absdiff(image, bg_img)
    dist = np.linalg.norm(diff, axis=2)
    mask = (dist > threshold).astype(np.uint8) * 255
    return mask
