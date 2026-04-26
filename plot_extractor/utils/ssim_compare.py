"""Pure NumPy SSIM implementation (no scikit-image)."""
import numpy as np
import cv2


def _gaussian_window(size, sigma):
    """Create a 2D Gaussian kernel."""
    x = np.arange(size) - size // 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = np.outer(g, g)
    return window


def _convolve2d(img, window):
    """Convolve image with window using OpenCV filter2D."""
    return cv2.filter2D(img, -1, window, borderType=cv2.BORDER_REFLECT)


def ssim(img1, img2, window_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """Compute SSIM between two grayscale images (0-255 or 0-1)."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    # Normalize to 0-1 if needed
    if img1.max() > 1.0:
        img1 = img1.astype(np.float64) / 255.0
    else:
        img1 = img1.astype(np.float64)
    if img2.max() > 1.0:
        img2 = img2.astype(np.float64) / 255.0
    else:
        img2 = img2.astype(np.float64)

    window = _gaussian_window(window_size, sigma)

    mu1 = _convolve2d(img1, window)
    mu2 = _convolve2d(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = _convolve2d(img1 ** 2, window) - mu1_sq
    sigma2_sq = _convolve2d(img2 ** 2, window) - mu2_sq
    sigma12 = _convolve2d(img1 * img2, window) - mu12

    data_range = 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    # Exclude border artifacts
    pad = window_size // 2
    if pad > 0:
        ssim_map = ssim_map[pad:-pad, pad:-pad]

    return float(ssim_map.mean())


def compare_images(path1, path2, crop_box=None):
    """Compute SSIM between two image files.

    Optional crop_box=(x1,y1,x2,y2) to compare only plot area.
    """
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))
    if img1 is None or img2 is None:
        return 0.0

    # Resize to same size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2 or w1 != w2:
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w1, x2), min(h1, y2)
        if x2 > x1 and y2 > y1:
            gray1 = gray1[y1:y2, x1:x2]
            gray2 = gray2[y1:y2, x1:x2]

    return ssim(gray1, gray2)
