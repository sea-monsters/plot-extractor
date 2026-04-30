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


def preprocess(image, denoise=True, policy=None):
    """Apply denoising and enhancement with noise-type awareness.

    If *policy* is provided, its ``noise_strategy`` overrides the
    auto-detected noise type.  This lets the policy ensemble suppress
    or activate specific filters based on chart-type probabilities.
    """
    if not denoise:
        return image

    if policy is not None:
        noise_type = policy.noise_strategy
        median_ksize = policy.median_ksize
        bilateral_d = policy.bilateral_d
        unsharp_amount = policy.unsharp_amount
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        noise_type = _detect_noise_type(gray)
        median_ksize = 3
        bilateral_d = 9
        unsharp_amount = 1.5 if noise_type == "blur" else 1.0 if noise_type == "rotation_noise" else 0.0

    if noise_type == "salt_pepper":
        image = cv2.medianBlur(image, median_ksize)
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    elif noise_type == "jpeg":
        image = cv2.bilateralFilter(image, d=bilateral_d, sigmaColor=75, sigmaSpace=75)
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    elif noise_type == "blur":
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        if unsharp_amount > 0:
            image = _unsharp_mask(image, amount=unsharp_amount)
    elif noise_type == "rotation_noise":
        image = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
        if unsharp_amount > 0:
            image = _unsharp_mask(image, amount=unsharp_amount)
    else:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return image


def _detect_noise_type(gray):
    """Classify dominant noise type in a grayscale image."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return "clean"

    # Salt & pepper: high local variance from extreme values
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sp_score = np.mean(np.abs(lap))
    extreme_ratio = np.sum((gray == 0) | (gray == 255)) / gray.size

    # JPEG: DCT block artifact detection via 8x8 block variance
    bh, bw = (h // 8) * 8, (w // 8) * 8
    if bh > 0 and bw > 0:
        blocks = gray[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8)
        block_vars = np.var(blocks, axis=(1, 3))
        jpeg_score = np.mean(block_vars) / (np.var(gray) + 1e-6)
    else:
        jpeg_score = 0.0

    # Blur: low high-frequency energy relative to total spectrum
    fft = np.fft.fft2(gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    cy, cx = h // 2, w // 2
    # High-freq ring (outer 20% of spectrum)
    y_start = max(0, cy - h // 10)
    y_end = min(h, cy + h // 10)
    x_start = max(0, cx - w // 10)
    x_end = min(w, cx + w // 10)
    high_freq = np.abs(fft_shift[y_start:y_end, x_start:x_end])
    total_energy = np.mean(np.abs(fft)) + 1e-6
    blur_score = np.mean(high_freq) / total_energy

    if sp_score > 80 and extreme_ratio > 0.005:
        return "salt_pepper"
    if jpeg_score > 2.0:
        return "jpeg"
    if blur_score < 0.08:
        return "blur"
    return "clean"


def _unsharp_mask(image, amount=1.5, sigma=1.0):
    """Unsharp mask to restore edge clarity after denoising."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


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


def apply_noise_aware_preprocessing(image: np.ndarray, policy) -> np.ndarray:
    """Conditionally apply NLM denoising based on noise detection.

    Detects noise level via Laplacian variance and applies fastNlMeansDenoising
    when noise exceeds the policy's noise_threshold.

    Args:
        image: Input image (RGB or grayscale).
        policy: ExtractionPolicy with noise_threshold attribute.

    Returns:
        Grayscale image with optional NLM denoising applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    noise_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    threshold = getattr(policy, "noise_threshold", 100.0)

    if noise_score > threshold:
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return denoised

    return gray
