"""Lightweight chart type guessing from image features (soft labels)."""
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from plot_extractor.core.axis_detector import detect_all_axes


@dataclass
class ImageFeatures:
    """Low-cost image features for chart type estimation."""

    # Geometric
    hough_horiz: int = 0
    hough_vert: int = 0
    hough_diag: int = 0
    edge_density: float = 0.0
    aspect_ratio: float = 1.0

    # Color
    hue_peak_count: int = 0
    saturation_mean: float = 0.0
    color_dominance: float = 0.0

    # Texture / noise
    laplacian_variance: float = 0.0
    extreme_pixel_ratio: float = 0.0
    fft_high_freq_ratio: float = 1.0
    block_variance_ratio: float = 0.0

    # Structural
    fg_col_density: float = 0.0
    cc_count: int = 0
    cc_area_mean: float = 0.0
    cc_area_variance: float = 0.0

    # Axis (coarse detection)
    axis_count: int = 0
    tick_regularity: float = 0.0
    has_dual_y: bool = False
    rotation_estimate: float = 0.0


def _extract_geometric_features(image_gray: np.ndarray) -> Dict[str, float]:
    """Extract Hough line counts and edge density."""
    h, w = image_gray.shape
    edges = cv2.Canny(image_gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / (h * w)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=int(min(h, w) * 0.15), maxLineGap=10,
    )
    horiz = vert = diag = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= 10 or angle >= 170:
                horiz += 1
            elif 80 <= angle <= 100:
                vert += 1
            else:
                diag += 1

    return {
        "hough_horiz": horiz,
        "hough_vert": vert,
        "hough_diag": diag,
        "edge_density": float(edge_density),
        "aspect_ratio": w / max(h, 1),
    }


def _extract_color_features(image_rgb: np.ndarray) -> Dict[str, float]:
    """Extract hue peaks, saturation, and color dominance."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hues = hsv[:, :, 0].flatten()
    sats = hsv[:, :, 1].flatten()

    # Ignore very low saturation (grays)
    color_mask = sats > 35
    if np.sum(color_mask) < 100:
        return {"hue_peak_count": 0, "saturation_mean": 0.0, "color_dominance": 0.0}

    color_hues = hues[color_mask]
    hist = np.bincount(color_hues.astype(int), minlength=180)
    smooth = np.convolve(hist, np.ones(5) / 5, mode="same")

    peaks = []
    for i in range(2, 178):
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]:
            if smooth[i] > len(color_hues) * 0.01:
                is_new = True
                for p in peaks:
                    if min(abs(i - p), 180 - abs(i - p)) < 20:
                        is_new = False
                        break
                if is_new:
                    peaks.append(i)

    sat_mean = float(np.mean(sats[color_mask]))
    dominance = float(np.max(hist) / (np.sum(hist) + 1e-6))

    return {
        "hue_peak_count": len(peaks),
        "saturation_mean": sat_mean,
        "color_dominance": dominance,
    }


def _extract_texture_features(image_gray: np.ndarray) -> Dict[str, float]:
    """Extract Laplacian variance, FFT high-freq ratio, block variance."""
    h, w = image_gray.shape

    # Salt & pepper indicator
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    lap_var = float(np.mean(np.abs(lap)))
    extreme_ratio = float(np.sum((image_gray == 0) | (image_gray == 255)) / image_gray.size)

    # JPEG block artifact indicator
    bh, bw = (h // 8) * 8, (w // 8) * 8
    if bh > 0 and bw > 0:
        blocks = image_gray[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8)
        block_vars = np.var(blocks, axis=(1, 3))
        jpeg_score = float(np.mean(block_vars) / (np.var(image_gray) + 1e-6))
    else:
        jpeg_score = 0.0

    # Blur indicator
    fft = np.fft.fft2(image_gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - h // 10)
    y1 = min(h, cy + h // 10)
    x0 = max(0, cx - w // 10)
    x1 = min(w, cx + w // 10)
    high_freq = np.abs(fft_shift[y0:y1, x0:x1])
    total_energy = np.mean(np.abs(fft)) + 1e-6
    blur_score = float(np.mean(high_freq) / total_energy)

    return {
        "laplacian_variance": lap_var,
        "extreme_pixel_ratio": extreme_ratio,
        "fft_high_freq_ratio": blur_score,
        "block_variance_ratio": jpeg_score,
    }


def _extract_structural_features(image_gray: np.ndarray) -> Dict[str, float]:
    """Extract foreground mask properties and connected component stats."""
    h, w = image_gray.shape

    # Use background detection consistent with main pipeline
    from plot_extractor.utils.image_utils import detect_background_color, make_foreground_mask
    # Convert gray to RGB for util compatibility
    gray_rgb = np.stack([image_gray, image_gray, image_gray], axis=2)
    bg = detect_background_color(gray_rgb)
    mask = make_foreground_mask(gray_rgb, bg, threshold=30)

    fg_cols = int(np.sum(np.any(mask > 0, axis=0)))
    col_density = fg_cols / max(w, 1)

    # Connected components with area filtering
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 5:  # Filter tiny noise specks
            areas.append(area)

    if areas:
        cc_count = len(areas)
        area_mean = float(np.mean(areas))
        area_var = float(np.var(areas) / (area_mean + 1e-6) ** 2)
    else:
        cc_count = 0
        area_mean = 0.0
        area_var = 0.0

    return {
        "fg_col_density": col_density,
        "cc_count": cc_count,
        "cc_area_mean": area_mean,
        "cc_area_variance": area_var,
    }


def _extract_axis_features(image_gray: np.ndarray) -> Dict[str, float]:
    """Coarse axis detection for feature extraction."""
    axes = detect_all_axes(image_gray)
    axis_count = len(axes)

    has_dual_y = False
    tick_reg = 0.0
    if axes:
        y_axes = [a for a in axes if a.direction == "y"]
        if len(y_axes) >= 2:
            sides = {a.side for a in y_axes}
            has_dual_y = "left" in sides and "right" in sides

        # Tick regularity: std of spacing normalized by mean spacing
        all_spacings = []
        for ax in axes:
            ticks = sorted([t[0] for t in (ax.ticks or [])])
            if len(ticks) >= 3:
                spacings = np.diff(ticks)
                all_spacings.extend(spacings)
        if all_spacings:
            tick_reg = float(np.std(all_spacings) / (np.mean(all_spacings) + 1e-6))

    return {
        "axis_count": axis_count,
        "tick_regularity": tick_reg,
        "has_dual_y": has_dual_y,
    }


def _estimate_rotation(image_gray: np.ndarray) -> float:
    """Fast rotation estimate using Hough lines (reuse axis_detector logic)."""
    from plot_extractor.core.axis_detector import estimate_rotation_angle
    return estimate_rotation_angle(image_gray)


def extract_all_features(
    image_rgb: np.ndarray,
    image_gray: Optional[np.ndarray] = None,
) -> ImageFeatures:
    """Extract all lightweight features from an image."""
    if image_gray is None:
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    geo = _extract_geometric_features(image_gray)
    col = _extract_color_features(image_rgb)
    tex = _extract_texture_features(image_gray)
    struct = _extract_structural_features(image_gray)
    axis = _extract_axis_features(image_gray)
    rot = _estimate_rotation(image_gray)

    return ImageFeatures(
        hough_horiz=geo["hough_horiz"],
        hough_vert=geo["hough_vert"],
        hough_diag=geo["hough_diag"],
        edge_density=geo["edge_density"],
        aspect_ratio=geo["aspect_ratio"],
        hue_peak_count=col["hue_peak_count"],
        saturation_mean=col["saturation_mean"],
        color_dominance=col["color_dominance"],
        laplacian_variance=tex["laplacian_variance"],
        extreme_pixel_ratio=tex["extreme_pixel_ratio"],
        fft_high_freq_ratio=tex["fft_high_freq_ratio"],
        block_variance_ratio=tex["block_variance_ratio"],
        fg_col_density=struct["fg_col_density"],
        cc_count=struct["cc_count"],
        cc_area_mean=struct["cc_area_mean"],
        cc_area_variance=struct["cc_area_variance"],
        axis_count=axis["axis_count"],
        tick_regularity=axis["tick_regularity"],
        has_dual_y=axis["has_dual_y"],
        rotation_estimate=rot,
    )


from plot_extractor.utils.math_utils import sigmoid as _sigmoid


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(scores.values()), dtype=float)
    exps = np.exp(vals - np.max(vals))
    probs = exps / np.sum(exps)
    return {k: float(v) for k, v in zip(scores.keys(), probs)}


CHART_TYPES = [
    "simple_linear", "log_y", "log_x", "loglog", "scatter",
    "multi_series", "dense", "dual_y", "inverted_y", "no_grid",
]


def guess_chart_type(features: ImageFeatures) -> Dict[str, float]:
    """Estimate chart-type probabilities using rule-based scoring.

    Uses lightweight heuristics derived from empirical analysis of
    v1-v4 datasets.  Returns softmax probabilities over 10 types.
    """
    scores: Dict[str, float] = {}

    # --- Multi series ---
    # DOMINANT signal: hue_peak_count >= 2.
    scores["multi_series"] = (
        5.0 * max(0, features.hue_peak_count - 1)
        - 0.5 * features.fg_col_density
    )

    # --- Dense ---
    # Very low saturation (< 130) is the dominant signal.
    scores["dense"] = (
        3.0 * (1 if features.saturation_mean < 130 else 0)
        + 0.5 * features.fg_col_density
        - 0.003 * features.cc_count
        - 0.002 * features.cc_area_mean
    )

    # --- Scatter ---
    # High saturation (> 220) + moderate cc_mean (> 35).
    scores["scatter"] = (
        2.0 * (1 if features.saturation_mean > 220 else 0)
        + 0.005 * features.cc_area_mean
        - 0.5 * features.fg_col_density
    )

    # --- No grid ---
    # Very low cc_count (< 60) + very high cc_mean (> 100).
    scores["no_grid"] = (
        2.0 * (1 if features.cc_count < 60 and features.cc_area_mean > 100 else 0)
        - 0.005 * features.cc_count
        + 0.1 * (1 if features.axis_count >= 2 else 0)
    )

    # --- Log types ---
    # Distinguish from simple linear by having many small CCs (grid lines).
    # Hough line counts are NOT used because all chart types have axes.
    log_signal = (
        1.5 * (1 if features.cc_count > 200 and features.cc_area_mean < 40 else 0)  # stronger log grid signal
        - 0.3 * features.fg_col_density
    )
    scores["log_y"] = log_signal + 0.1 * (1 if features.hough_horiz >= features.hough_vert else 0)
    scores["log_x"] = log_signal + 0.1 * (1 if features.hough_vert > features.hough_horiz else 0)
    scores["loglog"] = log_signal + 0.05

    # --- Dual Y ---
    # Requires multiple hue peaks (two series on separate Y axes).
    scores["dual_y"] = (
        0.8 * max(0, features.hue_peak_count - 1)
        + 0.1 * features.axis_count
        - 0.002 * features.cc_count
        - 1.0 * (1 if features.hue_peak_count <= 1 else 0)
    )

    # --- Inverted Y ---
    # Similar to simple linear; weak distinctive signal.
    scores["inverted_y"] = (
        -0.2 * features.fg_col_density
        + 0.1 * (1 if features.axis_count >= 2 else 0)
        - 0.0005 * features.cc_count
    )

    # --- Simple linear ---
    # Default type: single hue, moderate density, no log grid, has axes.
    # High baseline for single-hue charts; log grid penalty is softer.
    scores["simple_linear"] = (
        2.0 * (1 if features.hue_peak_count <= 1 else 0)
        + 0.3 * (1 if features.axis_count >= 2 else 0)
        - 0.3 * features.fg_col_density
        - 0.002 * max(0, features.cc_count - 100)  # only penalize above 100 CCs
        - 0.5 * (1 if features.cc_count > 200 and features.cc_area_mean < 35 else 0)  # softer log grid
    )

    return _softmax(scores)
