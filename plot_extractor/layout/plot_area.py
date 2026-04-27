"""Plot area detection.

Estimates the actual plotting region within a panel,
which defines the ROI for axis detection, text OCR, and data extraction.
"""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2

from plot_extractor.layout.panel_split import Panel


@dataclass
class PlotArea:
    """Detected plot area within a panel."""
    rect: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    confidence: float  # 0-100


def estimate_plot_area(panel: Panel, axis_candidates: Optional[List] = None) -> PlotArea:
    """
    Estimate plot area within a panel.

    Methods (priority order):
    1. Hough lines → orthogonal axis intersection
    2. Dense foreground region detection
    3. Panel border (fallback)

    Returns:
        PlotArea with confidence score
    """
    image = panel.image
    h, w = image.shape[:2]

    # Method 1: Axis line intersection
    if axis_candidates:
        plot_area = _estimate_from_axes(image, axis_candidates)
        if plot_area:
            return plot_area

    # Method 2: Dense foreground region
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    plot_area_dense = _estimate_from_density(gray)

    if plot_area_dense and plot_area_dense.confidence > 60:
        return plot_area_dense

    # Method 3: Border fallback
    border_margin = 50
    fallback_rect = (border_margin, border_margin, w - border_margin, h - border_margin)

    return PlotArea(rect=fallback_rect, confidence=30.0)


def _estimate_from_axes(image: np.ndarray, axis_candidates: List) -> Optional[PlotArea]:
    """Estimate plot area from axis line candidates."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) < 10:
        return None

    # Separate horizontal and vertical lines
    horizontal = []
    vertical = []
    for line in lines:
        x0, y0, x1, y1 = line[0]
        if abs(y1 - y0) < 5:  # Horizontal
            horizontal.append((x0, y0, x1, y1))
        elif abs(x1 - x0) < 5:  # Vertical
            vertical.append((x0, y0, x1, y1))

    if not horizontal or not vertical:
        return None

    # Find extreme positions
    h_y_values = [l[1] for l in horizontal] + [l[3] for l in horizontal]
    v_x_values = [l[0] for l in vertical] + [l[2] for l in vertical]

    # Plot area bounds
    top = min(h_y_values)
    bottom = max(h_y_values)
    left = min(v_x_values)
    right = max(v_x_values)

    # Validate: must be reasonable size
    h_img, w_img = image.shape[:2]
    if bottom - top < h_img * 0.3 or right - left < w_img * 0.3:
        return None

    confidence = 90.0  # High confidence for axis-based detection

    return PlotArea(rect=(int(left), int(top), int(right), int(bottom)), confidence=confidence)


def _estimate_from_density(gray: np.ndarray) -> Optional[PlotArea]:
    """Estimate plot area from foreground density."""
    h, w = gray.shape

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Morphological closing to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find largest contour (likely plot area)
    largest = max(contours, key=cv2.contourArea)

    # Bounding rectangle
    x, y, w_rect, h_rect = cv2.boundingRect(largest)

    # Validate size
    if w_rect < w * 0.3 or h_rect < h * 0.3:
        return None

    confidence = 70.0  # Medium confidence

    return PlotArea(rect=(x, y, x + w_rect, y + h_rect), confidence=confidence)


def score_plot_area(plot_area: PlotArea, panel: Panel) -> float:
    """
    Score plot area quality.

    Factors:
    - Contains most foreground pixels
    - Has tick marks nearby
    - Not too close to image edges
    """
    image = panel.image
    x0, y0, x1, y1 = plot_area.rect
    h, w = image.shape[:2]

    score = 0.0

    # Size合理性
    plot_width = x1 - x0
    plot_height = y1 - y0
    if plot_width >= w * 0.5 and plot_height >= h * 0.5:
        score += 30

    # 距离边缘
    edge_margin = 20
    if x0 > edge_margin and y0 > edge_margin and x1 < w - edge_margin and y1 < h - edge_margin:
        score += 20

    # 前景像素覆盖（TODO）

    return score
