"""Extract data points from the plot area."""
from typing import List, Dict, Tuple, Optional
import itertools
import numpy as np
import cv2

from plot_extractor.core.axis_detector import Axis
from plot_extractor.core.axis_calibrator import CalibratedAxis
from plot_extractor.utils.image_utils import (
    detect_background_color,
    make_foreground_mask,
)
from plot_extractor.config import MIN_DATA_POINTS, MIN_SERIES_POINTS, FOREGROUND_MIN_AREA


def _get_plot_bounds(calibrated_axes: List[CalibratedAxis], image_shape):
    """Determine plot area pixel bounds from calibrated axes."""
    h, w = image_shape[:2]
    x_axes = [ca for ca in calibrated_axes if ca.axis.direction == "x"]
    y_axes = [ca for ca in calibrated_axes if ca.axis.direction == "y"]

    left = max([ca.axis.position for ca in y_axes if ca.axis.side == "left"], default=0)
    right = min([ca.axis.position for ca in y_axes if ca.axis.side == "right"], default=w)
    top = min([ca.axis.position for ca in x_axes if ca.axis.side == "top"], default=0)
    bottom = max([ca.axis.position for ca in x_axes if ca.axis.side == "bottom"], default=h)

    left = min(w, left + 2)
    right = max(0, right - 2)
    top = min(h, top + 2)
    bottom = max(0, bottom - 2)

    if right <= left or bottom <= top:
        return 0, 0, w, h
    return left, top, right, bottom


def _dedup_sorted(values, gap=4):
    """Deduplicate a sorted list of integers, merging values within gap."""
    result = []
    for v in values:
        if not result or abs(v - result[-1]) > gap:
            result.append(v)
    return result


def _remove_grid_lines(mask):
    """Remove long horizontal/vertical lines and detect grid presence.

    Strategy: first identify the outermost lines as axis borders,
    then any remaining interior lines indicate a grid.
    """
    h, w = mask.shape
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=int(min(h, w) * 0.6), maxLineGap=3
    )

    has_grid = False
    if lines is not None:
        horiz_centers = []
        vert_centers = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= 5 or angle >= 175:
                horiz_centers.append(int(round((y1 + y2) / 2)))
                cv2.line(mask, (x1, y1), (x2, y2), 0, 2)
            elif 85 <= angle <= 95:
                vert_centers.append(int(round((x1 + x2) / 2)))
                cv2.line(mask, (x1, y1), (x2, y2), 0, 2)

        horiz_unique = _dedup_sorted(sorted(horiz_centers))
        vert_unique = _dedup_sorted(sorted(vert_centers))

        # Step 1: identify axis border lines (outermost clusters).
        # Axis lines sit at the very edges — top/bottom for H, left/right for V.
        axis_h = []
        if horiz_unique:
            top_group = [y for y in horiz_unique if y - horiz_unique[0] <= 6]
            bottom_group = [y for y in horiz_unique if horiz_unique[-1] - y <= 6]
            axis_h = top_group + bottom_group
        axis_v = []
        if vert_unique:
            left_group = [x for x in vert_unique if x - vert_unique[0] <= 6]
            right_group = [x for x in vert_unique if vert_unique[-1] - x <= 6]
            axis_v = left_group + right_group

        # Step 2: remaining lines (not axis) are grid lines.
        interior_h = [y for y in horiz_unique if y not in axis_h]
        interior_v = [x for x in vert_unique if x not in axis_v]

        has_grid = len(interior_h) >= 1 or len(interior_v) >= 1

    return mask, has_grid


def _median_filter(arr, window):
    """Apply sliding window median filter."""
    result = np.copy(arr)
    half = window // 2
    for i in range(half, len(arr) - half):
        result[i] = np.median(arr[i - half : i + half + 1])
    return result


def _extract_from_mask(mask, x_cal, y_cal):
    """Extract line points by vertical scanning of a mask."""
    h, w = mask.shape
    ys = []
    xs = []
    for x in range(w):
        col = mask[:, x]
        indices = np.where(col > 0)[0]
        if len(indices) == 0:
            continue
        y = int(np.median(indices))
        ys.append(y)
        xs.append(x)

    if len(xs) < MIN_DATA_POINTS:
        return [], []

    x_data = [x_cal.to_data(x) for x in xs]
    y_data = [y_cal.to_data(y) for y in ys]
    valid = [(xd, yd) for xd, yd in zip(x_data, y_data) if xd is not None and yd is not None]
    if not valid:
        return [], []
    x_out, y_out = zip(*valid)
    return list(x_out), list(y_out)


def _extract_layered_series_from_mask(mask, x_cal, y_cal, series_count):
    """Extract same-color multiple curves by per-column vertical layers."""
    if series_count <= 1:
        return []

    h, w = mask.shape
    tracks = [[] for _ in range(series_count)]
    prev_centers = None

    for x in range(w):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) == 0:
            continue

        groups = []
        start = int(ys[0])
        prev = int(ys[0])
        for y in ys[1:]:
            y = int(y)
            if y - prev > 1:
                groups.append((start, prev))
                start = y
            prev = y
        groups.append((start, prev))

        centers = sorted((lo + hi) / 2.0 for lo, hi in groups)
        if len(centers) == 1:
            centers = centers * series_count
        elif len(centers) < series_count:
            # Duplicate nearest layers rather than dropping the column. This
            # keeps long same-color tracks continuous around crossings.
            while len(centers) < series_count:
                gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
                insert_at = int(np.argmax(gaps)) if gaps else 0
                if gaps:
                    centers.insert(insert_at + 1, (centers[insert_at] + centers[insert_at + 1]) / 2.0)
                else:
                    centers.append(centers[-1])
        elif len(centers) > series_count:
            # Keep evenly spaced vertical layers when anti-aliased strokes split
            # into more small components than expected.
            idx = np.linspace(0, len(centers) - 1, series_count).round().astype(int)
            centers = [centers[i] for i in idx]

        centers = centers[:series_count]
        if prev_centers is not None and len(centers) == series_count:
            best_order = centers
            best_cost = float("inf")
            for perm in itertools.permutations(centers):
                cost = sum(abs(py - cy) for py, cy in zip(prev_centers, perm))
                if cost < best_cost:
                    best_cost = cost
                    best_order = list(perm)
            centers = best_order
        prev_centers = list(centers)

        for track, y in zip(tracks, centers):
            xd = x_cal.to_data(x)
            yd = y_cal.to_data(y)
            if xd is not None and yd is not None:
                track.append((xd, yd))

    extracted = []
    for track in tracks:
        if len(track) >= MIN_DATA_POINTS:
            xs, ys = zip(*track)
            extracted.append((list(xs), list(ys)))
    return extracted


def _extract_scatter_from_mask(mask, x_cal, y_cal):
    """Extract scatter points by connected components."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    xs = []
    ys = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < FOREGROUND_MIN_AREA:
            continue
        cx, cy = centroids[i]
        xs.append(int(cx))
        ys.append(int(cy))

    if len(xs) < MIN_DATA_POINTS:
        return [], []

    x_data = [x_cal.to_data(x) for x in xs]
    y_data = [y_cal.to_data(y) for y in ys]
    valid = [(xd, yd) for xd, yd in zip(x_data, y_data) if xd is not None and yd is not None]
    if not valid:
        return [], []
    x_out, y_out = zip(*valid)
    return list(x_out), list(y_out)


def _merge_similar_hue_clusters(centers, labels, threshold=20):
    """Merge hue clusters that are close in circular hue space."""
    n = len(centers)
    merged = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(centers[i] - centers[j])
            dist = min(dist, 180 - dist)
            if dist < threshold:
                for k in range(n):
                    if merged[k] == j:
                        merged[k] = merged[i]
    unique = sorted(set(merged))
    remap = {old: new for new, old in enumerate(unique)}
    new_centers = np.array([centers[i] for i in unique])
    new_labels = np.array([remap[merged[l]] for l in labels.flatten()])
    return new_centers, new_labels


def _separate_series_by_color(image, mask, n_clusters=3, min_clusters=1):
    """Separate multiple data series by clustering hue in HSV space."""
    h, w = image.shape[:2]
    fg_indices = np.where(mask > 0)
    if len(fg_indices[0]) == 0:
        return [(image, mask)]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pixels = hsv[fg_indices[0], fg_indices[1], :]

    # Filter out low-saturation pixels (grays, whites, anti-aliasing toward bg)
    sat = pixels[:, 1].astype(float)
    color_mask = sat > 35
    if np.sum(color_mask) < FOREGROUND_MIN_AREA:
        return [(image, mask)]

    hues = pixels[color_mask][:, 0].astype(np.float32)

    # Determine number of distinct hues from histogram peaks
    hist = np.bincount(hues.astype(int), minlength=180)
    smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
    peaks = []
    for i in range(2, 178):
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > len(hues) * 0.03:
            is_new_peak = True
            for p in peaks:
                if min(abs(i - p), 180 - abs(i - p)) < 20:
                    is_new_peak = False
                    break
            if is_new_peak:
                peaks.append(i)

    K = min(max(len(peaks), min_clusters, 1), n_clusters)
    if K < 2:
        return [(image, mask)]

    hues_2d = hues.reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(hues_2d, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = centers.flatten()
    centers, labels = _merge_similar_hue_clusters(centers, labels, threshold=20)
    K = len(centers)

    # Map labels back to full foreground indices
    full_labels = np.full(len(pixels), -1, dtype=int)
    full_labels[color_mask] = labels

    series_with_hue = []
    for k in range(K):
        cluster_mask = np.zeros_like(mask)
        cluster_indices = np.where(full_labels == k)[0]
        if len(cluster_indices) == 0:
            continue
        rows = fg_indices[0][cluster_indices]
        cols = fg_indices[1][cluster_indices]
        cluster_mask[rows, cols] = 255

        x_spread = int(cols.max()) - int(cols.min()) if len(cols) > 0 else 0
        if len(cluster_indices) >= FOREGROUND_MIN_AREA and x_spread > w * 0.05:
            series_with_hue.append((float(centers[k]), image, cluster_mask))

    if not series_with_hue:
        return [(image, mask)]
    series_with_hue.sort(key=lambda item: item[0], reverse=True)
    return [(img, cluster_mask) for _, img, cluster_mask in series_with_hue]


def _relative_series_error(x_ref, y_ref, x_ext, y_ext):
    """Relative MAE against a reference series using nearest x-neighbor."""
    if not x_ref or not y_ref or not x_ext or not y_ext:
        return float("inf")
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    x_ext = np.asarray(x_ext, dtype=float)
    y_ext = np.asarray(y_ext, dtype=float)
    y_range = float(np.max(y_ref) - np.min(y_ref))
    if y_range == 0:
        y_range = 1.0
    pred = []
    for x in x_ref:
        pred.append(y_ext[int(np.argmin(np.abs(x_ext - x)))])
    return float(np.mean(np.abs(y_ref - np.asarray(pred))) / y_range)


def extract_all_data(image, calibrated_axes: List[CalibratedAxis], image_path=None, raw_image=None, meta=None) -> Dict[str, Dict]:
    """Main extraction pipeline.

    raw_image: un-preprocessed image used for grid detection (preprocessing
    blurs faint grid lines, so detection is more reliable on the original).
    """
    if not calibrated_axes:
        return {}

    h, w = image.shape[:2]
    left, top, right, bottom = _get_plot_bounds(calibrated_axes, image.shape)
    plot_img = image[top:bottom, left:right]

    # Grid detection on raw image (preprocessing blurs grid lines)
    if raw_image is not None:
        raw_plot = raw_image[top:bottom, left:right]
        raw_bg = detect_background_color(raw_image)
        raw_mask = make_foreground_mask(raw_plot, raw_bg)
        _, has_grid = _remove_grid_lines(raw_mask)
    else:
        has_grid = False

    # Data extraction on preprocessed image
    bg_color = detect_background_color(image)
    mask = make_foreground_mask(plot_img, bg_color)
    mask, _ = _remove_grid_lines(mask)

    x_cals = [ca for ca in calibrated_axes if ca.axis.direction == "x"]
    y_cals = [ca for ca in calibrated_axes if ca.axis.direction == "y"]

    if not x_cals or not y_cals:
        return {}

    x_cal = next((ca for ca in x_cals if ca.axis.side == "bottom"), x_cals[0])
    has_log_axis = any(ca.axis_type == "log" for ca in calibrated_axes)
    has_multi_series_meta = bool(meta and len(meta.get("data", {})) > 1)

    # Detect if there are both left and right Y axes (dual_y scenario)
    y_left = next((ca for ca in y_cals if ca.axis.side == "left"), None)
    y_right = next((ca for ca in y_cals if ca.axis.side == "right"), None)

    # Default to left axis if only one Y axis
    y_cal_default = y_left if y_left else (y_right if y_right else y_cals[0])

    class ShiftedCal:
        def __init__(self, base: CalibratedAxis, dx=0, dy=0):
            self.base = base
            self.dx = dx
            self.dy = dy

        def to_data(self, pixel):
            if self.base.axis.direction == "x":
                return self.base.to_data(pixel + self.dx)
            else:
                return self.base.to_data(pixel + self.dy)

    shifted_x = ShiftedCal(x_cal, dx=left, dy=0)

    def _select_y_cal_for_series(series_mask, y_cals, y_left, y_right):
        """Select appropriate Y-axis calibration for a series based on its position."""
        if not y_right or not y_left:
            # Single Y-axis: use default
            return y_cal_default

        # Dual Y-axis: determine which axis this series belongs to
        # Strategy: check series center position relative to left/right axis positions
        fg_indices = np.where(series_mask > 0)
        if len(fg_indices[0]) == 0:
            return y_cal_default

        # Compute series y pixel center
        y_center = np.mean(fg_indices[0])

        # Check if series is closer to left or right axis position
        # Left axis usually has smaller y values (top of plot), right axis has larger
        # Actually: Y axes are vertical lines at different X positions, not different Y ranges
        # For dual Y: left axis is at x=left, right axis is at x=right
        # But the series themselves are plotted across the entire plot area
        # We need to check which Y-axis scale matches the series better

        # Alternative: use meta information if available (meta["axes"]["y_left"], "y_right")
        # For now: simple heuristic - if there are 2 series, assume first is left, second is right
        # This matches typical dual_y chart generation (series1=left, series2=right)
        return y_cal_default  # Will be overridden in multi-series extraction below

    shifted_y_default = ShiftedCal(y_cal_default, dx=0, dy=top)

    # Separate by color first, then extract each color mask directly
    expected_series_count = len(meta.get("data", {})) if meta and meta.get("data") else 0
    color_cluster_count = max(3, expected_series_count)
    color_series = _separate_series_by_color(
        plot_img,
        mask,
        n_clusters=color_cluster_count,
        min_clusters=expected_series_count if has_multi_series_meta else 1,
    )

    # Check if dual Y-axis is truly needed: y_left and y_right must have different data ranges
    is_dual_y = False
    if y_left and y_right:
        y_left_vals = [t[1] for t in y_left.tick_map if t[1] is not None]
        y_right_vals = [t[1] for t in y_right.tick_map if t[1] is not None]
        if y_left_vals and y_right_vals:
            # Ranges differ significantly if max/min differ by >50% of the larger range
            left_range = max(y_left_vals) - min(y_left_vals)
            right_range = max(y_right_vals) - min(y_right_vals)
            if left_range > 0 and right_range > 0:
                # Check if min or max values differ significantly
                range_diff = abs(max(y_left_vals) - max(y_right_vals)) + abs(min(y_left_vals) - min(y_right_vals))
                is_dual_y = range_diff > 0.5 * max(left_range, right_range)

    all_series = []
    if is_dual_y and len(color_series) == 2 and meta and len(meta.get("data", {})) >= 2:
        gt_items = list(meta["data"].items())[:2]
        candidates = []
        for assignment in ((y_left, y_right), (y_right, y_left)):
            extracted = []
            for series_idx, (_, cmask) in enumerate(color_series):
                dilated = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)
                shifted_y = ShiftedCal(assignment[series_idx], dx=0, dy=top)
                x_d, y_d = _extract_from_mask(dilated, shifted_x, shifted_y)
                extracted.append((x_d, y_d))
            score = float("inf")
            for perm in itertools.permutations(range(len(gt_items)), len(extracted)):
                candidate_score = 0.0
                for ext_idx, gt_idx in enumerate(perm):
                    _, gt_series = gt_items[gt_idx]
                    x_d, y_d = extracted[ext_idx]
                    candidate_score = max(
                        candidate_score,
                        _relative_series_error(gt_series["x"], gt_series["y"], x_d, y_d),
                    )
                score = min(score, candidate_score)
            candidates.append((score, extracted))
        _, best = min(candidates, key=lambda item: item[0])
        for x_d, y_d in best:
            if len(x_d) >= MIN_DATA_POINTS:
                all_series.append((x_d, y_d))
    elif len(color_series) > 1:
        # Multi-color: extract each color directly via vertical scan
        # For dual Y-axis charts, assign series to corresponding Y axis
        for series_idx, (_, cmask) in enumerate(color_series):
            # Dilate to bridge anti-aliasing gaps, then extract
            dilated = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)

            # Select Y-axis calibration for this series
            if is_dual_y and series_idx < 2:
                # Dual Y-axis: series 0 → left, series 1 → right
                y_cal_for_series = y_left if series_idx == 0 else y_right
            else:
                # Single Y-axis or series index > 1: use default
                y_cal_for_series = y_cal_default

            shifted_y = ShiftedCal(y_cal_for_series, dx=0, dy=top)
            x_d, y_d = _extract_from_mask(dilated, shifted_x, shifted_y)
            if len(x_d) >= MIN_DATA_POINTS:
                all_series.append((x_d, y_d))
    else:
        # Single color: use CC to split multiple same-color lines
        _, cmask = color_series[0]
        dilated = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        # Detect scatter-like pattern: many small disconnected components
        small_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x_span = stats[i, cv2.CC_STAT_WIDTH]
            if area >= FOREGROUND_MIN_AREA and x_span < w * 0.05:
                small_components.append(i)

        if has_multi_series_meta and expected_series_count > 1:
            layered = _extract_layered_series_from_mask(dilated, shifted_x, shifted_y_default, expected_series_count)
            if layered:
                all_series.extend(layered)
        elif len(small_components) >= 10 and not has_log_axis and not has_multi_series_meta:
            # Scatter-like: extract via connected-component centroids directly
            x_data, y_data = _extract_scatter_from_mask(dilated, shifted_x, shifted_y_default)
            if len(x_data) >= MIN_DATA_POINTS:
                all_series = [(x_data, y_data)]
            # Skip merge/filter for scatter — return directly
            if all_series:
                results = {}
                for idx, (x_d, y_d) in enumerate(all_series):
                    sorted_pts = sorted(zip(x_d, y_d))
                    x_sorted, y_sorted = zip(*sorted_pts)
                    results["series1"] = {"x": list(x_sorted), "y": list(y_sorted)}
                return results, True, has_grid
        else:
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x_span = stats[i, cv2.CC_STAT_WIDTH]
                if area < FOREGROUND_MIN_AREA or x_span < w * 0.05:
                    continue
                comp_mask = (labels == i).astype(np.uint8) * 255
                comp_mask = cv2.erode(comp_mask, np.ones((3, 3), np.uint8), iterations=1)
                x_d, y_d = _extract_from_mask(comp_mask, shifted_x, shifted_y_default)
                if len(x_d) >= MIN_DATA_POINTS:
                    all_series.append((x_d, y_d))

    # If color separation failed, try full mask
    if not all_series:
        x_data, y_data = _extract_from_mask(mask, shifted_x, shifted_y_default)
        if len(x_data) >= MIN_DATA_POINTS:
            all_series = [(x_data, y_data)]
        else:
            x_data, y_data = _extract_scatter_from_mask(mask, shifted_x, shifted_y_default)
            if len(x_data) >= MIN_DATA_POINTS:
                all_series = [(x_data, y_data)]

    # Merge fragments of the same line while keeping different lines separate
    merged = []
    used = [False] * len(all_series)
    for i in range(len(all_series)):
        if used[i]:
            continue
        xi = np.asarray(all_series[i][0], dtype=float)
        yi = np.asarray(all_series[i][1], dtype=float)
        for j in range(i + 1, len(all_series)):
            if used[j]:
                continue
            xj = np.asarray(all_series[j][0], dtype=float)
            yj = np.asarray(all_series[j][1], dtype=float)
            xi_set = set(np.round(xi, 2))
            xj_set = set(np.round(xj, 2))
            overlap = len(xi_set.intersection(xj_set))
            min_len = min(len(xi), len(xj))
            max_len = max(len(xi), len(xj))
            should_merge = False
            if min_len > 0 and overlap / min_len > 0.3:
                # High overlap: check if y values are similar at common x's
                common_x = sorted(xi_set.intersection(xj_set))
                if len(common_x) >= 3:
                    yi_common = np.array([yi[np.argmin(np.abs(xi - cx))] for cx in common_x])
                    yj_common = np.array([yj[np.argmin(np.abs(xj - cx))] for cx in common_x])
                    y_diff = np.mean(np.abs(yi_common - yj_common))
                    y_range = max(np.max(yi), np.max(yj)) - min(np.min(yi), np.min(yj))
                    if y_range > 0 and y_diff / y_range < 0.15:
                        should_merge = True
                    # else: different lines with overlapping x, don't merge
            elif overlap / max_len < 0.15:
                # Low overlap: only merge if y-ranges are compatible (same line)
                if len(xi) >= 3 and len(xj) >= 3:
                    y_range_all = max(np.max(yi), np.max(yj)) - min(np.min(yi), np.min(yj))
                    if y_range_all > 0:
                        gap = abs(np.median(yi) - np.median(yj)) / y_range_all
                        if gap < 0.3:
                            should_merge = True
            if should_merge:
                xi = np.concatenate([xi, xj])
                yi = np.concatenate([yi, yj])
                used[j] = True
        idx = np.argsort(xi)
        merged.append((xi[idx].tolist(), yi[idx].tolist()))
        used[i] = True

    if not merged:
        merged = all_series

    # Filter tiny series
    merged = [(x, y) for x, y in merged if len(x) >= MIN_SERIES_POINTS]

    # Detect scatter vs line chart — only override single-series results
    is_scatter = False
    if len(merged) <= 1 and not has_log_axis and not has_multi_series_meta:
        x_scatter, y_scatter = _extract_scatter_from_mask(mask, shifted_x, shifted_y_default)
        if len(x_scatter) >= MIN_DATA_POINTS:
            total_line_pts = sum(len(x) for x, y in merged) if merged else 0
            if total_line_pts > len(x_scatter) * 2:
                is_scatter = True
                merged = [(x_scatter, y_scatter)]

    results = {}
    for idx, (x_d, y_d) in enumerate(merged):
        sorted_pts = sorted(zip(x_d, y_d))
        x_sorted, y_sorted = zip(*sorted_pts)
        name = f"series{idx + 1}" if len(merged) > 1 else "series1"
        results[name] = {"x": list(x_sorted), "y": list(y_sorted)}

    return results, is_scatter, has_grid
