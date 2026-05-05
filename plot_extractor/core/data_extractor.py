"""Extract data points from the plot area."""
from typing import List, Dict
import numpy as np
import cv2

from plot_extractor.core.axis_calibrator import CalibratedAxis
from plot_extractor.core.series_candidates import extract_series_multi_candidate
from plot_extractor.utils.image_utils import (
    detect_background_color,
    make_foreground_mask,
)
from plot_extractor.config import MIN_DATA_POINTS, MIN_SERIES_POINTS, FOREGROUND_MIN_AREA
from plot_extractor.core.skeleton_path import trace_skeleton_paths, extract_path_data
from plot_extractor.core.skeleton_graph import (
    build_skeleton_graph,
    extract_branches,
    assign_branches_to_series,
    branches_to_mask,
)
from plot_extractor.core.scatter_overlap import (
    detect_overlap_candidates,
    extract_template_from_ccs,
    separate_overlap_greedy,
)


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
    then any remaining interior lines indicate a grid.  Grid removal
    is data-safe: pixels near data curves (foreground above/below for
    horizontal lines, left/right for vertical lines) are preserved.
    """
    h, w = mask.shape
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=int(min(h, w) * 0.6), maxLineGap=3
    )

    has_grid = False
    if lines is not None:
        horiz_lines = []
        vert_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= 5 or angle >= 175:
                horiz_lines.append((x1, y1, x2, y2))
            elif 85 <= angle <= 95:
                vert_lines.append((x1, y1, x2, y2))

        horiz_centers = [int(round((y1 + y2) / 2)) for x1, y1, x2, y2 in horiz_lines]
        vert_centers = [int(round((x1 + x2) / 2)) for x1, y1, x2, y2 in vert_lines]

        horiz_unique = _dedup_sorted(sorted(horiz_centers))
        vert_unique = _dedup_sorted(sorted(vert_centers))

        # Identify axis border lines (outermost clusters).
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

        interior_h = [y for y in horiz_unique if y not in axis_h]
        interior_v = [x for x in vert_unique if x not in axis_v]
        has_grid = len(interior_h) >= 1 or len(interior_v) >= 1

        # Remove grid pixels with data-safe per-pixel check.
        # For horizontal grid lines: skip pixels where foreground exists
        # above and below (data curve crossing through).
        for y_grid in interior_h:
            for x in range(w):
                if mask[y_grid, x] == 0:
                    continue
                # Data crossing check: foreground within ±3 pixels above/below
                y_lo = max(0, y_grid - 3)
                y_hi = min(h, y_grid + 4)
                above = np.any(mask[y_lo:y_grid, x] > 0)
                below = np.any(mask[y_grid + 1:y_hi, x] > 0)
                if above and below:
                    continue  # Data curve crossing — preserve
                mask[y_grid, x] = 0

        # For vertical grid lines: skip pixels where foreground exists
        # left and right (data curve crossing through).
        for x_grid in interior_v:
            for y in range(h):
                if mask[y, x_grid] == 0:
                    continue
                x_lo = max(0, x_grid - 3)
                x_hi = min(w, x_grid + 4)
                left_fg = np.any(mask[y, x_lo:x_grid] > 0)
                right_fg = np.any(mask[y, x_grid + 1:x_hi] > 0)
                if left_fg and right_fg:
                    continue  # Data curve crossing — preserve
                mask[y, x_grid] = 0

    return mask, has_grid


class _ShiftedCal:
    """Wraps CalibratedAxis with pixel offsets for plot-area-local coordinates."""

    def __init__(self, base: CalibratedAxis, dx=0, dy=0):
        self.base = base
        self.dx = dx
        self.dy = dy

    def to_data(self, pixel):
        if self.base.axis.direction == "x":
            return self.base.to_data(pixel + self.dx)
        return self.base.to_data(pixel + self.dy)


def _median_filter(arr, window):
    """Apply sliding window median filter."""
    result = np.copy(arr)
    half = window // 2
    for i in range(half, len(arr) - half):
        result[i] = np.median(arr[i - half : i + half + 1])
    return result


def _extract_from_mask(mask, x_cal, y_cal):
    """Extract line points by vertical scanning of a mask."""
    _, w = mask.shape
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


def _extract_from_skeleton_paths(mask, x_cal, y_cal):
    """Extract data points using skeleton path tracing.

    Replaces column-median strategy for dense charts. Traces continuous
    paths through the thinned skeleton, preserving curve identity at
    crossing points via direction-based branch resolution.

    Based on 1802.05902 (hand-drawn sketch vectorization).
    """
    paths = trace_skeleton_paths(mask, min_path_len=5)
    if not paths:
        return _extract_from_mask(mask, x_cal, y_cal)

    xs_raw, ys_raw = extract_path_data(paths)
    if len(xs_raw) < MIN_DATA_POINTS:
        return _extract_from_mask(mask, x_cal, y_cal)

    x_data = [x_cal.to_data(x) for x in xs_raw]
    y_data = [y_cal.to_data(y) for y in ys_raw]
    valid = [(xd, yd) for xd, yd in zip(x_data, y_data) if xd is not None and yd is not None]
    if not valid:
        return [], []
    x_out, y_out = zip(*valid)
    return list(x_out), list(y_out)


def _extract_layered_series_from_mask(mask, x_cal, y_cal, series_count):
    """Extract same-color multiple curves by per-column vertical layers.

    Enhanced with crossing-region smoothing to avoid identity jumps when
    detected group count changes rapidly between adjacent columns.
    """
    if series_count <= 1:
        return []

    _, w = mask.shape
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
            while len(centers) < series_count:
                gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
                insert_at = int(np.argmax(gaps)) if gaps else 0
                if gaps:
                    centers.insert(insert_at + 1, (centers[insert_at] + centers[insert_at + 1]) / 2.0)
                else:
                    centers.append(centers[-1])
        elif len(centers) > series_count:
            idx = np.linspace(0, len(centers) - 1, series_count).round().astype(int)
            centers = [centers[i] for i in idx]

        centers = centers[:series_count]
        if prev_centers is not None and len(centers) == series_count:
            # Greedy nearest-neighbor assignment: O(n²) instead of O(n!)
            assigned = [False] * len(centers)
            order = [None] * len(centers)
            # Match each prev center to its closest available new center
            for i, prev_y in sorted(enumerate(prev_centers),
                                    key=lambda ip: min(abs(prev_y - c)
                                                       for c in centers)):
                best_j = None
                best_dist = float("inf")
                for j, c in enumerate(centers):
                    if not assigned[j]:
                        d = abs(prev_y - c)
                        if d < best_dist:
                            best_dist = d
                            best_j = j
                if best_j is not None:
                    assigned[best_j] = True
                    order[i] = centers[best_j]
            # Fill any unmatched (shouldn't happen with equal lengths)
            for j, c in enumerate(centers):
                if not assigned[j]:
                    for i in range(len(order)):
                        if order[i] is None:
                            order[i] = c
                            break
            centers = order
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
    """Extract scatter points by connected components with overlap separation."""
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Build CC stats list for overlap detection
    cc_list = []
    for i in range(1, num_labels):
        cc_list.append({
            "label": i,
            "area": int(stats[i, cv2.CC_STAT_AREA]),
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]),
            "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1]),
        })

    # Detect and separate overlapping scatter points
    overlaps = detect_overlap_candidates(cc_list)
    template = extract_template_from_ccs(mask, cc_list) if overlaps else None

    xs = []
    ys = []

    overlap_labels = set(oc["label"] for oc in overlaps)

    for cc in cc_list:
        area = cc["area"]
        if area < FOREGROUND_MIN_AREA:
            continue

        if cc["label"] in overlap_labels and template is not None:
            # Separate overlapping points
            split_centroids = separate_overlap_greedy(mask, cc, template)
            for sx, sy in split_centroids:
                xs.append(sx)
                ys.append(sy)
        else:
            xs.append(int(cc["cx"]))
            ys.append(int(cc["cy"]))

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


def _apply_thinning(mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """Skeletonize a binary mask using Zhang-Suen thinning (NumPy fallback).

    Falls back to returning the original mask if cv2.ximgproc.thinning is
    unavailable.  Input is a uint8 mask; output is a uint8 skeleton mask.
    """
    # Prefer OpenCV contrib implementation if present
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            binary = (mask > 0).astype(np.uint8) * 255
            return cv2.ximgproc.thinning(
                binary, None, cv2.ximgproc.THINNING_ZHANGSUEN
            )
    except cv2.error:
        pass

    # Pure NumPy Zhang-Suen fallback
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iterations):
        changed = False
        for subiter_idx in range(2):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            neigh = np.stack([p2, p3, p4, p5, p6, p7, p8, p9, p2], axis=0)
            s = np.sum((neigh[:-1] == 0) & (neigh[1:] == 1), axis=0)

            if subiter_idx == 0:
                to_del = (
                    (img == 1)
                    & (n >= 2)
                    & (n <= 6)
                    & (s == 1)
                    & (p2 * p4 * p6 == 0)
                    & (p4 * p6 * p8 == 0)
                )
            else:
                to_del = (
                    (img == 1)
                    & (n >= 2)
                    & (n <= 6)
                    & (s == 1)
                    & (p2 * p4 * p8 == 0)
                    & (p2 * p6 * p8 == 0)
                )

            if np.any(to_del):
                img[to_del] = 0
                changed = True

        if not changed:
            break

    return (img * 255).astype(np.uint8)


def _estimate_x_bands(mask, image, color_mask, fg_indices):
    """Estimate number of series from x-axis color distribution."""
    h, w = mask.shape
    hues = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]
    col_hues = []
    for x in range(w):
        col_mask = (mask[:, x] > 0) & (hues[:, x] > 0)
        if np.sum(col_mask) > 0:
            col_hues.append(np.median(hues[:, x][col_mask]))
    if len(col_hues) < 10:
        return 1
    col_hues_arr = np.array(col_hues).reshape(-1, 1).astype(np.float32)
    if len(col_hues_arr) < 3:
        return 1
    _, labels, _ = cv2.kmeans(
        col_hues_arr,
        min(5, len(col_hues_arr)),
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    return len(set(labels.flatten()))


def _validate_color_clusters(series_list, image_width, min_x_coverage=0.05):
    """Validate that color-separated clusters are plausible series."""
    if len(series_list) < 2:
        return False
    valid_count = 0
    for _, _, cmask in series_list:
        cols = np.where(np.any(cmask > 0, axis=0))[0]
        if len(cols) > 0:
            coverage = (cols.max() - cols.min()) / image_width
            if coverage >= min_x_coverage:
                valid_count += 1
    return valid_count >= 2


def _merge_overlapping_clusters(series_list, overlap_threshold=0.5):
    """Merge 3D clusters that overlap significantly in x-range."""
    if len(series_list) < 2:
        return series_list
    merged = []
    used = [False] * len(series_list)
    for i in range(len(series_list)):
        if used[i]:
            continue
        _, img_i, mask_i = series_list[i]
        cols_i = set(np.where(np.any(mask_i > 0, axis=0))[0])
        for j in range(i + 1, len(series_list)):
            if used[j]:
                continue
            _, _, mask_j = series_list[j]
            cols_j = set(np.where(np.any(mask_j > 0, axis=0))[0])
            if not cols_i or not cols_j:
                continue
            overlap = len(cols_i & cols_j) / min(len(cols_i), len(cols_j))
            if overlap > overlap_threshold:
                # Merge j into i by OR-ing masks
                mask_i = cv2.bitwise_or(mask_i, mask_j)
                used[j] = True
        merged.append((i, img_i, mask_i))
        used[i] = True
    return merged


def _separate_series_by_color(image, mask, n_clusters=3, min_clusters=1):
    """Separate multiple data series by clustering hue in HSV space.

    Falls back to full H+S+V 3D clustering when hue-only separation quality
    is insufficient.
    """
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

    # Data-driven series count estimation (image-based only)
    x_bands = _estimate_x_bands(mask, image, color_mask, fg_indices)

    K_hue = min(max(len(peaks), min_clusters, 1), n_clusters)
    K_data = min(max(x_bands, min_clusters, 1), n_clusters)

    K = max(K_hue, K_data)
    if K < 2 and len(peaks) >= 2:
        K = len(peaks)

    if K < 2:
        return [(image, mask)]

    # Hue-only kmeans
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

    series_hue = []
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
            series_hue.append((float(centers[k]), image, cluster_mask))

    # Quality validation for hue-only result
    quality_ok = _validate_color_clusters(series_hue, w, min_x_coverage=0.05)

    if quality_ok and len(series_hue) >= 2:
        return [(img, cmask) for _, img, cmask in series_hue]

    # Fallback: Full H+S+V 3D clustering
    hsv_pixels = pixels[color_mask].astype(np.float32)
    hsv_norm = hsv_pixels.copy()
    hsv_norm[:, 0] /= 179.0
    hsv_norm[:, 1] /= 255.0
    hsv_norm[:, 2] /= 255.0

    K_3d = max(K, 2)

    criteria_hsv = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    cv2.setRNGSeed(0)
    _, labels_3d, _ = cv2.kmeans(
        hsv_norm,
        K_3d,
        None,
        criteria_hsv,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    full_labels_3d = np.full(len(pixels), -1, dtype=int)
    full_labels_3d[color_mask] = labels_3d.flatten()

    series_hsv = []
    for k in range(K_3d):
        cluster_mask = np.zeros_like(mask)
        cluster_indices = np.where(full_labels_3d == k)[0]
        if len(cluster_indices) == 0:
            continue
        rows = fg_indices[0][cluster_indices]
        cols = fg_indices[1][cluster_indices]
        cluster_mask[rows, cols] = 255

        x_spread = int(cols.max()) - int(cols.min()) if len(cols) > 0 else 0
        if len(cluster_indices) >= FOREGROUND_MIN_AREA and x_spread > w * 0.05:
            series_hsv.append((k, image, cluster_mask))

    # Post-merge over-segmented clusters
    series_hsv = _merge_overlapping_clusters(series_hsv, overlap_threshold=0.5)

    if len(series_hsv) >= 2:
        return [(img, cmask) for _, img, cmask in series_hsv]

    # P1: Skeleton graph fallback when color separation fails entirely
    if n_clusters >= 2 and np.sum(mask > 0) >= FOREGROUND_MIN_AREA * 2:
        skel_graph = build_skeleton_graph(mask > 0)
        branches = extract_branches(skel_graph)
        if len(branches) >= 2:
            series_branches = assign_branches_to_series(branches, n_series=n_clusters)
            skeleton_results = []
            for s_branches in series_branches:
                if not s_branches:
                    continue
                smask = branches_to_mask(s_branches, mask.shape)
                # Dilate slightly to recover original curve thickness
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                smask = cv2.dilate(smask, kernel, iterations=1)
                x_spread = int(np.ptp(np.where(smask > 0)[1])) if np.any(smask > 0) else 0
                if np.sum(smask > 0) >= FOREGROUND_MIN_AREA and x_spread > w * 0.05:
                    skeleton_results.append((image, smask))
            if len(skeleton_results) >= 2:
                return skeleton_results

    if not series_hue:
        return [(image, mask)]
    series_hue.sort(key=lambda item: item[0], reverse=True)
    return [(img, cmask) for _, img, cmask in series_hue]


def _is_dense_chart(mask, series_count=1):
    """Detect dense oscillating curves independent of color/CC path."""
    h, w = mask.shape

    # Vectorized: compute all metrics in single column-wise passes
    col_has_fg = np.any(mask > 0, axis=0)
    col_density = np.mean(col_has_fg)

    # Row spans: max row - min row per column (vectorized)
    fg_rows = np.where(mask > 0)
    if len(fg_rows[0]) == 0:
        return False
    # Build per-column min/max row index
    row_min = np.full(w, h, dtype=int)
    row_max = np.full(w, -1, dtype=int)
    np.minimum.at(row_min, fg_rows[1], fg_rows[0])
    np.maximum.at(row_max, fg_rows[1], fg_rows[0])
    has_fg = col_has_fg
    spans = np.where(has_fg, row_max - row_min, 0)
    avg_span = np.mean(spans[has_fg]) if np.any(has_fg) else 0
    span_ratio = avg_span / h

    # Dense if: high column density AND significant vertical span.
    return col_density > 0.6 and span_ratio > 0.15


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


def extract_all_data(image, calibrated_axes: List[CalibratedAxis], image_path=None, raw_image=None, policy=None, chart_struct=None) -> Dict[str, Dict]:
    """Main extraction pipeline.

    raw_image: un-preprocessed image used for grid detection (preprocessing
    blurs faint grid lines, so detection is more reliable on the original).
    chart_struct: optional ChartStructure for constraining extraction to plot_area.
    """
    if not calibrated_axes:
        return {}, False, False

    h, w = image.shape[:2]
    left, top, right, bottom = _get_plot_bounds(calibrated_axes, image.shape)

    # Refine with chart structure when available
    if chart_struct is not None:
        pa = chart_struct.plot_area
        left = max(left, pa.x1)
        top = max(top, pa.y1)
        right = min(right, pa.x2)
        bottom = min(bottom, pa.y2)
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

    # P3: FFT grid removal enhancement for periodic grids
    if has_grid and plot_img.size > 0:
        from plot_extractor.geometry.grid_suppress import suppress_grid_lines_fft  # pylint: disable=import-outside-toplevel
        fft_cleaned = suppress_grid_lines_fft(plot_img)
        # Convert back to RGB for make_foreground_mask compatibility
        if len(fft_cleaned.shape) == 2:
            fft_rgb = cv2.cvtColor(fft_cleaned, cv2.COLOR_GRAY2RGB)
        else:
            fft_rgb = fft_cleaned
        fft_mask = make_foreground_mask(fft_rgb, bg_color)
        fft_mask, _ = _remove_grid_lines(fft_mask)
        # Use FFT mask only if it preserves significant foreground
        # and does not blow up the mask (FFT artifacts can invert contrast)
        fft_fg = np.sum(fft_mask > 0)
        orig_fg = np.sum(mask > 0)
        if orig_fg > 0 and fft_fg >= orig_fg * 0.5 and fft_fg <= orig_fg * 3:
            mask = fft_mask

    x_cals = [ca for ca in calibrated_axes if ca.axis.direction == "x"]
    y_cals = [ca for ca in calibrated_axes if ca.axis.direction == "y"]

    if not x_cals or not y_cals:
        return {}, False, False

    def _axis_quality_score(ca) -> float:
        """Prefer the axis with the most reliable label/calibration evidence."""
        score = 0.0
        score += 10.0 if ca.axis.side in ("bottom", "left") else 0.0
        score += float(getattr(ca, "labeled_tick_count", 0) or len(getattr(ca, "tick_map", []) or [])) * 2.0
        score += float(getattr(ca, "formula_anchor_count", 0)) * 6.0
        score += float(getattr(ca, "tesseract_anchor_count", 0)) * 2.0
        source = getattr(ca, "tick_source", "heuristic")
        if source == "formula_generated":
            score += 35.0
        elif source == "formula":
            score += 28.0
        elif source == "fused":
            score += 22.0
        elif source == "tesseract":
            score += 8.0
        elif source == "heuristic":
            score -= 8.0
        residual = float(getattr(ca, "residual", 1e6) or 1e6)
        if residual < 1:
            score += 20.0
        elif residual < 10:
            score += 14.0
        elif residual < 100:
            score += 6.0
        elif residual > 1e5:
            score -= 20.0
        return score

    x_cal = max(x_cals, key=_axis_quality_score)
    has_log_axis = any(ca.axis_type == "log" for ca in calibrated_axes)

    # Detect if there are both left and right Y axes (dual_y scenario)
    y_left = next((ca for ca in y_cals if ca.axis.side == "left"), None)
    y_right = next((ca for ca in y_cals if ca.axis.side == "right"), None)

    # Default to left axis if only one Y axis
    y_cal_default = y_left if y_left else (y_right if y_right else y_cals[0])

    shifted_x = _ShiftedCal(x_cal, dx=left, dy=0)
    shifted_y_default = _ShiftedCal(y_cal_default, dx=0, dy=top)

    # --- UNIFIED DENSE DETECTION (runs on full mask before color separation) ---
    is_dense = _is_dense_chart(mask)
    # Policy override: if policy explicitly routes to thinning or scatter, respect it
    density_strategy = policy.density_strategy if policy is not None else "standard"
    if density_strategy == "thinning" or (is_dense and density_strategy != "scatter"):
        thinned = _apply_thinning(mask)
        # Quality gate: if thinning destroyed too much, fall back
        gate = policy.thinning_quality_gate if policy is not None else 0.3
        thinned_cols = np.sum(np.any(thinned > 0, axis=0))
        if thinned_cols < w * gate:
            thinned = mask
        x_data, y_data = _extract_from_skeleton_paths(thinned, shifted_x, shifted_y_default)
        if len(x_data) >= MIN_DATA_POINTS:
            results = {"series1": {"x": x_data, "y": y_data}}
            return results, False, has_grid

    # Separate by color first, then extract each color mask directly
    color_cluster_count = 3
    color_min_clusters = 1
    if policy is not None:
        if policy.color_strategy == "hsv3d":
            color_min_clusters = max(color_min_clusters, policy.min_clusters)
        elif policy.color_strategy == "layered":
            color_min_clusters = max(color_min_clusters, 2)
        color_cluster_count = max(color_cluster_count, color_min_clusters)

    color_series = _separate_series_by_color(
        plot_img,
        mask,
        n_clusters=color_cluster_count,
        min_clusters=color_min_clusters,
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
    if is_dual_y and len(color_series) == 2:
        # Default ordering: series 0 -> left, series 1 -> right
        for series_idx, (_, cmask) in enumerate(color_series):
            dilated = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)
            y_cal_for_series = y_left if series_idx == 0 else y_right
            shifted_y = _ShiftedCal(y_cal_for_series, dx=0, dy=top)
            x_d, y_d = _extract_from_mask(dilated, shifted_x, shifted_y)
            if len(x_d) >= MIN_DATA_POINTS:
                all_series.append((x_d, y_d))
    elif len(color_series) > 1:
        # Multi-color: extract each color directly via vertical scan
        # For dual Y-axis charts, assign series to corresponding Y axis
        for series_idx, (_, cmask) in enumerate(color_series):
            # Dilate to bridge anti-aliasing gaps, then extract
            dilated = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)

            # Select Y-axis calibration for this series
            y_cal_for_series = y_cal_default

            if is_dual_y and y_left and y_right:
                # Default ordering (series 0 → left, series 1 → right)
                y_cal_for_series = y_left if series_idx == 0 else y_right

            shifted_y = _ShiftedCal(y_cal_for_series, dx=0, dy=top)
            x_d, y_d = _extract_from_mask(dilated, shifted_x, shifted_y)
            if len(x_d) >= MIN_DATA_POINTS:
                all_series.append((x_d, y_d))
    else:
        # Single color: use CC to split multiple same-color lines
        _, cmask = color_series[0]
        dilated = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8,
        )

        # Detect scatter-like pattern: many small disconnected components
        small_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x_span = stats[i, cv2.CC_STAT_WIDTH]
            if area >= FOREGROUND_MIN_AREA and x_span < w * 0.05:
                small_components.append(i)

        if len(small_components) >= 10 and not has_log_axis:
            # Could be dense oscillating curve or scatter
            fg_cols = int(np.sum(np.any(dilated > 0, axis=0)))
            # Policy override for density strategy
            density_strategy = policy.density_strategy if policy is not None else "standard"
            force_scatter = density_strategy == "scatter"
            force_thinning = density_strategy == "thinning"

            if force_thinning or (fg_cols > w * 0.7 and not force_scatter):
                # Dense: thin to 1px skeleton then extract via path tracing
                thinned = _apply_thinning(dilated)
                x_data, y_data = _extract_from_skeleton_paths(thinned, shifted_x, shifted_y_default)
                if len(x_data) >= MIN_DATA_POINTS:
                    all_series = [(x_data, y_data)]
            elif force_scatter or fg_cols <= w * 0.7:
                # Scatter: extract via connected-component centroids
                x_data, y_data = _extract_scatter_from_mask(dilated, shifted_x, shifted_y_default)
                if len(x_data) >= MIN_DATA_POINTS:
                    all_series = [(x_data, y_data)]
            # Skip merge/filter for scatter/dense — return directly
            if all_series:
                results = {}
                for idx, (x_d, y_d) in enumerate(all_series):
                    sorted_pts = sorted(zip(x_d, y_d))
                    x_sorted, y_sorted = zip(*sorted_pts)
                    results[f"series{idx+1}"] = {"x": list(x_sorted), "y": list(y_sorted)}
                is_scatter_out = force_scatter or (not force_thinning and fg_cols <= w * 0.7)
                return results, is_scatter_out, has_grid
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

    # If color separation failed, try full mask then series_candidates
    if not all_series:
        x_data, y_data = _extract_from_mask(mask, shifted_x, shifted_y_default)
        if len(x_data) >= MIN_DATA_POINTS:
            all_series = [(x_data, y_data)]

    # Fallback: multi-candidate series extraction
    if not all_series:
        is_scatter_flag = density_strategy == "scatter"
        cand_result = extract_series_multi_candidate(
            plot_img, mask, shifted_x, shifted_y_default,
            is_scatter=is_scatter_flag,
        )
        for cand in cand_result.best:
            if len(cand.x_data) >= MIN_DATA_POINTS:
                all_series.append((cand.x_data, cand.y_data))

    if not all_series:
        x_data, y_data = _extract_scatter_from_mask(mask, shifted_x, shifted_y_default)
        if len(x_data) >= MIN_DATA_POINTS:
            all_series = [(x_data, y_data)]

    def _merge_series_fragments(series_list):
        # Merge fragments of the same line while keeping different lines separate.
        merged = []
        used = [False] * len(series_list)
        for i in range(len(series_list)):
            if used[i]:
                continue
            xi = np.asarray(series_list[i][0], dtype=float)
            yi = np.asarray(series_list[i][1], dtype=float)
            for j in range(i + 1, len(series_list)):
                if used[j]:
                    continue
                xj = np.asarray(series_list[j][0], dtype=float)
                yj = np.asarray(series_list[j][1], dtype=float)
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
        return merged

    generic_merged = _merge_series_fragments(all_series)
    merged = generic_merged

    if not merged:
        merged = all_series

    # Filter tiny series
    merged = [(x, y) for x, y in merged if len(x) >= MIN_SERIES_POINTS]

    # Detect scatter vs line chart — only override single-series results
    is_scatter = False
    density_strategy = policy.density_strategy if policy is not None else "standard"
    if density_strategy == "scatter":
        # Policy forces scatter extraction
        x_scatter, y_scatter = _extract_scatter_from_mask(mask, shifted_x, shifted_y_default)
        if len(x_scatter) >= MIN_DATA_POINTS:
            is_scatter = True
            merged = [(x_scatter, y_scatter)]
    elif density_strategy != "thinning" and len(merged) <= 1 and not has_log_axis:
        # Heuristic: scatter charts have many small CCs with moderate column density.
        # Count small connected components (area < 50px²) as scatter point candidates.
        num_labels, _, stats_scatter, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        small_cc = sum(
            1 for i in range(1, num_labels)
            if stats_scatter[i, cv2.CC_STAT_AREA] < 50
            and stats_scatter[i, cv2.CC_STAT_AREA] >= 3
        )
        col_has_fg = np.any(mask > 0, axis=0)
        col_dens = np.mean(col_has_fg)
        if small_cc > 30 and col_dens < 0.7:
            x_scatter, y_scatter = _extract_scatter_from_mask(mask, shifted_x, shifted_y_default)
            if len(x_scatter) >= MIN_DATA_POINTS:
                is_scatter = True
                merged = [(x_scatter, y_scatter)]

    results = {}
    for idx, (x_d, y_d) in enumerate(merged):
        sorted_pts = sorted(zip(x_d, y_d))
        x_sorted, y_sorted = zip(*sorted_pts)
        name = f"series{idx + 1}" if len(merged) > 1 else "series1"
        results[name] = {"x": list(x_sorted), "y": list(y_sorted)}

    return results, is_scatter, has_grid
