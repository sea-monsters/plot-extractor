"""Junction-aware skeleton path tracing.

Based on 1802.05902 (hand-drawn sketch vectorization):
detects endpoints and branch points in a thinned skeleton,
traces continuous paths, and resolves branches by direction continuity.

Replaces column-median extraction for dense charts.
"""
import numpy as np
from typing import List, Tuple

# 8-connected neighborhood offsets (clockwise from north)
_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]


def angle_diff(dir1: tuple, dir2: tuple) -> float:
    """Angle between two direction vectors in degrees."""
    dot = dir1[0] * dir2[0] + dir1[1] * dir2[1]
    norm1 = np.sqrt(dir1[0] ** 2 + dir1[1] ** 2)
    norm2 = np.sqrt(dir2[0] ** 2 + dir2[1] ** 2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return float(np.degrees(np.arccos(cos_angle)))


def _neighbor_count(skeleton: np.ndarray, x: int, y: int) -> int:
    h, w = skeleton.shape
    count = 0
    for dx, dy in _DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx]:
            count += 1
    return count


def classify_skeleton_points(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Classify skeleton foreground pixels into endpoints and junctions.

    Returns:
        (endpoints, junctions) where:
        - endpoints have exactly 1 foreground neighbor
        - junctions have >= 3 foreground neighbors
    """
    h, w = skeleton.shape
    ys, xs = np.where(skeleton > 0)

    endpoints = []
    junctions = []

    for x, y in zip(xs, ys):
        n = _neighbor_count(skeleton, int(x), int(y))
        if n == 1:
            endpoints.append((int(x), int(y)))
        elif n >= 3:
            junctions.append((int(x), int(y)))

    return endpoints, junctions


def trace_skeleton_paths(
    skeleton: np.ndarray,
    min_path_len: int = 10,
) -> List[List[Tuple[int, int]]]:
    """Trace continuous paths through a thinned skeleton.

    Algorithm (from 1802.05902):
    1. Detect endpoints (1 neighbor) and junctions (>=3 neighbors)
    2. From each unvisited endpoint, trace along the skeleton
    3. At junctions: continue in the direction closest to current heading
    4. For cycles (no endpoints): trace from junctions

    Args:
        skeleton: Binary skeleton image (uint8, >0 = foreground).
        min_path_len: Discard paths shorter than this.

    Returns:
        List of paths, each path is a list of (x, y) coordinates.
    """
    h, w = skeleton.shape
    skel = (skeleton > 0).astype(np.uint8)
    if skel.sum() == 0:
        return []

    visited = np.zeros((h, w), dtype=np.uint8)
    endpoints, junctions = classify_skeleton_points(skel)
    paths = []

    def _trace_from(start_x, start_y):
        path = [(start_x, start_y)]
        visited[start_y, start_x] = 1
        cx, cy = start_x, start_y

        while True:
            # Collect unvisited foreground neighbors
            candidates = []
            for dx, dy in _DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] and not visited[ny, nx]:
                    candidates.append((nx, ny))

            if not candidates:
                break

            if len(candidates) == 1:
                nx, ny = candidates[0]
            else:
                # Branch: choose direction closest to current heading
                if len(path) >= 2:
                    prev_dir = (path[-1][0] - path[-2][0],
                                path[-1][1] - path[-2][1])
                else:
                    prev_dir = (candidates[0][0] - cx,
                                candidates[0][1] - cy)
                nx, ny = min(
                    candidates,
                    key=lambda pt: angle_diff(prev_dir, (pt[0] - cx, pt[1] - cy)),
                )

            visited[ny, nx] = 1
            path.append((nx, ny))
            cx, cy = nx, ny

        return path

    # Phase 1: Trace from endpoints
    for ex, ey in endpoints:
        if visited[ey, ex]:
            continue
        path = _trace_from(ex, ey)
        if len(path) >= min_path_len:
            paths.append(path)

    # Phase 2: Trace cycles from junctions (closed loops have no endpoints)
    for jx, jy in junctions:
        if visited[jy, jx]:
            continue
        path = _trace_from(jx, jy)
        if len(path) >= min_path_len:
            paths.append(path)

    # Phase 3: Trace any remaining unvisited pixels (short segments)
    remaining_ys, remaining_xs = np.where((skel > 0) & (visited == 0))
    for rx, ry in zip(remaining_xs, remaining_ys):
        if visited[ry, rx]:
            continue
        path = _trace_from(int(rx), int(ry))
        if len(path) >= min_path_len:
            paths.append(path)

    return paths


def extract_path_data(
    paths: List[List[Tuple[int, int]]],
) -> Tuple[List[int], List[int]]:
    """Convert traced paths to flat x, y coordinate lists.

    Args:
        paths: Output of trace_skeleton_paths().

    Returns:
        (xs, ys) coordinate lists suitable for axis calibration.
    """
    if not paths:
        return [], []

    xs = []
    ys = []
    for path in paths:
        for x, y in path:
            xs.append(x)
            ys.append(y)

    return xs, ys
