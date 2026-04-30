"""Skeleton graph traversal for multi-series separation.

Builds a NetworkX graph from a binary skeleton image, extracts
branches between endpoints/junctions, and assigns branches to
series using tangential continuity at junctions.
"""
import numpy as np
import networkx as nx
from typing import List, Tuple

_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]


def _neighbor_coords(
    skeleton: np.ndarray, x: int, y: int,
) -> List[Tuple[int, int]]:
    """Return foreground 8-connected neighbor coordinates."""
    h, w = skeleton.shape
    result = []
    for dx, dy in _DIRECTIONS:
        nx_, ny_ = x + dx, y + dy
        if 0 <= nx_ < w and 0 <= ny_ < h and skeleton[ny_, nx_]:
            result.append((nx_, ny_))
    return result


def build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """Convert binary skeleton to NetworkX graph.

    Nodes: (x, y) coordinates with attribute 'node_type':
      - 'endpoint': exactly 1 foreground neighbor
      - 'junction': 3+ foreground neighbors
      - 'connection': exactly 2 foreground neighbors
    Edges: 8-connected neighbors on skeleton.
    """
    skel = (skeleton > 0).astype(np.uint8)
    if skel.sum() == 0:
        return nx.Graph()

    h, w = skel.shape
    graph = nx.Graph()

    ys, xs = np.where(skel > 0)
    for x, y in zip(xs, ys):
        neighbors = _neighbor_coords(skel, int(x), int(y))
        n = len(neighbors)
        if n <= 1:
            node_type = "endpoint"
        elif n >= 3:
            node_type = "junction"
        else:
            node_type = "connection"
        graph.add_node((int(x), int(y)), node_type=node_type)
        for nx_, ny_ in neighbors:
            graph.add_edge((int(x), int(y)), (nx_, ny_))

    return graph


def extract_branches(graph: nx.Graph) -> List[List[Tuple[int, int]]]:
    """Extract continuous branches between endpoints/junctions.

    A branch is a path whose internal nodes are all 'connection' nodes.
    The path includes the terminal endpoint/junction nodes at each end.
    """
    if len(graph.nodes) == 0:
        return []

    special_nodes = {
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") in ("endpoint", "junction")
    }

    if not special_nodes:
        return [list(graph.nodes)]

    visited_edges: set = set()
    branches: list = []

    for start in special_nodes:
        for neighbor in graph.neighbors(start):
            edge = frozenset({start, neighbor})
            if edge in visited_edges:
                continue

            path = [start, neighbor]
            visited_edges.add(edge)
            current = neighbor

            while current not in special_nodes:
                next_nodes = [
                    n for n in graph.neighbors(current)
                    if frozenset({current, n}) not in visited_edges
                ]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                visited_edges.add(frozenset({current, nxt}))
                path.append(nxt)
                current = nxt

            branches.append(path)

    return branches


def _branch_direction(branch: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Compute average direction vector of a branch."""
    if len(branch) < 2:
        return (0.0, 0.0)
    dx = float(branch[-1][0] - branch[0][0])
    dy = float(branch[-1][1] - branch[0][1])
    norm = np.sqrt(dx * dx + dy * dy)
    if norm < 1e-9:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def _angle_between(d1: Tuple[float, float], d2: Tuple[float, float]) -> float:
    """Angle in degrees between two direction vectors."""
    dot = d1[0] * d2[0] + d1[1] * d2[1]
    cos_a = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(cos_a)))


def assign_branches_to_series(
    branches: List[List[Tuple[int, int]]],
    n_series: int,
) -> List[List[List[Tuple[int, int]]]]:
    """Group branches into n_series continuous curves.

    Uses greedy angle-based assignment: at each junction, assign
    outgoing branches to series by tangential continuity (minimize
    angle change between branch directions).
    """
    series: List[List[List[Tuple[int, int]]]] = [
        [] for _ in range(n_series)
    ]

    if not branches:
        return series

    if n_series == 1:
        series[0] = list(branches)
        return series

    if len(branches) <= n_series:
        for i, branch in enumerate(branches):
            series[i % n_series].append(branch)
        return series

    directions = [_branch_direction(b) for b in branches]
    lengths = [len(b) for b in branches]

    sorted_indices = sorted(
        range(len(branches)),
        key=lambda i: lengths[i],
        reverse=True,
    )

    assigned: set = set()
    series_dirs: list = [None] * n_series

    for idx in sorted_indices[:n_series]:
        si = idx % n_series
        series[si].append(branches[idx])
        series_dirs[si] = directions[idx]
        assigned.add(idx)

    for idx in sorted_indices[n_series:]:
        branch_dir = directions[idx]
        best_series = 0
        best_angle = 360.0
        for si in range(n_series):
            if series_dirs[si] is None:
                angle = 180.0
            else:
                angle = _angle_between(branch_dir, series_dirs[si])
            if angle < best_angle:
                best_angle = angle
                best_series = si
        series[best_series].append(branches[idx])
        series_dirs[best_series] = branch_dir

    return series


def branches_to_mask(
    branches: List[List[Tuple[int, int]]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Render branch coordinates back to binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    for branch in branches:
        for x, y in branch:
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = 1
    return mask
