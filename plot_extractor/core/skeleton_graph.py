"""Skeleton graph traversal for multi-series separation.

Builds an adjacency-based graph from a binary skeleton image, extracts
branches between endpoints/junctions, and assigns branches to series
using tangential continuity at junctions.

Pure-Python adjacency replaces ``networkx.Graph`` for the hot path on
dense skeletons. The v4 ``log_y/0016`` profile previously showed
``networkx.add_edge`` consuming ~7.2s of a 12.4s ``build_skeleton_graph``
call (2.67M invocations); a forward-only edge sweep with vectorized
neighbor counting removes that overhead while keeping the public API
compatible with the original NetworkX usage.
"""
import math
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

NodeKey = Tuple[int, int]

_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)
# Forward-only subset (covers each undirected edge exactly once when the
# pixel sweep walks foreground coordinates in row-major order).
_FORWARD_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 1), (0, 1), (1, 1), (1, 0),
)


class _NodesView:
    """networkx-compatible view over a node-type dict."""

    __slots__ = ("_node_types",)

    def __init__(self, node_types: Dict[NodeKey, str]) -> None:
        self._node_types = node_types

    def __len__(self) -> int:
        return len(self._node_types)

    def __iter__(self) -> Iterator[NodeKey]:
        return iter(self._node_types)

    def __contains__(self, node: object) -> bool:
        return node in self._node_types

    def __call__(self, data: bool = False) -> List:
        if data:
            return [
                (node, {"node_type": t})
                for node, t in self._node_types.items()
            ]
        return list(self._node_types)


class SkeletonGraph:
    """Minimal pure-Python adjacency graph for skeleton traversal.

    Provides the subset of the ``networkx.Graph`` API used by the
    skeleton pipeline:

    - ``graph.nodes`` is an iterable, len-able, and callable view
      (``graph.nodes(data=True)`` yields ``(node, attr_dict)``).
    - ``graph.neighbors(node)`` returns an iterable of neighbor keys.
    """

    __slots__ = ("_node_types", "_adj")

    def __init__(self) -> None:
        self._node_types: Dict[NodeKey, str] = {}
        self._adj: Dict[NodeKey, List[NodeKey]] = {}

    @property
    def nodes(self) -> _NodesView:
        return _NodesView(self._node_types)

    def add_node(self, node: NodeKey, node_type: str) -> None:
        self._node_types[node] = node_type
        if node not in self._adj:
            self._adj[node] = []

    def add_edge(self, u: NodeKey, v: NodeKey) -> None:
        """Add a bidirectional edge with duplicate suppression.

        Provided for API parity. The hot path inside
        :func:`build_skeleton_graph` uses a forward-only sweep and bypasses
        the duplicate check for speed.
        """
        adj = self._adj
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    def neighbors(self, node: NodeKey) -> List[NodeKey]:
        return self._adj.get(node, [])


def _classify_neighbor_counts(skel: np.ndarray) -> np.ndarray:
    """Vectorized 8-connected neighbor count for each foreground pixel."""
    h, w = skel.shape
    pad = np.zeros((h + 2, w + 2), dtype=np.int16)
    pad[1:-1, 1:-1] = skel
    counts = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
        + pad[1:-1, 0:-2] + pad[1:-1, 2:]
        + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
    )
    return counts


def _neighbor_coords(
    skeleton: np.ndarray, x: int, y: int,
) -> List[Tuple[int, int]]:
    """Return foreground 8-connected neighbor coordinates."""
    h, w = skeleton.shape
    result: List[Tuple[int, int]] = []
    for dx, dy in _DIRECTIONS:
        nx_, ny_ = x + dx, y + dy
        if 0 <= nx_ < w and 0 <= ny_ < h and skeleton[ny_, nx_]:
            result.append((nx_, ny_))
    return result


def build_skeleton_graph(skeleton: np.ndarray) -> SkeletonGraph:
    """Convert a binary skeleton to a :class:`SkeletonGraph`.

    Nodes are ``(x, y)`` coordinates labeled with ``node_type``:

    - ``'endpoint'``: exactly 1 foreground neighbor
    - ``'junction'``: 3+ foreground neighbors
    - ``'connection'``: exactly 2 foreground neighbors

    Edges connect 8-connected foreground neighbors on the skeleton.
    """
    skel = (skeleton > 0).astype(np.uint8)
    graph = SkeletonGraph()
    if skel.sum() == 0:
        return graph

    h, w = skel.shape
    counts = _classify_neighbor_counts(skel)
    ys, xs = np.where(skel > 0)
    ys_list = ys.tolist()
    xs_list = xs.tolist()

    node_types = graph._node_types
    adj = graph._adj
    for y, x in zip(ys_list, xs_list):
        n = int(counts[y, x])
        if n <= 1:
            node_type = "endpoint"
        elif n >= 3:
            node_type = "junction"
        else:
            node_type = "connection"
        node = (x, y)
        node_types[node] = node_type
        adj[node] = []

    # Forward-only sweep: each undirected edge added exactly once.
    for y, x in zip(ys_list, xs_list):
        node = (x, y)
        node_adj = adj[node]
        for dx, dy in _FORWARD_DIRECTIONS:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h and skel[ny_, nx_]:
                neighbor = (nx_, ny_)
                node_adj.append(neighbor)
                adj[neighbor].append(node)

    return graph


def extract_branches(graph: SkeletonGraph) -> List[List[NodeKey]]:
    """Extract continuous branches between endpoints/junctions.

    A branch is a path whose internal nodes are all ``connection`` nodes;
    the path includes the terminal endpoint/junction node at each end.
    """
    node_types = graph._node_types
    adj = graph._adj
    if not node_types:
        return []

    special_nodes = {
        n for n, t in node_types.items()
        if t in ("endpoint", "junction")
    }

    if not special_nodes:
        return [list(node_types)]

    visited_edges: set = set()
    branches: List[List[NodeKey]] = []

    for start in special_nodes:
        for neighbor in adj.get(start, ()):
            edge = (start, neighbor) if start < neighbor else (neighbor, start)
            if edge in visited_edges:
                continue

            path: List[NodeKey] = [start, neighbor]
            visited_edges.add(edge)
            current = neighbor

            while current not in special_nodes:
                next_node: Optional[NodeKey] = None
                for cand in adj.get(current, ()):
                    cand_edge = (
                        (current, cand) if current < cand else (cand, current)
                    )
                    if cand_edge in visited_edges:
                        continue
                    visited_edges.add(cand_edge)
                    next_node = cand
                    break
                if next_node is None:
                    break
                path.append(next_node)
                current = next_node

            branches.append(path)

    return branches


def _branch_direction(branch: List[NodeKey]) -> Tuple[float, float]:
    """Compute the average direction vector of a branch."""
    if len(branch) < 2:
        return (0.0, 0.0)
    dx = float(branch[-1][0] - branch[0][0])
    dy = float(branch[-1][1] - branch[0][1])
    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-9:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def _angle_between(
    d1: Tuple[float, float], d2: Tuple[float, float],
) -> float:
    """Angle in degrees between two unit-length direction vectors."""
    dot = d1[0] * d2[0] + d1[1] * d2[1]
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.degrees(math.acos(dot))


def assign_branches_to_series(
    branches: List[List[NodeKey]],
    n_series: int,
) -> List[List[List[NodeKey]]]:
    """Group branches into ``n_series`` continuous curves.

    Uses greedy angle-based assignment: at each junction, assign outgoing
    branches to series by tangential continuity (minimize angle change
    between branch directions).
    """
    series: List[List[List[NodeKey]]] = [[] for _ in range(n_series)]

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

    series_dirs: List[Optional[Tuple[float, float]]] = [None] * n_series

    for idx in sorted_indices[:n_series]:
        si = idx % n_series
        series[si].append(branches[idx])
        series_dirs[si] = directions[idx]

    for idx in sorted_indices[n_series:]:
        branch_dir = directions[idx]
        best_series = 0
        best_angle = 360.0
        for si in range(n_series):
            current_dir = series_dirs[si]
            if current_dir is None:
                angle = 180.0
            else:
                angle = _angle_between(branch_dir, current_dir)
            if angle < best_angle:
                best_angle = angle
                best_series = si
        series[best_series].append(branches[idx])
        series_dirs[best_series] = branch_dir

    return series


def branches_to_mask(
    branches: List[List[NodeKey]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Render branch coordinates back to a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    for branch in branches:
        for x, y in branch:
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = 1
    return mask
