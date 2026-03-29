"""Utilities to compute Delaunay-based coordination numbers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


Point = Tuple[float, float]
Edge = Tuple[int, int]
AgentPositions = Dict[int, Point]


@dataclass(frozen=True)
class Triangle:
    """Triangle defined by indices into a point array."""

    a: int
    b: int
    c: int

    def edges(self) -> Tuple[Edge, Edge, Edge]:
        """Return sorted triangle edges."""
        return (
            _sorted_edge(self.a, self.b),
            _sorted_edge(self.b, self.c),
            _sorted_edge(self.c, self.a),
        )


def _sorted_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _is_collinear(points: Sequence[Point]) -> bool:
    if len(points) < 3:
        return True
    array = np.asarray(points, dtype=float)
    centered = array - array.mean(axis=0)
    return np.linalg.matrix_rank(centered) < 2


def _fallback_edges(points: Sequence[Point]) -> Set[Edge]:
    """Connect adjacent points along the dominant axis for degenerate frames."""
    if len(points) < 2:
        return set()
    if len(points) == 2:
        return {(0, 1)}

    array = np.asarray(points, dtype=float)
    spread_x = float(np.ptp(array[:, 0]))
    spread_y = float(np.ptp(array[:, 1]))
    axis = 0 if spread_x >= spread_y else 1
    order = np.argsort(array[:, axis], kind="mergesort")
    return {
        _sorted_edge(int(order[index]), int(order[index + 1]))
        for index in range(len(order) - 1)
    }


def _circumcircle_contains(
    triangle: Triangle, point_index: int, points: Sequence[Point], eps: float = 1e-9
) -> bool:
    ax, ay = points[triangle.a]
    bx, by = points[triangle.b]
    cx, cy = points[triangle.c]
    px, py = points[point_index]

    matrix = np.array(
        [
            [ax - px, ay - py, (ax - px) ** 2 + (ay - py) ** 2],
            [bx - px, by - py, (bx - px) ** 2 + (by - py) ** 2],
            [cx - px, cy - py, (cx - px) ** 2 + (cy - py) ** 2],
        ],
        dtype=float,
    )
    orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    det = float(np.linalg.det(matrix))
    return det > eps if orient > 0 else det < -eps


def delaunay_edges(points: Sequence[Point]) -> Set[Edge]:
    """Return Delaunay edges for a set of 2D points."""
    if len(points) < 2:
        return set()
    if len(points) == 2:
        return {(0, 1)}
    if _is_collinear(points):
        return _fallback_edges(points)

    base_points = [tuple(map(float, point)) for point in points]
    xs = [point[0] for point in base_points]
    ys = [point[1] for point in base_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    delta = max(max_x - min_x, max_y - min_y, 1.0)
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)

    supertriangle = [
        (mid_x - 20.0 * delta, mid_y - delta),
        (mid_x, mid_y + 20.0 * delta),
        (mid_x + 20.0 * delta, mid_y - delta),
    ]
    all_points = base_points + supertriangle
    super_indices = (
        len(base_points),
        len(base_points) + 1,
        len(base_points) + 2,
    )

    triangles: List[Triangle] = [Triangle(*super_indices)]

    for point_index in range(len(base_points)):
        bad_triangles = [
            triangle
            for triangle in triangles
            if _circumcircle_contains(triangle, point_index, all_points)
        ]
        if not bad_triangles:
            continue

        edge_counter = Counter(
            edge for triangle in bad_triangles for edge in triangle.edges()
        )
        boundary_edges = [
            edge for edge, count in edge_counter.items() if count == 1
        ]
        triangles = [
            triangle for triangle in triangles if triangle not in bad_triangles
        ]
        for edge in boundary_edges:
            triangles.append(Triangle(edge[0], edge[1], point_index))

    valid_triangles = [
        triangle
        for triangle in triangles
        if triangle.a < len(base_points)
        and triangle.b < len(base_points)
        and triangle.c < len(base_points)
    ]
    if not valid_triangles:
        return _fallback_edges(points)

    edges = {edge for triangle in valid_triangles for edge in triangle.edges()}
    return edges or _fallback_edges(points)


def coordination_numbers(positions: AgentPositions) -> Dict[int, int]:
    """Compute the number of Delaunay neighbors for each agent."""
    agent_ids = sorted(positions)
    frame_points = [positions[agent_id] for agent_id in agent_ids]
    edges = delaunay_edges(frame_points)
    neighbors: Dict[int, Set[int]] = {agent_id: set() for agent_id in agent_ids}

    for i, j in edges:
        agent_i = agent_ids[i]
        agent_j = agent_ids[j]
        neighbors[agent_i].add(agent_j)
        neighbors[agent_j].add(agent_i)

    return {agent_id: len(adjacent) for agent_id, adjacent in neighbors.items()}


def mean_pair_distance(points: Iterable[Point]) -> float:
    """Return the mean pairwise distance for a small set of points."""
    values = list(points)
    if len(values) < 2:
        return 0.0
    total = 0.0
    count = 0
    for index, point_a in enumerate(values):
        for point_b in values[index + 1 :]:
            total += hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
            count += 1
    return total / count
