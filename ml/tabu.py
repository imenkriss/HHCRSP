"""Tabu Search router used per cluster in the D3 pipeline.

Each cluster is routed as a single tour that starts and ends at the depot.
The search uses a swap neighbourhood with a tabu list on recently swapped
node pairs and the standard aspiration criterion (a tabu move is allowed when
it improves the best known tour). All randomness is seeded; nodes are points
given as ``(x, y)`` pairs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Point = Sequence[float]


def _distance_matrix(points: np.ndarray, depot: Point) -> np.ndarray:
    coords = np.vstack([np.asarray(depot, dtype=float), points])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def tour_cost(order: Sequence[int], points: Sequence[Point], depot: Point) -> float:
    """Closed-tour length: depot -> order -> depot (Euclidean)."""
    pts = np.asarray(points, dtype=float)
    matrix = _distance_matrix(pts, depot)
    cost = 0.0
    current = 0
    for node in order:
        cost += matrix[current, node + 1]
        current = node + 1
    cost += matrix[current, 0]
    return float(cost)


def _order_cost(order: Sequence[int], matrix: np.ndarray) -> float:
    cost = matrix[0, order[0] + 1] if order else 0.0
    for a, b in zip(order, order[1:]):
        cost += matrix[a + 1, b + 1]
    if order:
        cost += matrix[order[-1] + 1, 0]
    return float(cost)


def _node(order: Sequence[int], position: int) -> int:
    """Matrix index of the tour node at ``position`` (depot = 0 at the ends)."""
    if 0 <= position < len(order):
        return order[position] + 1
    return 0


def swap_delta(order: Sequence[int], i: int, j: int, matrix: np.ndarray) -> float:
    """O(1) change in closed-tour length when swapping positions ``i`` and ``j``."""
    prev_i, node_i, next_i = _node(order, i - 1), _node(order, i), _node(order, i + 1)
    prev_j, node_j, next_j = _node(order, j - 1), _node(order, j), _node(order, j + 1)
    if j == i + 1:
        removed = matrix[prev_i, node_i] + matrix[node_j, next_j]
        added = matrix[prev_i, node_j] + matrix[node_i, next_j]
        return float(added - removed)
    removed = (
        matrix[prev_i, node_i]
        + matrix[node_i, next_i]
        + matrix[prev_j, node_j]
        + matrix[node_j, next_j]
    )
    added = (
        matrix[prev_i, node_j]
        + matrix[node_j, next_i]
        + matrix[prev_j, node_i]
        + matrix[node_i, next_j]
    )
    return float(added - removed)


def nearest_neighbour_order(points: Sequence[Point], depot: Point) -> list[int]:
    """Greedy nearest-neighbour seed tour (deterministic)."""
    pts = np.asarray(points, dtype=float)
    matrix = _distance_matrix(pts, depot)
    n = pts.shape[0]
    unvisited = set(range(n))
    order: list[int] = []
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda node: matrix[current, node + 1])
        order.append(nxt)
        unvisited.discard(nxt)
        current = nxt + 1
    return order


def tabu_search(
    points: Sequence[Point],
    depot: Point = (0.0, 0.0),
    iterations: int = 100,
    tabu_tenure: int = 10,
    seed: int = 0,
) -> dict:
    """Route ``points`` from ``depot`` with Tabu Search (swap neighbourhood).

    ``seed`` is accepted for API stability; the neighbourhood scan is
    deterministic (nearest-neighbour seed + best improving swap), so results
    do not depend on it.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]

    if n == 0:
        return {"order": [], "cost": 0.0, "start_cost": 0.0, "iterations": 0}
    if n == 1:
        return {
            "order": [0],
            "cost": tour_cost([0], pts, depot),
            "start_cost": tour_cost([0], pts, depot),
            "iterations": 0,
        }

    matrix = _distance_matrix(pts, depot)
    order = nearest_neighbour_order(pts, depot)
    start_cost = _order_cost(order, matrix)
    best_order = list(order)
    best_cost = start_cost
    current_cost = start_cost
    tabu: dict[tuple[int, int], int] = {}

    for step in range(int(iterations)):
        best_move = None
        best_move_cost = float("inf")
        best_delta = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                delta = swap_delta(order, i, j, matrix)
                cost = current_cost + delta
                is_tabu = tabu.get((i, j), -1) > step
                if is_tabu and cost >= best_cost:
                    continue
                if cost < best_move_cost:
                    best_move_cost = cost
                    best_move = (i, j)
                    best_delta = delta
        if best_move is None:
            break
        i, j = best_move
        order[i], order[j] = order[j], order[i]
        current_cost += best_delta
        tabu[(i, j)] = step + int(tabu_tenure)
        tabu[(j, i)] = step + int(tabu_tenure)
        if current_cost < best_cost:
            best_cost = current_cost
            best_order = list(order)

    return {
        "order": best_order,
        "cost": best_cost,
        "start_cost": start_cost,
        "iterations": int(iterations),
    }
