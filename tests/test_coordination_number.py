"""Tests for Delaunay-based coordination numbers."""

from __future__ import annotations

import unittest

from src.coordination_number import coordination_numbers, delaunay_edges


class CoordinationNumberTests(unittest.TestCase):
    def test_triangle_gives_two_neighbors_per_agent(self) -> None:
        positions = {
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (0.0, 1.0),
        }
        self.assertEqual(coordination_numbers(positions), {1: 2, 2: 2, 3: 2})

    def test_square_has_four_boundary_edges(self) -> None:
        edges = delaunay_edges(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ]
        )
        self.assertEqual(len(edges), 5)

    def test_collinear_points_fall_back_to_chain_neighbors(self) -> None:
        positions = {
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (2.0, 0.0),
            4: (3.0, 0.0),
        }
        self.assertEqual(coordination_numbers(positions), {1: 1, 2: 2, 3: 2, 4: 1})


if __name__ == "__main__":
    unittest.main()
