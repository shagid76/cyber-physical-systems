import unittest
import numpy as np
from forest_fire import create_forest, update_forest, EMPTY, TREE, BURNING

class TestForestFireModel(unittest.TestCase):

    def test_forest_initialization(self):
        size = 10
        fire_spots = [(5, 5), (2, 3)]
        forest, burning_time = create_forest(size, fire_spots)

        self.assertEqual(forest.shape, (size, size))
        self.assertTrue(np.all(forest[forest != BURNING] == TREE))
        for x, y in fire_spots:
            self.assertEqual(forest[x, y], BURNING)
            self.assertEqual(burning_time[x, y], 0)

    def test_burning_to_empty(self):
        size = 5
        forest = np.full((size, size), EMPTY)
        forest[2, 2] = BURNING
        burning_time = np.zeros((size, size), dtype=int)
        burning_time[2, 2] = 3

        new_forest, new_burning_time = update_forest(forest, burning_time, P_burn=0.5, T_burn=2)
        self.assertEqual(new_forest[2, 2], EMPTY)

    def test_tree_ignites(self):
        size = 3
        forest = np.array([
            [TREE, TREE, TREE],
            [TREE, BURNING, TREE],
            [TREE, TREE, TREE]
        ])
        burning_time = np.zeros((size, size), dtype=int)

        new_forest, new_burning_time = update_forest(forest, burning_time, P_burn=1.0, T_burn=2)

        # All neighbors of the center should catch fire (except the center which is already burning)
        for x in range(size):
            for y in range(size):
                if (x, y) != (1, 1):
                    self.assertEqual(new_forest[x, y], BURNING)

    def test_no_fire_spread_without_burning_neighbors(self):
        size = 3
        forest = np.full((size, size), TREE)
        burning_time = np.zeros((size, size), dtype=int)

        new_forest, _ = update_forest(forest, burning_time, P_burn=1.0, T_burn=2)
        self.assertTrue(np.all(new_forest == TREE))
