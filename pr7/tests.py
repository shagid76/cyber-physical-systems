import unittest

from ant import City, AntColony


class TestAntColony(unittest.TestCase):
    def setUp(self):
        self.cities = [City(0, 0), City(0, 1), City(1, 0)]
        self.colony = AntColony(self.cities, n_ants=3, n_iterations=1, decay=0.1)

    def test_distance(self):
        self.assertAlmostEqual(self.cities[0].distance_to(self.cities[1]), 1.0)

    def test_generate_path_length(self):
        path = self.colony.generate_path()
        self.assertEqual(len(path), len(self.cities))
        self.assertEqual(len(set(path)), len(self.cities))

    def test_pick_next_city_excludes_visited(self):
        visited = {0, 1}
        next_city = self.colony.pick_next_city(0, visited)
        self.assertEqual(next_city, 2)

    def test_path_distance(self):
        path = [0, 1, 2]
        distance = self.colony.path_distance(path)
        self.assertTrue(distance > 0)

    def test_spread_pheromone_increases_value(self):
        initial_pheromone = self.colony.pheromone[0][1]
        self.colony.spread_pheromone([([0, 1, 2], 3)], 1)
        self.assertGreater(self.colony.pheromone[0][1], initial_pheromone)

    def test_evaporate_pheromone_decreases_value(self):
        self.colony.pheromone[0][1] = 5.0
        self.colony.evaporate_pheromone()
        self.assertLess(self.colony.pheromone[0][1], 5.0)

    def test_all_paths_are_valid(self):
        paths = self.colony.generate_all_paths()
        for path, dist in paths:
            with self.subTest(path=path):
                self.assertEqual(len(path), len(self.cities))
                self.assertEqual(len(set(path)), len(self.cities))
                self.assertTrue(dist > 0)

    def test_all_pheromones_remain_positive(self):
        self.colony.run()
        for row in self.colony.pheromone:
            for value in row:
                self.assertGreaterEqual(value, 0.0)

    def test_run_returns_valid_path(self):
        best_path, best_distance = self.colony.run()
        self.assertEqual(len(best_path), len(self.cities))
        self.assertEqual(len(set(best_path)), len(self.cities))
        self.assertTrue(best_distance > 0)

    def test_no_zero_division_in_probabilities(self):
        self.colony.distances = [[1e-6 if i != j else 0 for j in range(len(self.cities))] for i in
                                 range(len(self.cities))]
        try:
            self.colony.generate_all_paths()
        except ZeroDivisionError:
            self.fail("ZeroDivisionError occurred during path generation")

if __name__ == '__main__':
    unittest.main()