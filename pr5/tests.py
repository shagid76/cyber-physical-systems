import unittest
import random
from tsp_genetic import (
    CITY_COUNT, initialize_population, evaluate_fitness, crossover,
    mutate, evaluate_population, tournament_selection, get_best
)

class TestGeneticAlgorithm(unittest.TestCase):

    def test_initialize_population_size(self):
        population = initialize_population()
        self.assertEqual(len(population), 50)
        for individual in population:
            self.assertEqual(len(individual), CITY_COUNT)
            self.assertEqual(sorted(individual), list(range(CITY_COUNT)))

    def test_evaluate_fitness_positive(self):
        population = initialize_population()
        fitness = evaluate_fitness(population[0])
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0)

    def test_evaluate_population_length(self):
        population = initialize_population()
        fitnesses = evaluate_population(population)
        self.assertEqual(len(fitnesses), len(population))
        self.assertTrue(all(isinstance(f, float) for f in fitnesses))

    def test_tournament_selection_returns_individual(self):
        population = initialize_population()
        fitnesses = evaluate_population(population)
        selected = tournament_selection(population, fitnesses)
        self.assertIn(sorted(selected), [sorted(ind) for ind in population])

    def test_crossover_produces_valid_child(self):
        population = initialize_population()
        p1, p2 = population[0], population[1]
        child = crossover(p1, p2)
        self.assertEqual(len(child), CITY_COUNT)
        self.assertEqual(sorted(child), list(range(CITY_COUNT)))

    def test_mutation_preserves_genes(self):
        individual = list(range(CITY_COUNT))
        mutated = mutate(individual.copy())
        self.assertEqual(sorted(mutated), list(range(CITY_COUNT)))

    def test_get_best_returns_valid_result(self):
        population = initialize_population()
        fitnesses = evaluate_population(population)
        best, dist = get_best(population, fitnesses)
        self.assertEqual(len(best), CITY_COUNT)
        self.assertIsInstance(dist, float)
        self.assertGreater(dist, 0)


if __name__ == '__main__':
    unittest.main()