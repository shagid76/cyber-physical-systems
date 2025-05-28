import random
import matplotlib.pyplot as plt
import numpy as np

CITY_COUNT = 10
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.05
ELITE_COUNT = 2

def initialize_population():
    return [random.sample(range(CITY_COUNT), CITY_COUNT) for _ in range(POPULATION_SIZE)]

cities = np.random.rand(CITY_COUNT, 2) * 100

def evaluate_fitness(individual):
    dist = sum(np.linalg.norm(cities[individual[i]] - cities[individual[(i + 1) % CITY_COUNT]]) for i in range(CITY_COUNT))
    return 1 / dist 

def tournament_selection(population, fitnesses, k=5):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(CITY_COUNT), 2))
    child = [None] * CITY_COUNT
    child[start:end] = parent1[start:end]
    fill = [gene for gene in parent2 if gene not in child]
    pos = 0
    for i in range(CITY_COUNT):
        if child[i] is None:
            child[i] = fill[pos]
            pos += 1
    return child

def mutate(individual):
    if random.random() < MUTATION_RATE:
        a, b = random.sample(range(CITY_COUNT), 2)
        individual[a], individual[b] = individual[b], individual[a]
    return individual

def evaluate_population(population):
    return [evaluate_fitness(ind) for ind in population]

def next_generation(population, fitnesses):
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
    new_pop = sorted_pop[:ELITE_COUNT]
    while len(new_pop) < POPULATION_SIZE:
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)
        child = mutate(crossover(p1, p2))
        new_pop.append(child)
    return new_pop

def should_stop(gen):
    return gen >= GENERATIONS

def get_best(population, fitnesses):
    best_idx = np.argmax(fitnesses)
    return population[best_idx], 1 / fitnesses[best_idx]