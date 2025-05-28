import random
import numpy as np

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, city):
        return np.hypot(self.x - city.x, self.y - city.y)

class AntColony:
    def __init__(self, cities, n_ants, n_iterations, decay, alpha=1, beta=2):
        self.cities = cities
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = [[1.0 for _ in cities] for _ in cities]
        self.distances = [[city.distance_to(other) for other in cities] for city in cities]

    def run(self):
        all_time_shortest_path = ([], float('inf'))

        for iteration in range(self.n_iterations):
            all_paths = self.generate_all_paths()
            self.spread_pheromone(all_paths, self.n_ants)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.evaporate_pheromone()

        return all_time_shortest_path

    def generate_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.generate_path()
            all_paths.append((path, self.path_distance(path)))
        return all_paths

    def generate_path(self):
        path = []
        visited = set()
        current = random.randint(0, len(self.cities) - 1)
        path.append(current)
        visited.add(current)

        for _ in range(len(self.cities) - 1):
            next_city = self.pick_next_city(current, visited)
            path.append(next_city)
            visited.add(next_city)
            current = next_city

        return path

    def pick_next_city(self, current, visited):
        pheromone = self.pheromone[current]
        distances = self.distances[current]
        probs = []
        for i in range(len(self.cities)):
            if i in visited:
                probs.append(0)
            else:
                probs.append((pheromone[i] ** self.alpha) * ((1 / distances[i]) ** self.beta))

        total = sum(probs)
        if total == 0:
            candidates = [i for i in range(len(self.cities)) if i not in visited]
            return random.choice(candidates)
        probs = [p / total for p in probs]
        return np.random.choice(len(self.cities), p=probs)

    def path_distance(self, path):
        total = 0
        for i in range(len(path)):
            total += self.distances[path[i]][path[(i + 1) % len(path)]]
        return total

    def spread_pheromone(self, all_paths, n_ants):
        for path, dist in all_paths:
            for i in range(len(path)):
                self.pheromone[path[i]][path[(i + 1) % len(path)]] += 1.0 / dist

    def evaporate_pheromone(self):
        for i in range(len(self.pheromone)):
            for j in range(len(self.pheromone)):
                self.pheromone[i][j] *= (1 - self.decay)
