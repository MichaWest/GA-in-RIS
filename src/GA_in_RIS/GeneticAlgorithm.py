from typing import Callable

import numpy as np

from src.GA_in_RIS.Metasurface import MetasurfaceConfig, ScatteringCalculator


class GeneticAlgorithmOptimizer:
    def __init__(self,
                 config: MetasurfaceConfig,
                 population_size: int = 100,
                 max_generations: int = 350,
                 pc: float = 0.85, # Вероятность кроссовера
                 pm: float = 0.01, # Вероятность мутации
                 ):
        self.config = config
        self.population_size = population_size
        self.max_generation_size = max_generations

        self.pc = pc
        self.pm = pm

        self.M = config.M
        self.N = config.N

        self.calculator = ScatteringCalculator(config)
        self.history = []  # История значений целевой функции

    def optimize(self,
                 fitness_func: Callable):
        population = self.initialize_population()

        best_individual = None
        best_fitness = float('inf')
        self.history = []

        for generation in range(self.max_generation_size):
            fitness_values = self.evaluate_population(population, fitness_func)

            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            self.history.append(best_fitness)

            parents = self.selection(population, fitness_values)

            children = self.crossover(parents)

            population = self.mutation(children)

        return best_individual, np.array(self.history)



    def initialize_population(self) -> np.ndarray:
        size = (self.population_size, self.M, self.N)
        population = np.random.randint(0, 2, size=size, dtype=np.int8)
        return population

    '''
    Вычисление значений приспособленности
    '''
    def evaluate_population(self,
                            population: np.ndarray,
                            fitness_func: Callable) -> np.ndarray:
        fitness_values = np.zeros(self.population_size)

        for i in range(self.population_size):
            coding_matrix = population[i]
            fitness_values[i] = fitness_func(coding_matrix)

        return fitness_values

    '''
    Репродукция
    '''
    def selection(self, population: np.ndarray,
                  fitness_values: np.ndarray) -> np.ndarray:
        inverse_fitness = -fitness_values
        total_fitness = np.sum(inverse_fitness)

        if total_fitness <= 1e-10:
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = inverse_fitness / total_fitness

        parents_indices = np.random.choice(
            len(population),
            size=len(population),
            p=probabilities,
            replace=True
        )

        parents = population[parents_indices]
        return parents

    '''
    Кроссинговер
    '''
    def crossover(self, population):
        new_population = population.copy()
        total_cells = self.M * self.N

        indices = list(range(self.population_size))

        pairs = []
        for i in range(0, len(indices) - 1, 2):
            pairs.append((indices[i], indices[i + 1]))

        for idx1, idx2 in pairs:
            if np.random.random() < self.pc:
                crossover_point = np.random.randint(1, total_cells)

                p1_flat = new_population[idx1].flatten()
                p2_flat = new_population[idx2].flatten()

                child1_flat = np.concatenate([
                    p1_flat[:crossover_point],
                    p2_flat[crossover_point:]
                ])

                child2_flat = np.concatenate([
                    p2_flat[:crossover_point],
                    p1_flat[crossover_point:]
                ])

                new_population[idx1] = child1_flat.reshape((self.M, self.N))
                new_population[idx2] = child2_flat.reshape((self.M, self.N))

        return new_population

    '''
    Мутация 
    '''
    def mutation(self, population):
        new_population = population.copy()
        total_cells = self.M * self.N

        for i in range(self.population_size):
            for j in range(total_cells):
                if np.random.random() < self.pm:
                    row = j // self.N
                    col = j % self.N

                    new_population[i, row, col] = 1 - new_population[i, row, col]

        return new_population