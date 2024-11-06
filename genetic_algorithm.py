import random
import numpy as np  # For diversity calculations
import heapq  # To efficiently track top solutions

class GeneticAlgorithm:
    def __init__(self, problem, pop_size=20, mutation_rate=0.1, repopulate_threshold=0.3):
        """
        Initializes the Genetic Algorithm.

        Args:
            problem (KnapsackProblem): The knapsack problem instance to solve.
            pop_size (int): Initial number of solutions in the population.
            mutation_rate (float): Probability of mutation for each gene in a solution.
            repopulate_threshold (float): Diversity threshold for repopulation.
        """
        self.problem = problem
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.repopulate_threshold = repopulate_threshold
        self.population = self.initialize_population()
        self.logs = []

    def initialize_population(self):
        return [[random.choice([0, 1]) for _ in self.problem.items] for _ in range(self.pop_size)]

    def fitness(self, solution):
        return self.problem.evaluate_solution(solution)

    def calculate_diversity(self):
        unique_solutions = {tuple(individual) for individual in self.population}
        return len(unique_solutions) / len(self.population)

    def repopulate(self):
        num_replacements = int(self.pop_size * 0.3)
        new_individuals = self.initialize_population()[:num_replacements]
        self.population = random.sample(self.population, self.pop_size - num_replacements) + new_individuals

    def selection(self):
        total_fitness = sum(self.fitness(individual) for individual in self.population)
        if total_fitness == 0:
            return random.choices(self.population, k=2)
        selection_probs = [self.fitness(individual) / total_fitness for individual in self.population]
        return tuple(random.choices(self.population, weights=selection_probs, k=2))

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]

    def mutate(self, solution):
        for i in range(len(solution)):
            if random.random() < self.mutation_rate:
                solution[i] = 1 - solution[i]

    def run_generation(self, generation):
        if self.calculate_diversity() < self.repopulate_threshold:
            self.repopulate()

        new_population = []

        for _ in range(self.pop_size // 2):
            parent1, parent2 = self.selection()
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])

        self.population = new_population
        fitness_values = [self.fitness(individual) for individual in self.population]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        self.log_generation(generation, best_fitness, avg_fitness)

    def log_generation(self, generation, best_fitness, avg_fitness):
        self.logs.append({
            "Generation": generation,
            "Best Fitness": best_fitness,
            "Average Fitness": avg_fitness,
            "Mutation Rate": self.mutation_rate,
            "Population Size": len(self.population)
        })
