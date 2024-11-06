from genetic_algorithm import GeneticAlgorithm
from mab_agent import MABAgent

class KnapsackGAEnvironment:
    def __init__(self, problem, generations=50, pop_size_range=(10, 50)):
        """
        Initializes the environment to run the Genetic Algorithm with MAB-based mutation rate tuning.

        Args:
            problem (KnapsackProblem): The knapsack problem instance to optimize.
            generations (int): Number of generations to run the genetic algorithm.
            pop_size_range (tuple of int): Range for dynamic population size adjustments.
        """
        self.problem = problem
        self.ga = GeneticAlgorithm(problem, pop_size=pop_size_range[0])  # GA with initial population size
        self.mab_agent = MABAgent()  # MAB agent for tuning mutation rate
        self.generations = generations  # Total number of generations to run
        self.pop_size_range = pop_size_range  # Minimum and maximum allowed population sizes
        self.logs = []  # Logs for overall environment performance

    def adjust_population_size(self, fitness_improvement):
        """
        Adjusts population size based on fitness improvement trends.

        Args:
            fitness_improvement (float): Improvement in fitness from previous generation.
        """
        if fitness_improvement < 0.01:  # Threshold for minimal improvement
            self.ga.pop_size = min(self.ga.pop_size + 5, self.pop_size_range[1])
        elif fitness_improvement > 0.05:  # Threshold for significant improvement
            self.ga.pop_size = max(self.ga.pop_size - 5, self.pop_size_range[0])

        # Reinitialize population if size has changed
        self.ga.population = self.ga.initialize_population()[:self.ga.pop_size]

    def run_episode(self):
        """
        Runs the entire episode, executing the genetic algorithm over the specified number of generations.
        Uses the MAB agent to dynamically adjust the mutation rate and logs results.

        Returns:
            tuple: Best solution found and its fitness (total value).
        """
        best_solution = None
        best_fitness = 0

        for generation in range(self.generations):
            # Select mutation rate from MAB agent
            mutation_rate = self.mab_agent.select_action()
            self.ga.mutation_rate = mutation_rate

            # Track previous best fitness to calculate fitness improvement
            prev_best_fitness = best_fitness

            # Run a generation of the genetic algorithm
            self.ga.run_generation(generation)

            # Calculate the best fitness in the current generation
            fitness_values = [self.ga.fitness(individual) for individual in self.ga.population]
            generation_best_fitness = max(fitness_values)
            generation_best_solution = self.ga.population[fitness_values.index(generation_best_fitness)]

            # Update the best solution and fitness if there's an improvement
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_solution = generation_best_solution

            # Calculate fitness improvement and adjust population size if needed
            fitness_improvement = best_fitness - prev_best_fitness
            self.adjust_population_size(fitness_improvement)

            # Calculate reward as the improvement in best fitness and update MAB
            reward = generation_best_fitness - prev_best_fitness
            self.mab_agent.update_action_reward(mutation_rate, reward, generation)

            # Log metrics for the environment at each generation
            self.logs.append({
                "Generation": generation,
                "Best Fitness": best_fitness,
                "Average Fitness": sum(fitness_values) / len(fitness_values),
                "Mutation Rate": mutation_rate,
                "Population Size": self.ga.pop_size,
                "Reward": reward
            })

        return best_solution, best_fitness

    def get_logs(self):
        """
        Retrieves the logs of the environment for analysis or visualization.

        Returns:
            list of dict: Logs containing generation, best fitness, average fitness, mutation rate, population size, and reward.
        """
        return self.logs
