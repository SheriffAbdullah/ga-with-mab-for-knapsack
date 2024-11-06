class KnapsackProblem:
    def __init__(self, items, weight_limit):
        """
        Initializes the Knapsack problem.

        Args:
            items (list of tuples): Each item is represented as a tuple (weight, value).
            weight_limit (int): The maximum weight capacity of the knapsack.
        """
        self.items = items
        self.weight_limit = weight_limit

    def evaluate_solution(self, solution):
        """
        Evaluates the fitness of a solution.

        Args:
            solution (list of int): Binary list indicating which items are included (1) or not (0).

        Returns:
            int: The total value of the selected items if the solution is valid, otherwise 0.
        """
        total_weight = sum(item[0] for item, include in zip(self.items, solution) if include)
        total_value = sum(item[1] for item, include in zip(self.items, solution) if include)

        if total_weight <= self.weight_limit:
            return total_value
        else:
            return 0  # Invalid solution if it exceeds weight limit
