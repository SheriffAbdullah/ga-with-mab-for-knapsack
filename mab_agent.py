import math

class MABAgent:
    def __init__(self, actions=[x/100 for x in range(0, 30)], exploration_weight=2):
        """
        Initializes the Multi-Armed Bandit (MAB) agent using UCB for action selection.

        Args:
            actions (list of float): List of mutation rates the agent can choose from.
            exploration_weight (float): Controls the exploration-exploitation balance in UCB.
        """
        self.actions = actions  # Possible mutation rates
        self.exploration_weight = exploration_weight  # UCB exploration weight
        self.action_rewards = {action: 0 for action in actions}  # Total rewards for each action
        self.action_counts = {action: 0 for action in actions}   # Number of times each action was chosen
        self.total_trials = 0  # Total number of trials (generations in GA)
        self.logs = []  # Log to store rewards for each action and generation

    def select_action(self):
        """
        Selects an action (mutation rate) using the UCB1 formula.

        Returns:
            float: The chosen mutation rate.
        """
        self.total_trials += 1
        action_ucb_values = {}

        # Calculate UCB value for each action
        for action in self.actions:
            if self.action_counts[action] == 0:
                # If action hasn't been tried, prioritize exploration
                action_ucb_values[action] = float("inf")
            else:
                # Calculate UCB value using average reward and exploration term
                avg_reward = self.action_rewards[action] / self.action_counts[action]
                exploration_bonus = math.sqrt(
                    (self.exploration_weight * math.log(self.total_trials)) / self.action_counts[action]
                )
                action_ucb_values[action] = avg_reward + exploration_bonus

        # Select action with the highest UCB value
        best_action = max(action_ucb_values, key=action_ucb_values.get)
        return best_action

    def update_action_reward(self, action, reward, generation):
        """
        Updates the reward for the chosen action and logs the result.

        Args:
            action (float): The mutation rate that was used.
            reward (float): The reward observed after using the mutation rate.
            generation (int): The generation number for logging.
        """
        # Update reward and count for the chosen action
        self.action_rewards[action] += reward
        self.action_counts[action] += 1

        # Log the action, reward, and generation for analysis
        self.logs.append({
            "Generation": generation,
            "Action": action,
            "Reward": reward,
            "Times Chosen": self.action_counts[action]
        })

    def get_logs(self):
        """
        Retrieves the logs of the MAB agent for analysis or visualization.

        Returns:
            list of dict: Logs containing generation, action, reward, and times chosen.
        """
        return self.logs
