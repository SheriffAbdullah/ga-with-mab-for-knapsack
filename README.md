
# 🧬 RL-Optimized Genetic Algorithm for Knapsack Problem

**Hosted App**: [https://ga-with-mab-for-knapsack.streamlit.app](https://ga-with-mab-for-knapsack.streamlit.app)

### Project Structure

- app.py: Main application file, handling the UI, input collection, and displaying results.
- knapsack.py: Contains the KnapsackProblem class, which defines the problem structure and solution evaluation.
- genetic_algorithm.py: Implements the core GA logic, including selection, crossover, mutation, and logging of each generation’s metrics.
- mab_agent.py: Defines the MAB agent responsible for adaptively tuning the mutation rate using the UCB1 (Upper Confidence Bound) formula.
- environment.py: Coordinates the GA and MAB to create an interactive environment where RL optimizes the GA’s mutation rate.

## 🔍 Overview

Using **Reinforcement Learning (RL)** to optimize the **mutation rate** in a **Genetic Algorithm (GA)** for the Knapsack Problem. A **Multi-Armed Bandit (MAB)** approach tunes the mutation rate dynamically.

### 💡 Key Concepts

- **Reward Mechanism**: Based on improvement in best fitness between generations.
  - Positive reward for fitness gain, negative for loss.
  - Helps MAB learn effective mutation rates over time.

- **Exploration vs. Exploitation**: Using **Upper Confidence Bound (UCB1)** to select mutation rates.
  - Formula: UCB1 = (Total Reward for Action / Times Action Chosen) + sqrt((2 * ln(Total Trials)) / Times Action Chosen)

### 🔧 Genetic Algorithm Components

- **Crossover Method**: Single-point crossover for combining parent genes.
- **Selection Strategy**: Roulette wheel selection based on fitness.
- **Mutation Strategy**: Bit-flip mutation with rate optimized by MAB.

### 🚀 Future Expansion

Extend MAB optimization to:
- **Mutation Policy** (e.g., bit-flip, swap)
- **Crossover Method** (e.g., single-point, two-point, uniform)
- **Selection Strategy** (e.g., roulette, tournament)

Each could use its own MAB agent for further optimization.

### 📊 Analysis

1. **Best and Average Fitness over Generations**
- Observation: Best fitness stabilizes quickly, indicating early convergence. Average fitness fluctuates.
![4a6110588c7f6e4e76e2e3cb0923bcbaad51e37876bc3da15d94f345](https://github.com/user-attachments/assets/16a960e9-6786-4e9b-be94-d80605911db1)

2. **Mutation Rate over Generations**
   - Observation: Initial mutation rate increase, then oscillations to balance exploration and exploitation.
![eac742542ee21e41ae20f5b7dae11e7cbd8a9b083c1eae5e410401b8](https://github.com/user-attachments/assets/101fef4a-c773-422f-9117-183f71b9d40b)
   
6. **Reward per Generation**
   - Observation: High initial fluctuations in rewards, gradually stabilizing.
   - Insight: High early rewards indicate fitness improvements. Negative rewards signal adjustments in mutation rates. Stability over time shows effective mutation rate learning.
![c61c1f7d03ac8ee5552b7dbeb2df13d0ea3000a283995c6754372674](https://github.com/user-attachments/assets/f98e3daa-3ad9-45b0-be08-fbec42d700e3)

### 🏁 Conclusion
   - We could use a more complex problem to better exploit the RL Agent.
