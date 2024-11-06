import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from knapsack import KnapsackProblem
from environment import KnapsackGAEnvironment

# Title and Knapsack Problem Input
st.title("RL-Optimized Genetic Algorithm for Knapsack Problem")

# Placeholder items with diverse weights and values
default_items = [
    (5, 10), (10, 40), (8, 15), (12, 30), (3, 5), (7, 25), (9, 20), (14, 35),
    (6, 12), (4, 8), (11, 22), (13, 40), (2, 3), (15, 50), (1, 1), (20, 80),
    (18, 60), (6, 18), (4, 7), (16, 45)
]

default_weight_limit = 50

# Checkbox for using default items
use_default_items = st.checkbox("Use default items for testing (20 items with varied weights and values)")

# If using defaults, display the default items
if use_default_items:
    st.write("Using default items for optimization:")
    items = default_items
    weight_limit = default_weight_limit
else:
    # Allow user input for number of items and weight limit
    num_items = st.number_input("Number of Items", min_value=1, step=1, value=1)
    weight_limit = st.number_input("Knapsack Weight Limit", min_value=1)

    # Dynamic Input for Item Weights and Values
    st.write("Enter the weight and value for each item:")
    weights = []
    values = []

    for i in range(num_items):
        cols = st.columns(2)
        with cols[0]:
            weight = st.number_input(f"Weight of item {i+1}", min_value=1, key=f"weight_{i}")
            weights.append(weight)
        with cols[1]:
            value = st.number_input(f"Value of item {i+1}", min_value=1, key=f"value_{i}")
            values.append(value)

    items = list(zip(weights, values))

# Display the items and weight limit being used
st.write("Items (Weight, Value):")
items_df = pd.DataFrame(items, columns=["Weight", "Value"])
st.table(items_df)
st.write(f"Knapsack Weight Limit: {weight_limit}")

# Run the optimization process when button is clicked
if st.button("Run Optimization"):
    knapsack = KnapsackProblem(items, weight_limit)
    environment = KnapsackGAEnvironment(knapsack, generations=100)

    # Run Optimization and Collect Best Solution
    best_solution, best_fitness = environment.run_episode()

    # Display Best Solution
    st.subheader("Optimization Results")
    st.metric(label="Best Total Value", value=best_fitness)

    # Show Selected Items in Best Solution
    st.write("### Items Included in the Best Solution:")
    selected_items = [(i + 1, items[i][0], items[i][1]) for i in range(len(best_solution)) if best_solution[i] == 1]
    selected_items_df = pd.DataFrame(selected_items, columns=["Item #", "Weight", "Value"])

    if not selected_items:
        st.write("No items selected (Knapsack is empty).")
    else:
        st.table(selected_items_df)

        # Summary of Total Weight and Value
        total_weight = sum(items[i][0] for i in range(len(best_solution)) if best_solution[i] == 1)
        total_value = sum(items[i][1] for i in range(len(best_solution)) if best_solution[i] == 1)
        st.write(f"**Total Weight:** {total_weight} (Limit: {weight_limit})")
        st.write(f"**Total Value:** {total_value}")

    # Sidebar for Logs and Plots Section
    st.sidebar.subheader("Logs and Plots")

    # Display Logs as Table in Expander
    with st.sidebar.expander("View Logs", expanded=False):
        ga_logs = pd.DataFrame(environment.get_logs())
        mab_logs = pd.DataFrame(environment.mab_agent.get_logs())
        
        st.write("**Genetic Algorithm Logs**")
        st.dataframe(ga_logs)
        
        st.write("**MAB Agent Logs**")
        st.dataframe(mab_logs)

    # Plot Best and Average Fitness over Generations
    st.sidebar.subheader("Best and Average Fitness over Generations")
    fig1, ax1 = plt.subplots()
    ax1.plot(ga_logs["Generation"], ga_logs["Best Fitness"], label="Best Fitness")
    ax1.plot(ga_logs["Generation"], ga_logs["Average Fitness"], label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    st.sidebar.pyplot(fig1)

    # Plot Mutation Rate over Generations
    st.sidebar.subheader("Mutation Rate over Generations")
    fig2, ax2 = plt.subplots()
    ax2.plot(mab_logs["Generation"], mab_logs["Action"], label="Mutation Rate", color="orange")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mutation Rate")
    st.sidebar.pyplot(fig2)

    # Plot Reward per Generation for MAB
    st.sidebar.subheader("Reward per Generation")
    fig3, ax3 = plt.subplots()
    ax3.plot(mab_logs["Generation"], mab_logs["Reward"], label="Reward", color="green")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Reward")
    st.sidebar.pyplot(fig3)
