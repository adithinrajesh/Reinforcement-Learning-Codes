import gymnasium as gym
import numpy as np
import pandas as pd
import time

# Create the Frozen Lake environment (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Define the MDP components
states = env.observation_space.n  # Number of states
actions = env.action_space.n  # Number of actions
gamma = 0.9  # Discount factor
theta = 1e-6  # Threshold for policy evaluation convergence

# Initialize the policy randomly (equally likely actions)
policy = np.ones([states, actions]) / actions

# Initialize the state value function with zeros
V = np.zeros(states)

# Function to display the value function as a table
def display_value_table(V):
    value_table = V.reshape((4, 4))
    df = pd.DataFrame(value_table, columns=['Col 0', 'Col 1', 'Col 2', 'Col 3'], 
                      index=['Row 0', 'Row 1', 'Row 2', 'Row 3'])
    print(df)

# Function to display the action value function as a table
def display_action_value_table(Q):
    action_value_tables = Q.reshape((4, 4, actions))
    for a in range(actions):
        print(f"Action Value Table for Action {a}:")
        df = pd.DataFrame(action_value_tables[:, :, a], columns=['Col 0', 'Col 1', 'Col 2', 'Col 3'], 
                          index=['Row 0', 'Row 1', 'Row 2', 'Row 3'])
        print(df)

# Function to print the transition probabilities for the environment
def print_transition_probabilities(env):
    transition_probs = env.unwrapped.P
    for state in transition_probs:
        for action in transition_probs[state]:
            print(f"State {state}, Action {action}:")
            for prob, next_state, reward, done in transition_probs[state][action]:
                print(f"  P={prob}, Next State={next_state}, Reward={reward}, Done={done}")

# Policy Evaluation: Update state value function V(s) for the current policy
def policy_evaluation(policy, V, gamma, theta, iterations=1):
    for i in range(iterations):
        delta = 0
        # Loop over each state
        for state in range(states):
            v = 0
            # Loop over each action
            for action, action_prob in enumerate(policy[state]):
                # Get the transition for this state-action pair
                prob, next_state, reward, done = env.unwrapped.P[state][action][0]
                # Since the environment is non-slippery, the transition probability is always 1
                v += action_prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        # Print the state values after each iteration
        print(f"Iteration {i+1} State Values:")
        display_value_table(V)
    return V

# Function to run the environment with the given policy and print the rewards
def run_policy(env, policy, num_steps=10):
    state, _ = env.reset()
    total_reward = 0
    for step in range(num_steps):
        action = np.argmax(policy[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Step: {step + 1}, State: {state}, Action: {action}, Reward: {reward}")
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    print(f"Total Reward: {total_reward}")

# Print the initial random policy
print("Initial Random Policy:")
policy_mapping = {0: '←', 1: '↓', 2: '→', 3: '↑'}
initial_policy = np.array([policy_mapping[np.argmax(policy[s])] for s in range(states)])
print(initial_policy.reshape((4, 4)))

# Demonstrate the initial random policy
print("\nDemonstrating Initial Random Policy:")
run_policy(env, policy)
time.sleep(3)  # Pause for 3 seconds

# Print the transition probabilities for the initial random policy
print("\nTransition Probabilities (Non-Slippery Environment):")
print_transition_probabilities(env)
time.sleep(3)  # Pause for 3 seconds

# Demonstrate and print state values for initial random policy for two iterations
print("\nDemonstrating Initial State Values for Two Iterations:")
V = policy_evaluation(policy, V, gamma, theta, iterations=4)

# Policy Improvement: Improve the policy based on the current state value function
def policy_improvement(V, gamma):
    policy_stable = True
    for state in range(states):
        old_action = np.argmax(policy[state])
        action_values = np.zeros(actions)
        for action in range(actions):
            # Get the transition for this state-action pair
            prob, next_state, reward, done = env.unwrapped.P[state][action][0]
            # Since the environment is non-slippery, the transition probability is always 1
            action_values[action] = reward + gamma * V[next_state]
        best_action = np.argmax(action_values)
        # Check if the best action has changed
        if old_action != best_action:
            policy_stable = False
        policy[state] = np.eye(actions)[best_action]
    return policy, policy_stable

# Policy Iteration: Alternate between policy evaluation and policy improvement
def policy_iteration(policy, V, gamma, theta):
    iteration = 0
    while True:
        print(f"Policy Iteration Step {iteration}")
        V = policy_evaluation(policy, V, gamma, theta)
        print("State Value Function (V):")
        display_value_table(V)
        print("Policy:")
        current_policy = np.array([policy_mapping[np.argmax(policy[s])] for s in range(states)])
        print(current_policy.reshape((4, 4)))
        
        policy, policy_stable = policy_improvement(V, gamma)
        if policy_stable:
            break
        iteration += 1
    return policy, V

# Perform policy iteration
optimal_policy, optimal_V = policy_iteration(policy, V, gamma, theta)

# Compute the action value function Q(s, a) using the optimal state value function
def compute_action_value_function(V, gamma):
    Q = np.zeros((states, actions))
    for state in range(states):
        for action in range(actions):
            prob, next_state, reward, done = env.unwrapped.P[state][action][0]
            # Since the environment is non-slippery, the transition probability is always 1
            Q[state, action] = reward + gamma * V[next_state]
    return Q

optimal_Q = compute_action_value_function(optimal_V, gamma)

# Display the optimal state value function
print("Optimal State Value Function (V):")
display_value_table(optimal_V)

# Display the optimal action value function
print("Optimal Action Value Function (Q):")
display_action_value_table(optimal_Q)

# Display the optimal policy
print("Optimal Policy:")
optimal_policy_mapping = np.array([policy_mapping[np.argmax(optimal_policy[s])] for s in range(states)])
print(optimal_policy_mapping.reshape((4, 4)))

# Run the environment with the optimal policy
print("\nDemonstrating Optimal Policy:")
run_policy(env, optimal_policy)

env.close()