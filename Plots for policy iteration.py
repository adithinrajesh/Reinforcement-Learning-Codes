import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the Frozen Lake environment (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Define the MDP components
states = env.observation_space.n
actions = env.action_space.n
gamma = 0.9  # Discount factor

# Initialize the state and action value functions
V = np.zeros(states)
Q = np.zeros((states, actions))

# Print the transition probabilities for the environment
def print_transition_probabilities(env):
    transition_probs = env.unwrapped.P
    for state in transition_probs:
        for action in transition_probs[state]:
            print(f"State {state}, Action {action}:")
            for prob, next_state, reward, done in transition_probs[state][action]:
                print(f"  P={prob}, Next State={next_state}, Reward={reward}, Done={done}")

print_transition_probabilities(env)

# Compute the state value function V(s) using the Bellman equation (deterministic version)
def compute_state_value_function(V, gamma):
    new_V = np.zeros_like(V)
    for state in range(states):
        action_values = []
        for action in range(actions):
            transitions = env.unwrapped.P[state][action]
            # Since transitions are deterministic, we take the first one
            prob, next_state, reward, done = transitions[0]
            action_values.append(reward + gamma * V[next_state])
        new_V[state] = max(action_values) if action_values else 0
    return new_V

# Compute the action value function Q(s, a) using the Bellman equation (deterministic version)
def compute_action_value_function(V, gamma):
    Q = np.zeros((states, actions))
    for state in range(states):
        for action in range(actions):
            transitions = env.unwrapped.P[state][action]
            # Since transitions are deterministic, we take the first one
            prob, next_state, reward, done = transitions[0]
            Q[state, action] = reward + gamma * V[next_state]
    return Q

# Initialize the state and action value functions
V = np.zeros(states)

# Update the state and action value functions
V = compute_state_value_function(V, gamma)
Q = compute_action_value_function(V, gamma)

# Visualize the state value function V(s)
plt.figure(figsize=(8, 6))
plt.bar(range(states), V)
plt.xlabel('States')
plt.ylabel('State Value V(s)')
plt.title('State Value Function')
plt.show()

# Visualize the action value function Q(s, a)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(Q, cmap='viridis')
fig.colorbar(cax)
plt.xlabel('Actions')
plt.ylabel('States')
plt.title('Action Value Function Q(s, a)')
plt.show()

# Function to run the environment with a random policy and print the rewards
def run_random_policy(env, num_steps=10):
    state, _ = env.reset()
    total_reward = 0
    for step in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Step: {step + 1}, State: {state}, Action: {action}, Reward: {reward}")
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    print(f"Total Reward: {total_reward}")

# Run the environment with a random policy
run_random_policy(env)

env.close()