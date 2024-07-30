import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the Frozen Lake environment (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Define the MDP components
states = env.observation_space.n
actions = env.action_space.n
gamma = 0.9  # Discount factor
theta = 1e-6  # Threshold for policy evaluation convergence

# Initialize the policy randomly
policy = np.ones([states, actions]) / actions

# Initialize the state value function
V = np.zeros(states)

# Print the transition probabilities for the environment
def print_transition_probabilities(env):
    transition_probs = env.unwrapped.P
    for state in transition_probs:
        for action in transition_probs[state]:
            print(f"State {state}, Action {action}:")
            for prob, next_state, reward, done in transition_probs[state][action]:
                print(f"  P={prob}, Next State={next_state}, Reward={reward}, Done={done}")

print_transition_probabilities(env)

# Policy Evaluation
def policy_evaluation(policy, V, gamma, theta):
    while True:
        delta = 0
        for state in range(states):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                transitions = env.unwrapped.P[state][action]
                prob, next_state, reward, done = transitions[0]
                v += action_prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        if delta < theta:
            break
    return V

# Policy Improvement
def policy_improvement(V, gamma):
    policy_stable = True
    for state in range(states):
        old_action = np.argmax(policy[state])
        action_values = np.zeros(actions)
        for action in range(actions):
            prob, next_state, reward, done = env.unwrapped.P[state][action][0]
            action_values[action] = reward + gamma * V[next_state]
        best_action = np.argmax(action_values)
        if old_action != best_action:
            policy_stable = False
        policy[state] = np.eye(actions)[best_action]
    return policy, policy_stable

# Policy Iteration
def policy_iteration(policy, V, gamma, theta):
    while True:
        V = policy_evaluation(policy, V, gamma, theta)
        policy, policy_stable = policy_improvement(V, gamma)
        if policy_stable:
            break
    return policy, V

# Perform policy iteration
optimal_policy, optimal_V = policy_iteration(policy, V, gamma, theta)

# Compute the action value function Q(s, a) using the optimal state value function
def compute_action_value_function(V, gamma):
    Q = np.zeros((states, actions))
    for state in range(states):
        for action in range(actions):
            prob, next_state, reward, done = env.unwrapped.P[state][action][0]
            Q[state, action] = reward + gamma * V[next_state]
    return Q

optimal_Q = compute_action_value_function(optimal_V, gamma)

# Display the optimal state value function
print("Optimal State Value Function (V):")
print(optimal_V.reshape((4, 4)))

# Display the optimal action value function
print("Optimal Action Value Function (Q):")
print(optimal_Q.reshape((4, 4, actions)))

# Display the optimal policy
print("Optimal Policy:")
policy_mapping = {0: '←', 1: '↓', 2: '→', 3: '↑'}
optimal_policy_mapping = np.array([policy_mapping[np.argmax(optimal_policy[s])] for s in range(states)])
print(optimal_policy_mapping.reshape((4, 4)))

# Visualize the state value function V(s)
plt.figure(figsize=(8, 6))
plt.bar(range(states), optimal_V)
plt.xlabel('States')
plt.ylabel('State Value V(s)')
plt.title('Optimal State Value Function')
plt.show()

# Visualize the action value function Q(s, a)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(optimal_Q, cmap='viridis')
fig.colorbar(cax)
plt.xlabel('Actions')
plt.ylabel('States')
plt.title('Optimal Action Value Function Q(s, a)')
plt.show()

# Function to run the environment with the optimal policy and print the rewards
def run_optimal_policy(env, policy, num_steps=10):
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

# Run the environment with the optimal policy
run_optimal_policy(env, optimal_policy)

env.close()