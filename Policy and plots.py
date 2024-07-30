
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Environment configurations
ENVIRONMENTS = {
    "FrozenLake-v1": {
        "render_mode": "human",
        "config": {
            "is_slippery": True,  # Default slippery condition
            "map_name": "4x4",    # Default map size
        },
        "policies": {
            "always_right_policy": lambda state: 2,
            "random_policy": lambda state: gym.make("FrozenLake-v1").action_space.sample(),
            "greedy_policy": lambda state: 2 if state in [0, 1, 2, 4, 5, 6, 8, 9, 10] else 1,
            "avoid_holes_policy": lambda state: 3 if state in [5, 7, 11, 12] else (2 if state % 4 == 0 else 1),
            "zigzag_policy": lambda state: 2 if state % 2 == 0 else 1,
        },
    },
    "MountainCar-v0": {
        "render_mode": "human",
        "config": {},
        "policies": {
            "always_right_policy": lambda state: 2,
            "random_policy": lambda state: gym.make("MountainCar-v0").action_space.sample(),
            "always_left_policy": lambda state: 0,
            "accelerate_right_policy": lambda state: 2 if state[1] < 0 else 0,
        },
    },
    "CartPole-v1": {
        "render_mode": "human",
        "config": {},
        "policies": {
            "random_policy": lambda state: gym.make("CartPole-v1").action_space.sample(),
            "left_policy": lambda state: 0,
            "right_policy": lambda state: 1,
            "angle_based_policy": lambda state: 0 if state[2] < 0 else 1,
        },
    },
}

# Function to create the environment with optional configurations
def create_env(env_name):
    config = ENVIRONMENTS[env_name].get("config", {})
    return gym.make(env_name, render_mode=ENVIRONMENTS[env_name]["render_mode"], **config)

# Function to print transition probabilities
def print_transition_probabilities(env):
    transition_probs = env.unwrapped.P
    for state in transition_probs:
        for action in transition_probs[state]:
            print(f"State {state}, Action {action}:")
            for prob, next_state, reward, done in transition_probs[state][action]:
                print(f"  P={prob}, Next State={next_state}, Reward={reward}, Done={done}")

# Function to run the simulation and record data
def run_simulation(env_name, policy_name, num_steps=100, delay=0.05):
    env = create_env(env_name)
    policy = ENVIRONMENTS[env_name]["policies"][policy_name]
    
    state, _ = env.reset()
    env.render()

    print_transition_probabilities(env)
    
    performance_data = []

    for step in range(num_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(delay)
        
        # Record performance data
        performance_data.append((step, state, action, reward, terminated, truncated, next_state))
        
        # Print detailed information at each step
        print(f"Step: {step + 1}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        print("---------")
        
        state = next_state
        if terminated or truncated:
            print(f"Episode finished after {step + 1} timesteps")
            print(f"Final Reward: {reward}")
            break
    env.close()
    
    return performance_data

# Function to plot performance data
def plot_performance(performance_data):
    steps = [data[0] for data in performance_data]
    rewards = [data[3] for data in performance_data]
    
    # Calculate cumulative rewards using numpy
    cumulative_rewards = np.cumsum(rewards)
    
    plt.plot(steps, rewards, marker='o', label='Reward at each step')
    plt.plot(steps, cumulative_rewards, marker='x', label='Cumulative reward')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title('Rewards at each step and cumulative rewards')
    plt.legend()
    plt.show()

# Function to save data using pickle
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Function to load data using pickle
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Run a specific instance and plot/save the data
environment_name = "FrozenLake-v1"  # Change to "MountainCar-v0" or "CartPole-v1" as needed
policy_name = "always_right_policy"  # Change to another policy name as needed

# Set custom configurations for FrozenLake
ENVIRONMENTS[environment_name]["config"]["is_slippery"] = True  # Change to False to remove slipperiness
ENVIRONMENTS[environment_name]["config"]["map_name"] = "4x4"     # Change to "8x8" for a larger map

print(f"\nTesting {policy_name} in {environment_name} with custom configurations")
performance_data = run_simulation(environment_name, policy_name)

# Plot the performance
plot_performance(performance_data)

# Save the performance data
save_data(performance_data, 'performance_data.pkl')

# Load the performance data and plot it again
loaded_data = load_data('performance_data.pkl')
plot_performance(loaded_data)