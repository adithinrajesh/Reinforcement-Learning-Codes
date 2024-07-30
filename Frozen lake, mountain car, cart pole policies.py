import gymnasium as gym
import time

ENVIRONMENTS = {
    "FrozenLake-v1": {
        "render_mode": "human",
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
        "policies": {
            "always_right_policy": lambda state: 2,
            "random_policy": lambda state: gym.make("MountainCar-v0").action_space.sample(),
            "always_left_policy": lambda state: 0,
            "accelerate_right_policy": lambda state: 2 if state[1] < 0 else 0,
        },
    },
    "CartPole-v1": {
        "render_mode": "human",
        "policies": {
            "random_policy": lambda state: gym.make("CartPole-v1").action_space.sample(),
            "left_policy": lambda state: 0,
            "right_policy": lambda state: 1,
            "angle_based_policy": lambda state: 0 if state[2] < 0 else 1,
        },
    },
}

# Function to run the simulation
def run_simulation(env_name, policy_name, num_steps=100, delay=0.05):
    env = gym.make(env_name, render_mode=ENVIRONMENTS[env_name]["render_mode"])
    policy = ENVIRONMENTS[env_name]["policies"][policy_name]
    
    state, _ = env.reset()
    env.render()
    
    for step in range(num_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(delay)
        
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

# Run a specific instance
environment_name = "MountainCar-v0"  # Change to "MountainCar-v0" or "CartPole-v1" as needed
policy_name = "always_right_policy"  # Change to another policy name as needed

print(f"\nTesting {policy_name} in {environment_name}")
run_simulation(environment_name, policy_name)