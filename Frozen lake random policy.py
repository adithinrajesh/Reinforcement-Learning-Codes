import gymnasium as gym
import time
env = gym.make(
    "FrozenLake-v1",
    render_mode = "human",
    map_name = "4x4",
    is_slippery = True
)
state,_ = env.reset()
env.render()

def random_policy(state):
    return env.action_space.sample()

num_steps = 10
for step in range(num_steps):
    action = random_policy(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1)

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
        break

env.close()