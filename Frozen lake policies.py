import gymnasium as gym
import time

env = gym.make(
    "FrozenLake-v1",
    render_mode="human",
    map_name="4x4",           
    is_slippery=False,  
)

def always_right_policy(state):
    return 2 #right

def random_policy(state):
    return env.action_space.sample()

def greedy_policy(state):
    if state in [0, 1, 2, 4, 5, 6, 8, 9, 10]: #move right is possible
        return 2 #right
    else:
        return 1  # Down

def avoid_holes_policy(state):
    hole_below = [1, 3, 8]
    if state in hole_below:
        return 2  # Left
    if state % 4 == 0 or state % 8 == 0:  # If in the leftmost column, move right
        return 2  # Right
    else:
        return 1  # Down

def zigzag_policy(state):
    if state == 1:
        return 2
    elif state % 2 == 0:
        return 2  # Right on even states
    else:
        return 1  # Down on odd states

def run_simulation(env, policy, num_steps=10, delay=1):
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

# Run the simulation with different policies
#print("Testing Always Right Policy")
#run_simulation(env, always_right_policy)

#print("Testing Random Policy")
#run_simulation(env, random_policy)

#print("Testing Greedy Policy")
#run_simulation(env, greedy_policy)

#print("Testing Avoid Holes Policy")
#run_simulation(env, avoid_holes_policy)

print("Testing Zigzag Policy")
run_simulation(env, zigzag_policy)

'''
1. Always right policy - If is_slippery is equal to True, it might 
                            not always move right. The env becomes 
                            stochastic and the slippery random nature
                            makes it move randomly (like downwards)
                            apart from moving right.

2. Random policy - Ignores the current state of the env and takes actions
                    randomly, as each action has equal probability.

3. Greedy policy - Exploits the current knowledge of the env. It selects
                    actions based on the expected reward from the current
                    state without considering long term rewards.
                    With slippery, it might take random actions, but without
                    slippery, itll follow exactly how we mention.

4. Avoid holes policy - I changed it because once the agent enters into 
                        the hole, it cannot take any action as it will be
                        terminated.

5. Zigzag policy - odd states move down, even states move right

'''