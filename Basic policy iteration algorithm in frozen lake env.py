import numpy as np
import gym
import time #Importing time to control the execution time of the policy visualisation

'''
The purpose of policy iteration is to compute the value function V
for a given policy pi. The value function represents the expected 
return starting from state s and following policy pi.
'''
# Initialize the Frozen Lake environment and is_slippery is false as its deterministic
env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=False)

num_states = env.observation_space.n #This returns the number of states in the env, in 4x4 frozen lake there are 16 states
num_actions = env.action_space.n #This returns the number of action spaces, which is 4 here
gamma = 0.9  # Discount factor - Future rewards are valued at 90% of their face value
theta = 1e-6  # Convergence threshold

'''
1. Policy evaluation:
    Theta is used to check for convergence of V for a given policy.
    We iteratively update the value funtion for each state until the change
    in value function across all states in less than theta.

2. Policy improvement:
    We use the updated value function to improve the policy.
    Find the policy that maximizees the return.
'''

def compute_state_value(state, policy, V): #Compute the value of a state given the policy.
    action = policy[state]
    total = 0
    for prob, next_state, reward, done in env.P[state][action]:
        total += prob * (reward + gamma * V[next_state])
    return total

'''
In MDPs, the value of a state is the expected return when starting from
that state and following a given policy.
To compute the V value, 
Each outcome has a certain probability, return from each outcome is the reward
plus the discounted value of the next state.
By adding the returns of all possible outcomes, we get the value.
'''

def compute_q_value(state, action, V): #Compute the Q-value of a state-action pair.
    total = 0
    for prob, next_state, reward, done in env.P[state][action]:
        total += prob * (reward + gamma * V[next_state])
    return total

def policy_evaluation(policy):
    """Evaluate the given policy."""
    V = np.zeros(num_states)
    while True:
        delta = 0
        for state in range(num_states):
            v = V[state]
            V[state] = compute_state_value(state, policy, V)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(V):
    """Improve the policy based on the current value function."""
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        q_values = np.zeros(num_actions)
        for action in range(num_actions):
            q_values[action] = compute_q_value(state, action, V)
        policy[state] = np.argmax(q_values)
    return policy

def policy_iteration():
    """Perform policy iteration."""
    policy = np.zeros(num_states, dtype=int)  # Initial policy
    while True:
        V = policy_evaluation(policy)
        improved_policy = policy_improvement(V)

        if np.array_equal(improved_policy, policy):
            break
        policy = improved_policy
    return policy, V

optimal_policy, optimal_value_function = policy_iteration()

def run_policy(env, policy, num_steps=100):
    state, _ = env.reset()  # Unpack state and info
    env.render()
    
    for step in range(num_steps):
        action = policy[state]
        result = env.step(action)
        
        if len(result) == 4:
            next_state, reward, done, info = result
        elif len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            raise ValueError("Unexpected step result length")

        print(f"Step: {step + 1}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print("---------")
        
        env.render()
        time.sleep(1)
        
        state = next_state
        if done:
            print(f"Episode finished after {step + 1} timesteps")
            break

    env.close()

# Run the optimal policy
run_policy(env, optimal_policy)
