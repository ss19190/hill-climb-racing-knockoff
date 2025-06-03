import gymnasium as gym
import numpy as np
import pickle
import os
import time
from config import N_BINS, MODEL_PATH, SUMMARY_PATH
from utils import create_bins, discretize, plot_training_summary, save_model, load_model

def train_agent(episodes=5000, render=False):
    # Create the environment
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Discretize the continuous observation space into bins
    pos_bins, vel_bins = create_bins(env, N_BINS)

    # Initialize the Q-table with zeros: shape = (position_bins, velocity_bins, actions)
    q_table = np.zeros((N_BINS, N_BINS, env.action_space.n))

    # Learning parameters
    lr = 0.9       # Learning rate
    gamma = 0.9    # Discount factor
    epsilon = 1.0  # Exploration rate
    decay = 2 / episodes  # Linear decay for epsilon

    # Random number generator
    rng = np.random.default_rng()

    # Tracking statistics
    rewards = np.zeros(episodes)
    steps = np.zeros(episodes)
    velocities = np.zeros(episodes)

    for ep in range(episodes):
        print ('Episode: ', ep)
        state = env.reset()[0]  # Get initial state
        s_p, s_v = discretize(state, pos_bins, vel_bins)

        done, ep_reward, ep_vel, ep_steps = False, 0, 0, 0

        # Run episode
        while not done and ep_steps < 1000:
            # ε-greedy policy
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[s_p, s_v])  # Exploit

            # Take action in the environment
            next_state, reward, done, _, _ = env.step(action)
            ns_p, ns_v = discretize(next_state, pos_bins, vel_bins)

            # Q-learning update rule
            q_table[s_p, s_v, action] += lr * (
                reward + gamma * np.max(q_table[ns_p, ns_v]) - q_table[s_p, s_v, action]
            )

            # Update current state and statistics
            state, s_p, s_v = next_state, ns_p, ns_v
            ep_reward += reward
            ep_vel += abs(state[1])
            ep_steps += 1

        # Decrease exploration rate
        epsilon = max(epsilon - decay, 0)

        # Record episode statistics
        rewards[ep], steps[ep], velocities[ep] = ep_reward, ep_steps, ep_vel / max(ep_steps, 1)

    # Close the environment
    env.close()

    # Save Q-table and plot results
    save_model(q_table, MODEL_PATH)
    plot_training_summary(rewards, steps, velocities, q_table, pos_bins, vel_bins, SUMMARY_PATH)

def evaluate_agent(episodes=1, render=True):
    # Load environment and trained Q-table
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    pos_bins, vel_bins = create_bins(env, N_BINS)
    q_table = load_model(MODEL_PATH)

    # Tracking statistics
    velocities, steps, times = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

    for ep in range(episodes):
        state = env.reset()[0]
        s_p, s_v = discretize(state, pos_bins, vel_bins)

        done, ep_vel, ep_steps = False, 0, 0
        start_time = time.time()

        while not done and ep_steps < 1000:
            # Always exploit during evaluation
            action = np.argmax(q_table[s_p, s_v])
            state, _, done, _, _ = env.step(action)
            s_p, s_v = discretize(state, pos_bins, vel_bins)
            ep_vel += abs(state[1])
            ep_steps += 1

        # Record stats
        times[ep] = time.time() - start_time
        velocities[ep], steps[ep] = ep_vel / ep_steps, ep_steps

    # Close environment and display results
    env.close()

    # Print evaluation summary
    print(f"\n--- Evaluation for {episodes} episode(s) ---")
    print("Max velocity reached:", np.max(velocities))
    if episodes == 1:
        print("Time to goal (s):", times[ep])
        print("Steps to goal:", steps[ep])
    else:
        print("Average max velocity:", np.mean(velocities))
        print("Min steps to goal:", np.min(steps))
        print("Avg steps to goal:", np.mean(steps))
        print("Avg time to goal (s):", np.mean(times))
        print("Min time to goal (s):", np.min(times))