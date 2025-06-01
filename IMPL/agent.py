import gymnasium as gym
import numpy as np
import pickle
import os
from config import N_BINS, MODEL_PATH, SUMMARY_PATH
from utils import create_bins, discretize, plot_training_summary, save_model, load_model

def train_agent(episodes=5000, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    pos_bins, vel_bins = create_bins(env, N_BINS)
    q_table = np.zeros((N_BINS, N_BINS, env.action_space.n))

    lr, gamma = 0.9, 0.9
    epsilon, decay = 1.0, 2 / episodes
    rng = np.random.default_rng()

    rewards, steps, velocities = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

    for ep in range(episodes):
        state = env.reset()[0]
        s_p, s_v = discretize(state, pos_bins, vel_bins)

        done, ep_reward, ep_vel, ep_steps = False, 0, 0, 0

        while not done and ep_steps < 1000:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[s_p, s_v])

            next_state, reward, done, _, _ = env.step(action)
            ns_p, ns_v = discretize(next_state, pos_bins, vel_bins)

            q_table[s_p, s_v, action] += lr * (
                reward + gamma * np.max(q_table[ns_p, ns_v]) - q_table[s_p, s_v, action]
            )

            state, s_p, s_v = next_state, ns_p, ns_v
            ep_reward += reward
            ep_vel += abs(state[1])
            ep_steps += 1

        epsilon = max(epsilon - decay, 0)
        rewards[ep], steps[ep], velocities[ep] = ep_reward, ep_steps, ep_vel / max(ep_steps, 1)

    env.close()
    save_model(q_table, MODEL_PATH)
    plot_training_summary(rewards, steps, velocities, q_table, pos_bins, vel_bins, SUMMARY_PATH)

def evaluate_agent(episodes=1, render=True):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    pos_bins, vel_bins = create_bins(env, N_BINS)
    q_table = load_model(MODEL_PATH)

    velocities, steps = np.zeros(episodes), np.zeros(episodes)

    for ep in range(episodes):
        state = env.reset()[0]
        s_p, s_v = discretize(state, pos_bins, vel_bins)

        done, ep_vel, ep_steps = False, 0, 0

        while not done and ep_steps < 1000:
            action = np.argmax(q_table[s_p, s_v])
            state, _, done, _, _ = env.step(action)
            s_p, s_v = discretize(state, pos_bins, vel_bins)
            ep_vel += abs(state[1])
            ep_steps += 1

        velocities[ep], steps[ep] = ep_vel / ep_steps, ep_steps

    env.close()
    print(f"\n--- Evaluation for {episodes} episode(s) ---")
    print("Average velocity:", np.mean(velocities))
    print("Max velocity reached:", np.max(velocities))
    print("Steps to reach goal:", np.min(steps))
