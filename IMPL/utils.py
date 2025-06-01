import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def create_bins(env, bins):
    # Create bin thresholds for discretizing continuous observation space
    pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], bins)
    vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], bins)
    return pos_bins, vel_bins

def discretize(state, pos_bins, vel_bins):
    # Map continuous state to discrete indices based on bins
    p_idx = np.digitize(state[0], pos_bins)
    v_idx = np.digitize(state[1], vel_bins)
    return p_idx, v_idx

def save_model(q_table, path):
    # Save Q-table to disk using pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(q_table, f)

def load_model(path):
    # Load Q-table from disk
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_training_summary(rewards, steps, velocities, q, pos_bins, vel_bins, save_path):
    episodes = len(rewards)
    
    # Compute moving average reward over last 100 episodes
    mean_rewards = np.array([np.mean(rewards[max(0, t - 100): t + 1]) for t in range(episodes)])

    # Create vector field for Q-policy visualization
    X, Y = np.meshgrid(range(len(pos_bins)), range(len(vel_bins)))
    U, V = np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)

    for i in range(len(pos_bins)):
        for j in range(len(vel_bins)):
            action = np.argmax(q[i, j])
            U[j, i] = [-1, 0, 1][action]  # Left, Neutral, Right

    # Plot reward, steps, velocity, and policy field
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MountainCar-v0 Training Summary", fontsize=16)

    axs[0, 0].plot(mean_rewards, color='tab:blue')
    axs[0, 0].set_title("Mean Reward (100-ep avg)")

    axs[0, 1].plot(steps, color='tab:orange')
    axs[0, 1].set_title("Steps to Goal")

    axs[1, 0].plot(velocities, color='tab:green')
    axs[1, 0].set_title("Average Velocity")

    axs[1, 1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='tab:purple')
    axs[1, 1].set_title("Q-Policy (Preferred Action)")
    axs[1, 1].set_xlabel("Position Bin")
    axs[1, 1].set_ylabel("Velocity Bin")

    # Add grid and labels
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel("Episode" if ax != axs[1, 1] else "Position Bin")
        ax.set_ylabel("Value" if ax != axs[1, 1] else "Velocity Bin")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Saved training summary to {save_path}")
