# 🏔️ MountainCar-v0 Q-Learning Agent

This project implements a Q-learning agent for the `MountainCar-v0` environment from OpenAI Gymnasium. The agent learns to reach the goal by discretizing the continuous state space and visualizing its policy using a vector field.

## 📁 Project Structure

```
project-root/
│
├── IMPL/                     # Implementation code
│   ├── main.py               # Running programm 
│   ├── agent.py              # Main training and evaluation logic
│   ├── utils.py              # Utility functions (binning, saving, etc.)
│   └── config.py             # Config for paths and parameters
│
├── DATA/                     # Output data (models, plots)
│   ├── mountain_car.pkl      # Trained Q-table (after training)
│   └── mountain_car_summary.png # Training summary plots
│
└── DOC/                      # Documentation files
```

## ⚙️ Configuration (`IMPL/config.py`)

Before running, make sure the paths and parameters in `IMPL/config.py` are correctly set:

```python
# Number of bins for discretizing position and velocity
N_BINS = 20

# Path to save the trained Q-table
MODEL_PATH = "../DATA/mountain_car.pkl"

# Path to save the training summary plot
SUMMARY_PATH = "../DATA/mountain_car_summary.png"
```

## 🚀 How to Run

Navigate to the `IMPL` folder:
```bash
cd IMPL
```

### 🔧 1. Train the Agent
Run the `main.py` file to train the agent:
```bash
python main.py
```

By default, it trains for 5000 episodes. The model and training summary will be saved to `../DATA/`.

> You can modify the number of episodes by changing the `train_agent()` call main.py file.

### 👁️ 2. Evaluate the Agent
After training, you can evaluate the agent's performance:
```python
# In main.py, uncomment the evaluation line:
evaluate_agent(episodes=10, render=True)
```
This will run 10 episodes with rendering and print statistics such as time to goal, steps taken, and average velocity.
You can change the amount of episodes.

## 📊 Training Summary

After training, a plot is generated that includes:
- **Mean Reward** — 100-episode moving average
- **Steps to Goal** — How many steps it takes per episode
- **Average Velocity** — Average absolute velocity in each episode
- **Q-Policy** — A vector field showing the best action in each (position, velocity) bin

Saved to: `DATA/mountain_car_summary.png`

## 🧠 Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `matplotlib`

Install dependencies using:
```bash
pip install gymnasium numpy matplotlib
```

## 📌 Notes

- If you move files or folders, remember to update the paths in `config.py`.
