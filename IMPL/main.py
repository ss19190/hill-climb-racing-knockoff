from agent import train_agent, evaluate_agent

if __name__ == "__main__":
    # Train the agent
    train_agent(episodes=5000, render=False)

    # Evaluate the agent
    # evaluate_agent(episodes=10, render=True)
