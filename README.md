# MultiAgent_Adversarial_Game

A Python implementation of a Minimax Q-Learning algorithm in a zero-sum game environment.

## Features
- Grid world environment with customizable size.
- Minimax Q-learning for a protagonist (minimizer) and adversary (maximizer).
- Visualization of trajectories and policies.
- Evaluation of learned policies.

## Requirements
Install the required Python packages:
```bash
pip install numpy matplotlib pulp

## Usage
from zero_sum_gridworld import ZeroSumGridWorld

env = ZeroSumGridWorld(grid_size=4)
env.train(num_episodes=10000)
env.visualize_trajectory()

