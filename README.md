# Deep Reinforcement Learning for Atari Breakout

A PyTorch implementation of a **Deep Q-Network (DQN)** agent trained to play **Atari Breakout** using the Gymnasium Atari environment. This project also includes a baseline DQN implementation for the CartPole environment and an experimental BERT embedding script used for NLP exploration.

---

## Overview

This project explores the implementation of Deep Reinforcement Learning using the DQN algorithm introduced by DeepMind.

The primary goal is to train an autonomous agent that learns to play Atari Breakout directly from stacked grayscale game frames through trial-and-error interactions with the environment.

The repository contains:

- A DQN implementation for Atari Breakout
- A trained Breakout model
- A DQN implementation for CartPole
- Utilities for testing trained models
- An experimental BERT embedding example

---

## Features

- Deep Q-Network (DQN)
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Soft Target Network Updates
- Frame Stacking (4 consecutive frames)
- GPU acceleration using PyTorch
- Play trained Breakout agent

---

## Repository Structure

```text
.
├── breakout.py              # Main Breakout implementation
├── test_breakout.py         # Alternative training/testing script
├── breakoutdqn.py           # Google Colab version
├── cartpole.py              # DQN implementation for CartPole
├── berthw.py                # BERT embedding experiment
├── Breakout_net.pth         # Trained Breakout model
└── README.md
```

---

## Deep Q-Network Architecture

### Input

The Breakout agent receives:

- 4 stacked grayscale frames
- FrameStackObservation wrapper
- Shape: (4, Height, Width)

### Network

```
Input
   ↓
Conv2D
   ↓
ReLU
   ↓
MaxPooling
   ↓
Conv2D
   ↓
ReLU
   ↓
MaxPooling
   ↓
Flatten
   ↓
Fully Connected
   ↓
Fully Connected
   ↓
Output Q-values
```

---

## Reinforcement Learning Pipeline

```
Game Environment
       │
       ▼
Current State
       │
       ▼
Policy Network
       │
       ▼
Choose Action
       │
       ▼
Environment
       │
       ▼
Reward + Next State
       │
       ▼
Replay Memory
       │
       ▼
Random Mini-batch
       │
       ▼
Optimize DQN
       │
       ▼
Soft Update Target Network
```

---

## Technologies Used

- Python
- PyTorch
- Gymnasium
- ALE (Arcade Learning Environment)
- NumPy
- Matplotlib

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/breakout-dqn.git

cd breakout-dqn
```

Install dependencies

```bash
pip install torch
pip install gymnasium
pip install gymnasium[atari]
pip install ale-py
pip install matplotlib
```

Or install everything at once

```bash
pip install torch gymnasium gymnasium[atari] ale-py matplotlib
```

---

## Training the Agent

To train the Breakout agent:

```bash
python breakout.py
```

The network periodically saves its weights as

```
Breakout_net.pth
```

---

## Testing a Trained Model

Load the trained network and watch the agent play:

```bash
python breakout.py
```

The environment opens in human rendering mode and the trained model selects actions greedily.

---

## Training Hyperparameters

| Parameter | Value |
|------------|-------|
| Batch Size | 256 |
| Replay Buffer | 50,000 |
| Learning Rate | 5e-4 |
| Gamma | 0.99 |
| Epsilon Start | 0.90 |
| Epsilon End | 0.01 |
| Target Update (τ) | 0.005 |
| Episodes | 3001 |

---

## CartPole Example

The repository also includes a simpler DQN implementation for the classic CartPole environment.

Run:

```bash
python cartpole.py
```

This serves as an introduction before tackling the more complex Atari environment.

---

## Experimental BERT Script

`berthw.py` is an independent experiment demonstrating how to:

- Load a pretrained BERT model
- Tokenize sentences
- Generate contextual embeddings
- Compare sentence embeddings using vector norms

It is unrelated to the reinforcement learning implementation but included as an NLP learning exercise.

---

## Future Improvements

Potential improvements include:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Noisy Networks
- Multi-step Returns
- Rainbow DQN
- TensorBoard logging
- Model checkpointing
- Performance evaluation metrics

---

## Learning Objectives

This project demonstrates understanding of:

- Reinforcement Learning
- Deep Q-Learning
- Convolutional Neural Networks
- Experience Replay
- Target Networks
- Atari Reinforcement Learning
- PyTorch model development

---

## References

- Mnih et al. (2015), *Human-level Control Through Deep Reinforcement Learning*
- PyTorch Reinforcement Learning Tutorial
- Gymnasium Documentation
- Arcade Learning Environment (ALE)

---

## License

This project is intended for educational and research purposes.
