# Lux AI Season 3 - PPO Agent

A reinforcement learning agent for the [Lux AI Challenge Season 3](https://www.kaggle.com/competitions/lux-ai-season-3) competition, implemented using Proximal Policy Optimization (PPO) with JAX and Flax.

## Features

- **PPO Implementation**: Custom implementation of PPO using JAX for efficient training
- **Self-play Training**: Agent learns by playing against itself
- **Visualization**: Tools to visualize game replays in browser
- **Checkpointing**: Save and load model weights during training
- **Competition Ready**: Easy submission generation for the Kaggle competition

## Files

- `simple_transformer.py`: Core agent implementation with neural network architecture and PPO algorithm
- `train_simple_ppo.py`: Training script for self-play learning
- `visualize_replay.py`: Tool to visualize game replays
- `create_submission.py`: Script for creating competition submissions

## Getting Started

### Installation

```bash
# Clone the Lux AI S3 environment first
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3.git
cd Lux-Design-S3
pip install -e .

# Now clone this repository
git clone https://github.com/YOUR_USERNAME/lux-ppo-agent.git
cd lux-ppo-agent
pip install -r requirements.txt
```

### Training

```bash
python train_simple_ppo.py --iterations 1000
```

### Creating a Submission

```bash
python create_submission.py --checkpoint checkpoints/checkpoint_final --output-dir submission
```

### Running on Google Colab

This repository includes a notebook for training on Google Colab with GPU acceleration:

```python
!git clone https://github.com/YOUR_USERNAME/lux-ppo-agent.git
!cd lux-ppo-agent && pip install -r requirements.txt
!cd lux-ppo-agent && python train_simple_ppo.py --iterations 100
```

## Training on Colab

To train on Colab:

1. Open a new Colab notebook
2. Select a GPU runtime (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Clone this repository and install dependencies
4. Run the training script with desired iterations

## License

MIT