#!/usr/bin/env python
import os
# Configure JAX to use Metal
os.environ['METAL_DEVICE_WRITABLE'] = '1'

import jax
# Make sure we're using Metal backend
print(f"JAX is using: {jax.devices()}")

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import time
import argparse

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from simple_transformer import SimplePPOAgent, Colors

def parse_args():
    parser = argparse.ArgumentParser(description="Train Simple PPO agent for Lux AI S3 with self-play")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128, help="Number of steps per iteration")
    parser.add_argument("--eval-frequency", type=int, default=10, help="Evaluate every N iterations")
    parser.add_argument("--save-frequency", type=int, default=10, help="Save checkpoint every N iterations")
    parser.add_argument("--small-test", action="store_true", help="Run a small test with minimal steps")
    
    # Agent parameters
    parser.add_argument("--hidden-size", type=int, default=512, help="Size of hidden layers")
    
    # PPO parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda-gae", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of update epochs")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches")
    
    # Environment parameters
    parser.add_argument("--map-type", type=int, default=0, help="Map type")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per match")
    
    # Output parameters
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--submission-dir", type=str, default="submission", help="Submission directory")
    parser.add_argument("--load-checkpoint", type=str, default="", help="Load checkpoint from file (use 'latest' for most recent)")
    
    # Other
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--create-submission", action="store_true", help="Create submission after training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n{Colors.HEADER}Lux AI S3 - Simple PPO Self-Play Training{Colors.ENDC}")
    print(f"{Colors.BLUE}Setting up training environment...{Colors.ENDC}")
    
    # Set random seed
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # Create environment
    env = LuxAIS3Env()
    env_params = EnvParams(
        map_type=args.map_type,
        max_steps_in_match=args.max_steps
    )
    
    # Create agent
    agent = SimplePPOAgent(
        env=env,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        num_envs=1,  # We're actually only using 1 environment for now
        num_steps=args.num_steps,
        anneal_lr=True,
        debug=args.debug,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        env_params=env_params  # Pass the environment parameters
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        if args.load_checkpoint == "latest":
            # Find the latest checkpoint
            checkpoint_dir = args.checkpoint_dir
            # List all directories matching checkpoint_*
            import glob
            checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))
            if checkpoint_dirs:
                # Get the most recent checkpoint
                latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
                # Extract the suffix
                suffix = latest_checkpoint.split("checkpoint_")[-1]
                agent.load_checkpoint(suffix)
            else:
                print(f"{Colors.RED}No checkpoints found in {checkpoint_dir}{Colors.ENDC}")
        else:
            agent.load_checkpoint(args.load_checkpoint)
    
    # Train agent
    metrics = agent.train_selfplay(
        num_iterations=args.iterations,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        small_test=args.small_test
    )
    
    # Create submission if requested
    if args.create_submission:
        print(f"\n{Colors.BOLD}Creating submission...{Colors.ENDC}")
        submission_path = agent.create_submission(args.submission_dir)
        print(f"{Colors.GREEN}Submission created at {submission_path}{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}Training complete!{Colors.ENDC}")

if __name__ == "__main__":
    main()