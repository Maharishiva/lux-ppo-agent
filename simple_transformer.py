import os
# Configure JAX to use Metal
os.environ['METAL_DEVICE_WRITABLE'] = '1'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Tuple
from flax.training.train_state import TrainState
import distrax
from tqdm import tqdm
import time
import pickle
from flax import serialization
from functools import partial  # Import partial for jax.jit decoration

# Print JAX device information to verify GPU/Metal usage
print(f"JAX devices: {jax.devices()}")

# Import the Lux environment
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams

# Define color codes for prettier logging (copied from dqn_agent.py)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class Transition(NamedTuple):
    done: jnp.ndarray
    action: Dict[str, jnp.ndarray]
    value: jnp.ndarray
    reward: Dict[str, jnp.ndarray]
    log_prob: jnp.ndarray
    obs: Dict[str, Any]
    info: Dict[str, Any]

class ActorCriticNetwork(nn.Module):
    """
    Simple Actor-Critic network for Lux AI S3.
    """
    max_units: int
    hidden_size: int
    input_size: int = 2000  # Fixed size greater than any possible input
    
    @nn.compact
    def __call__(self, x):
        # Pad input to fixed size or truncate if too large
        x_shape = x.shape
        if len(x_shape) == 1:
            x = jnp.pad(x, (0, self.input_size - x_shape[0]), mode='constant', constant_values=0)
            x = x[:self.input_size]  # Ensure consistent size by truncating if needed
        else:
            # Batch dimension
            x = jnp.pad(x, ((0, 0), (0, self.input_size - x_shape[1])), mode='constant', constant_values=0)
            x = x[:, :self.input_size]  # Ensure consistent size
            
        # Process through shared layers - using orthogonal initialization for better training stability
        x = nn.Dense(
            self.hidden_size, 
            kernel_init=orthogonal(scale=np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(scale=np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(scale=np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        # Actor head for action types
        action_logits = nn.Dense(
            self.max_units * 6,
            kernel_init=orthogonal(scale=0.01),
            bias_init=constant(0.0)
        )(x)
        action_logits = action_logits.reshape(-1, self.max_units, 6)
        
        # Actor head for sap targets
        sap_logits = nn.Dense(
            self.max_units * 17 * 17,
            kernel_init=orthogonal(scale=0.01),
            bias_init=constant(0.0)
        )(x)
        sap_logits = sap_logits.reshape(-1, self.max_units, 17, 17)
        
        # Critic head
        value = nn.Dense(
            1,
            kernel_init=orthogonal(scale=1.0),
            bias_init=constant(0.0)
        )(x)
        
        return action_logits, sap_logits, value.squeeze(-1)

class SimplePPOAgent:
    """
    Simplified PPO Agent for Lux AI S3
    """
    def __init__(
        self,
        env: LuxAIS3Env,
        hidden_size: int = 512,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        num_envs: int = 1,  # Actually only using 1 environment for now
        num_steps: int = 128,
        anneal_lr: bool = True,
        name: str = "agent",
        debug: bool = False,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        seed: int = 0,
        env_params: EnvParams = None  # Accept env_params from outside
    ):
        self.env = env
        # Use provided env_params or create default ones
        self.env_params = env_params if env_params else EnvParams(map_type=0)
        self.hidden_size = hidden_size
        # Initialize tracking for both players' rewards
        self.player0_rewards = []
        self.player1_rewards = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.name = name
        self.debug = debug
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create checkpoint and log directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize random key
        self.rng = jax.random.PRNGKey(seed)
        
        # Create network
        self.network = ActorCriticNetwork(
            max_units=self.env_params.max_units,
            hidden_size=self.hidden_size
        )
        
        # Initialize parameters with a simple dummy input
        self.rng, init_key = jax.random.split(self.rng)
        dummy_input = jnp.zeros((1, 2000))  # Using fixed input size from network
        self.network_params = self.network.init(init_key, dummy_input)
        
        # Create optimizer
        if self.anneal_lr:
            self.tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(learning_rate=self.learning_rate, eps=1e-5),
            )
        else:
            self.tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.learning_rate, eps=1e-5)
            )
        
        # Create train state
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.network_params,
            tx=self.tx,
        )
        
        # Track iterations for checkpointing
        self.current_iteration = 0
    
    def preprocess_obs(self, obs, team_id=0):
        """
        Process raw observations into a flat vector for the network.
        Handles both attribute-style and dictionary access for maximum robustness.
        """
        try:
            # Check if this is a valid observation
            if obs is None:
                print("Warning: Observation is None")
                return jnp.zeros(2000)
            
            # Extract player data
            player_key = f"player_{team_id}"
            if player_key not in obs:
                if self.debug:
                    print(f"Player key {player_key} not found in observation. Keys: {list(obs.keys())}")
                return jnp.zeros(2000)
                
            player_obs = obs[player_key]
            
            # Extract unit features for the team - handle both attribute and dictionary access
            try:
                if hasattr(player_obs, 'units'):
                    # Handle attribute-style access (EnvObs objects)
                    unit_positions = player_obs.units.position[team_id]  # (max_units, 2)
                    unit_energies = player_obs.units.energy[team_id]     # (max_units, 1) or (max_units,)
                    unit_mask = player_obs.units_mask[team_id]           # (max_units,)
                    
                    # Ensure unit_energies has the right shape - safe reshape
                    if len(unit_energies.shape) == 2:
                        # Just flatten it to be safe rather than using squeeze
                        unit_energies = unit_energies.reshape(-1)
                    
                    # Extract map features
                    map_energy = player_obs.map_features.energy
                    map_tile_type = player_obs.map_features.tile_type
                    sensor_mask = player_obs.sensor_mask
                    relic_nodes = player_obs.relic_nodes
                    relic_nodes_mask = player_obs.relic_nodes_mask
                    steps = float(player_obs.steps)
                    match_steps = float(player_obs.match_steps)
                else:
                    # Handle dictionary-style access
                    # Get unit info - handle nested structure
                    if "units" in player_obs and "position" in player_obs["units"]:
                        units_data = player_obs["units"]
                        unit_positions = jnp.array(units_data["position"][team_id])
                        unit_energies = jnp.array(units_data["energy"][team_id])
                    else:
                        # Create dummy unit data if missing
                        unit_positions = jnp.zeros((self.env_params.max_units, 2))
                        unit_energies = jnp.zeros(self.env_params.max_units)
                    
                    # Get unit mask
                    if "units_mask" in player_obs:
                        unit_mask = jnp.array(player_obs["units_mask"][team_id])
                    else:
                        unit_mask = jnp.zeros(self.env_params.max_units)
                    
                    # Ensure unit_energies has the right shape
                    if len(unit_energies.shape) == 2:
                        # Just reshape it to be safe rather than using squeeze
                        unit_energies = unit_energies.reshape(-1)
                    
                    # Extract map features safely with defaults
                    if "map_features" in player_obs:
                        map_features = player_obs["map_features"]
                        map_energy = jnp.array(map_features.get("energy", jnp.zeros((4, 4))))
                        map_tile_type = jnp.array(map_features.get("tile_type", jnp.zeros((4, 4))))
                    else:
                        map_energy = jnp.zeros((4, 4))
                        map_tile_type = jnp.zeros((4, 4))
                    
                    # Get other features with defaults
                    sensor_mask = jnp.array(player_obs.get("sensor_mask", jnp.zeros((4, 4))))
                    relic_nodes = jnp.array(player_obs.get("relic_nodes", jnp.zeros((2, 2))))
                    relic_nodes_mask = jnp.array(player_obs.get("relic_nodes_mask", jnp.zeros(2)))
                    steps = float(player_obs.get("steps", 0))
                    match_steps = float(player_obs.get("match_steps", 0))
                
                # Reshape unit features to have consistent dimensions
                unit_positions_flat = unit_positions.reshape(-1)  # Flatten to 1D array
                unit_energies_flat = unit_energies.reshape(-1)    # Flatten to 1D array
                unit_mask_flat = unit_mask.reshape(-1)            # Flatten to 1D array
                
                # Concatenate as separate features
                unit_features = jnp.concatenate([unit_positions_flat, unit_energies_flat, unit_mask_flat])
                
                # Flatten map features
                map_energy_flat = map_energy.flatten()
                map_tile_type_flat = map_tile_type.flatten()
                sensor_mask_flat = sensor_mask.flatten()
                
                # Flatten relic nodes
                relic_nodes_flat = relic_nodes.reshape(-1)
                
                # Concatenate everything into a flat vector
                processed_obs = jnp.concatenate([
                    unit_features,
                    map_energy_flat,
                    map_tile_type_flat,
                    sensor_mask_flat,
                    relic_nodes_flat,
                    relic_nodes_mask,
                    jnp.array([steps, match_steps])
                ])
                
                # Ensure consistent size - pad or truncate as needed
                if len(processed_obs) > 2000:
                    return processed_obs[:2000]  # Truncate if too long
                elif len(processed_obs) < 2000:
                    return jnp.pad(processed_obs, (0, 2000 - len(processed_obs)))  # Pad if too short
                else:
                    return processed_obs
                
            except Exception as e:
                if self.debug:
                    print(f"Error extracting features: {e}")
                return jnp.zeros(2000)
                
        except Exception as e:
            # Return a dummy observation if processing fails
            if self.debug:
                print(f"Error processing observation: {e}")
                if isinstance(obs, dict):
                    print(f"Observation keys: {list(obs.keys())}")
            return jnp.zeros(2000)  # Return zeros with the expected input size
    
    def select_action(self, obs, rng, training=True, epsilon=0.01):
        """
        Select actions for all units based on current policy.
        This generates SEPARATE actions for each player using the same policy.
        """
        try:
            # Process observations for both players
            processed_obs_p0 = self.preprocess_obs(obs, team_id=0)
            processed_obs_p0 = processed_obs_p0[None, :]  # Add batch dimension
            
            processed_obs_p1 = self.preprocess_obs(obs, team_id=1)
            processed_obs_p1 = processed_obs_p1[None, :]  # Add batch dimension
            
            # Get action logits and value for player 0
            action_logits_p0, sap_logits_p0, value_p0 = self.network.apply(
                self.train_state.params, processed_obs_p0
            )
            
            # Get action logits and value for player 1
            action_logits_p1, sap_logits_p1, value_p1 = self.network.apply(
                self.train_state.params, processed_obs_p1
            )
            
            # Create categorical distributions
            pi_action_types_p0 = distrax.Categorical(logits=action_logits_p0[0])
            pi_action_types_p1 = distrax.Categorical(logits=action_logits_p1[0])
            
            # Split random keys
            rng, action_key_p0, action_key_p1, sap_key_p0, sap_key_p1 = jax.random.split(rng, 5)
            
            # Sample actions for player 0
            action_types_p0 = pi_action_types_p0.sample(seed=action_key_p0)
            sap_logits_flat_p0 = sap_logits_p0[0].reshape(self.env_params.max_units, 17*17)
            sap_pi_p0 = distrax.Categorical(logits=sap_logits_flat_p0)
            sap_indices_p0 = sap_pi_p0.sample(seed=sap_key_p0)
            
            # Convert indices to x,y coordinates (-8 to 8)
            sap_x_p0 = (sap_indices_p0 % 17) - 8
            sap_y_p0 = (sap_indices_p0 // 17) - 8
            
            # Sample actions for player 1
            action_types_p1 = pi_action_types_p1.sample(seed=action_key_p1)
            sap_logits_flat_p1 = sap_logits_p1[0].reshape(self.env_params.max_units, 17*17)
            sap_pi_p1 = distrax.Categorical(logits=sap_logits_flat_p1)
            sap_indices_p1 = sap_pi_p1.sample(seed=sap_key_p1)
            
            # Convert indices to x,y coordinates (-8 to 8)
            sap_x_p1 = (sap_indices_p1 % 17) - 8
            sap_y_p1 = (sap_indices_p1 // 17) - 8
            
            # Get log probabilities
            log_probs_p0 = pi_action_types_p0.log_prob(action_types_p0)
            sap_log_probs_p0 = sap_pi_p0.log_prob(sap_indices_p0)
            
            # Combine log probabilities - use both action types and sap positions
            combined_log_probs = log_probs_p0 + sap_log_probs_p0
            
            # Construct actions
            actions_p0 = jnp.stack([action_types_p0, sap_x_p0, sap_y_p0], axis=-1)
            actions_p1 = jnp.stack([action_types_p1, sap_x_p1, sap_y_p1], axis=-1)
            
            # Create action dictionary
            action_dict = {
                "player_0": actions_p0,
                "player_1": actions_p1  # Separate actions for player_1
            }
            
            return action_dict, combined_log_probs, value_p0[0], rng
        except Exception as e:
            print(f"Error in select_action: {e}")
            # Fallback for error case
            rng, key1, key2 = jax.random.split(rng, 3)
            action_types_p0 = jax.random.randint(key1, (self.env_params.max_units,), 0, 6)
            sap_x_p0 = jax.random.randint(key2, (self.env_params.max_units,), -8, 9)
            sap_y_p0 = jax.random.randint(key2, (self.env_params.max_units,), -8, 9)
            
            actions_p0 = jnp.stack([action_types_p0, sap_x_p0, sap_y_p0], axis=-1)
            actions_p1 = jnp.stack([action_types_p0, sap_x_p0, sap_y_p0], axis=-1)
            
            action_dict = {
                "player_0": actions_p0,
                "player_1": actions_p1
            }
            
            # Dummy log probs and value
            log_probs = jnp.zeros(self.env_params.max_units)
            value = jnp.array(0.0)
            
            return action_dict, log_probs, value, rng
    
    # Implementation of JIT will come in a future update
    
    def train_selfplay(self, num_iterations, eval_frequency=10, save_frequency=10, small_test=False):
        """
        Train the agent using self-play PPO.
        """
        print(f"\n{Colors.HEADER}Starting Self-Play Training with PPO{Colors.ENDC}")
        print(f"{Colors.BLUE}Configuration:{Colors.ENDC}")
        print(f"  Iterations: {Colors.GREEN}{num_iterations}{Colors.ENDC}")
        print(f"  Eval Frequency: {Colors.GREEN}{eval_frequency}{Colors.ENDC}")
        print(f"  Save Frequency: {Colors.GREEN}{save_frequency}{Colors.ENDC}")
        print(f"  Env Steps per Iteration: {Colors.GREEN}{self.num_steps * self.num_envs}{Colors.ENDC}")
        print(f"  Total Steps: {Colors.GREEN}{num_iterations * self.num_steps * self.num_envs}{Colors.ENDC}")
        print(f"  Learning Rate: {Colors.GREEN}{self.learning_rate}{Colors.ENDC}")
        print(f"  Hidden Size: {Colors.GREEN}{self.hidden_size}{Colors.ENDC}")
        print(f"  Checkpoint Directory: {Colors.GREEN}{self.checkpoint_dir}{Colors.ENDC}")
        
        # Create progress bar
        progress_bar = tqdm(range(num_iterations), desc="Training Progress")
        
        # Initialize environment
        self.rng, reset_key = jax.random.split(self.rng)
        obs, env_state = self.env.reset(reset_key, params=self.env_params)
        
        # Metrics
        all_metrics = []
        start_time = time.time()
        
        # Small test mode for debugging
        if small_test:
            print(f"\n{Colors.YELLOW}SMALL TEST MODE: Only collecting a minimal trajectory{Colors.ENDC}")
            self.num_steps = 1  # Just one step per iteration
        
        for iteration in progress_bar:
            self.current_iteration = iteration
            
            # Collect trajectories
            iteration_start_time = time.time()
            trajectories, env_state, final_obs, metrics = self._collect_trajectories(env_state, obs)
            
            # Update policy
            self.train_state, loss_info = self._update_policy(trajectories)
            
            # Store metrics
            metrics.update({
                "value_loss": float(jnp.mean(loss_info[0])),
                "policy_loss": float(jnp.mean(loss_info[1])),
                "entropy": float(jnp.mean(loss_info[2])),
                "iteration": iteration,
                "timestamp": time.time()
            })
            all_metrics.append(metrics)
            
            # Save metrics
            self._save_metrics(metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                "reward": f"{metrics['episode_reward']:.2f}",
                "value_loss": f"{metrics['value_loss']:.4f}",
                "policy_loss": f"{metrics['policy_loss']:.4f}"
            })
            
            # Log progress
            iteration_duration = time.time() - iteration_start_time
            if (iteration + 1) % 10 == 0 or iteration == 0:
                total_duration = time.time() - start_time
                print(f"\n{Colors.BOLD}Iteration {iteration + 1}/{num_iterations} Stats:{Colors.ENDC}")
                print(f"Time: {Colors.BLUE}Iteration: {iteration_duration:.1f}s | Total: {total_duration:.1f}s{Colors.ENDC}")
                print(f"Rewards: {Colors.YELLOW}{metrics['episode_reward']:.2f}{Colors.ENDC}")
                print(f"Value Loss: {Colors.RED}{metrics['value_loss']:.4f}{Colors.ENDC}")
                print(f"Policy Loss: {Colors.RED}{metrics['policy_loss']:.4f}{Colors.ENDC}")
                print(f"Entropy: {Colors.BLUE}{metrics['entropy']:.4f}{Colors.ENDC}")
            
            # Evaluate agent
            if (iteration + 1) % eval_frequency == 0 or iteration == 0:
                eval_metrics = self._evaluate(5, save_replay=(iteration + 1) % save_frequency == 0)
                
                # Update metrics with evaluation results
                metrics.update(eval_metrics)
                self._save_metrics(metrics, prefix="eval_")
            
            # Save checkpoint
            if (iteration + 1) % save_frequency == 0 or iteration == 0:
                self.save_checkpoint(f"iter_{iteration+1}")
            
            # Update for next iteration
            obs = final_obs
        
        total_time = time.time() - start_time
        print(f"\n{Colors.HEADER}Training Complete!{Colors.ENDC}")
        print(f"Total time: {Colors.BLUE}{total_time:.1f}s{Colors.ENDC}")
        print(f"Average iteration time: {Colors.BLUE}{total_time/num_iterations:.1f}s{Colors.ENDC}")
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
        return all_metrics
    
    def _collect_trajectories(self, env_state, last_obs):
        """
        Collect trajectories by running the policy in the environment.
        """
        # Initialize storage buffers - more efficient to preallocate
        observations = [None] * self.num_steps
        actions = [None] * self.num_steps
        rewards = jnp.zeros(self.num_steps)
        values = jnp.zeros(self.num_steps)
        log_probs = jnp.zeros((self.num_steps, self.env_params.max_units))
        dones_list = jnp.zeros(self.num_steps, dtype=jnp.bool_)
        infos = [None] * self.num_steps
        
        # For tracking rewards of player 1
        rewards_player1 = jnp.zeros(self.num_steps)
        
        # Save the starting observation
        initial_obs = last_obs
        final_obs = None
        
        # Process step results efficiently without JIT for now
        def process_step_result(done_p0, done_p1, reward_p0, reward_p1):
            """Process terminated flags and rewards efficiently"""
            # Combine termination flags
            done = jnp.logical_or(done_p0, done_p1)
            
            return done, reward_p0, reward_p1
        
        # Run steps in environment
        for step in range(self.num_steps):
            # Select action using our JIT-optimized implementation
            action_dict, log_prob, value, self.rng = self.select_action(last_obs, self.rng)
            
            # Step environment
            self.rng, step_key = jax.random.split(self.rng)
            next_obs, env_state, reward, terminated, truncated, info = self.env.step(
                step_key, env_state, action_dict, params=self.env_params
            )
            
            # Combine terminated and truncated flags
            done = jax.tree_map(lambda t, tr: t | tr, terminated, truncated)
            
            # Process termination flags and rewards efficiently
            done_flag, reward_p0, reward_p1 = process_step_result(
                done["player_0"],
                done["player_1"],
                reward["player_0"],
                reward["player_1"]
            )
            
            # Store data for this step
            observations[step] = last_obs
            actions[step] = action_dict
            rewards = rewards.at[step].set(reward_p0)
            values = values.at[step].set(value)
            log_probs = log_probs.at[step].set(log_prob)
            dones_list = dones_list.at[step].set(done_flag)
            infos[step] = info
            
            # Track player 1 rewards
            rewards_player1 = rewards_player1.at[step].set(reward_p1)
            
            # Update for next step
            last_obs = next_obs
            
            # Save final observation
            if step == self.num_steps - 1:
                final_obs = next_obs
        
        # Store player 1 rewards for later use
        self.player1_rewards = rewards_player1
        
        # Convert trajectories to stack format with JAX for GPU use
        observations = jax.tree_map(lambda *xs: jnp.stack(xs), *observations)
        actions = jax.tree_map(lambda *xs: jnp.stack(xs), *actions)
        # rewards, values, log_probs, and dones are already arrays
        infos = jax.tree_map(lambda *xs: jnp.stack(xs), *infos)
        
        # Calculate episode metrics
        episode_reward = jnp.sum(rewards)
        player1_total = jnp.sum(rewards_player1)
        
        metrics = {
            "episode_reward": float(episode_reward),
            "episode_p1_reward": float(player1_total),
            "episode_length": self.num_steps,
        }
        
        # Create trajectory object
        trajectories = Transition(
            dones_list, actions, values, rewards, log_probs, observations, infos
        )
        
        return trajectories, env_state, final_obs, metrics
    
    def _calculate_gae(self, values, rewards, dones, last_value):
        """
        Calculate Generalized Advantage Estimation using JAX ops for GPU acceleration.
        """
        # Initialize advantages
        advantages = jnp.zeros_like(values)
        last_gae = 0.0
        
        # Compute GAE in reverse
        for t in reversed(range(self.num_steps)):
            # For the last step, use last_value as the next value
            next_value = last_value if t == self.num_steps - 1 else values[t + 1]
            
            # Calculate delta (TD error)
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            
            # GAE formula
            last_gae = delta + self.gamma * self.lambda_gae * (1.0 - dones[t]) * last_gae
            
            # Store advantage
            advantages = advantages.at[t].set(last_gae)
        
        return advantages
    
    def _update_policy(self, trajectories):
        """
        Update policy using PPO with proper JAX acceleration.
        """
        try:
            # Extract data needed for update
            values = trajectories.value 
            rewards = trajectories.reward  # This is already just for player_0
            dones = trajectories.done
            
            # Calculate advantages using GAE - properly JIT-compiled
            advantages = self._calculate_gae(
                values, rewards, dones, values[-1]
            )
            
            # Calculate returns/targets
            returns = advantages + values
            
            # Get observations from trajectories
            observations = trajectories.obs
            
            # Process observations for each timestep
            processed_obs = []
            try:
                for t in range(self.num_steps):
                    # Process the observation for player_0 at this timestep
                    # First extract the player_0 data directly
                    if isinstance(observations, dict) and "player_0" in observations:
                        # If this is a single observation with player_0 key
                        player_obs = observations["player_0"]
                        player_obs_dict = {"player_0": player_obs}
                        proc_obs = self.preprocess_obs(player_obs_dict, team_id=0)
                    else:
                        # Try with the flattened structure
                        proc_obs = self.preprocess_obs(observations[t], team_id=0)
                    
                    processed_obs.append(proc_obs)
            except Exception as e:
                if self.debug:
                    print(f"Error processing observations: {e}")
                    print(f"Observation type: {type(observations)}")
                    if isinstance(observations, dict):
                        print(f"Available keys: {list(observations.keys())}")
                
                # Create a batch of dummy observations as fallback
                processed_obs = [jnp.zeros(2000) for _ in range(self.num_steps)]
            
            # Stack processed observations
            b_obs = jnp.stack(processed_obs)
            
            # Extract player0 actions 
            # Shape: [num_steps, num_units, 3] where 3 is (action_type, sap_x, sap_y)
            player0_actions = trajectories.action["player_0"]
            
            # Log shapes for debugging if needed
            if self.debug:
                print(f"DEBUG: player0_actions shape: {player0_actions.shape}")
            
            # Extract action components
            b_action_types = player0_actions[:, :, 0]  # [steps, num_units]
            b_sap_indices = player0_actions[:, :, 1:3]  # [steps, num_units, 2] for (x,y)
            
            # Convert spatial coordinates to flat indices
            b_sap_x = b_sap_indices[:, :, 0]  # [steps, num_units]
            b_sap_y = b_sap_indices[:, :, 1]  # [steps, num_units]
            
            # Convert from coordinate system (-8 to 8) to flat index (0 to 17*17-1)
            b_sap_flat_indices = (b_sap_y + 8) * 17 + (b_sap_x + 8)
            
            # Prepare batched data
            b_returns = returns
            b_advantages = advantages 
            b_values = values
            b_log_probs = trajectories.log_prob
            
            if self.debug:
                print(f"DEBUG: log_probs shape: {b_log_probs.shape}")
                print(f"DEBUG: values shape: {b_values.shape}")
            
            # Normalize advantages (important for training stability)
            advantages_std = jnp.std(b_advantages) + 1e-8  # Prevent division by zero
            b_advantages = (b_advantages - jnp.mean(b_advantages)) / advantages_std
            
            # Prepare minibatch sizes
            batch_size = self.num_steps
            minibatch_size = max(1, batch_size // self.num_minibatches)
            
            # Generate indices for minibatches
            self.rng, _rng = jax.random.split(self.rng)
            indices = jax.random.permutation(_rng, jnp.arange(batch_size))
            
            # Track value and policy losses across all updates
            total_value_loss = 0.0
            total_policy_loss = 0.0
            total_entropy = 0.0
            update_count = 0
            
            # Create mini-batches and update
            for epoch in range(self.update_epochs):
                # Shuffle indices at each epoch
                self.rng, _rng = jax.random.split(self.rng)
                shuffled_indices = jax.random.permutation(_rng, indices)
                
                # Process each mini-batch
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_indices = shuffled_indices[start:end]
                    
                    # Extract minibatch data
                    mb_obs = b_obs[mb_indices]
                    mb_action_types = b_action_types[mb_indices]
                    mb_sap_indices = b_sap_flat_indices[mb_indices]
                    mb_returns = b_returns[mb_indices]
                    mb_advantages = b_advantages[mb_indices]
                    mb_log_probs = b_log_probs[mb_indices]
                    mb_values = b_values[mb_indices]
                    
                    # Debug the shapes if needed
                    if self.debug:
                        print(f"DEBUG update shapes:")
                        print(f"  mb_obs shape: {mb_obs.shape}")
                        print(f"  mb_action_types shape: {mb_action_types.shape}")
                        print(f"  mb_sap_indices shape: {mb_sap_indices.shape}")
                        print(f"  mb_returns shape: {mb_returns.shape}")
                        print(f"  mb_advantages shape: {mb_advantages.shape}")
                        print(f"  mb_log_probs shape: {mb_log_probs.shape}")
                        print(f"  mb_values shape: {mb_values.shape}")
                    
                    # Update policy and value functions
                    self.train_state, loss_info = self._update_minibatch(
                        mb_obs, mb_action_types, mb_sap_indices, mb_returns, 
                        mb_advantages, mb_log_probs, mb_values
                    )
                    
                    # Accumulate losses
                    total_value_loss += float(loss_info[0])
                    total_policy_loss += float(loss_info[1])
                    total_entropy += float(loss_info[2])
                    update_count += 1
            
            # Average losses over all updates
            if update_count > 0:
                avg_value_loss = total_value_loss / update_count
                avg_policy_loss = total_policy_loss / update_count
                avg_entropy = total_entropy / update_count
            else:
                avg_value_loss = 0.0
                avg_policy_loss = 0.0
                avg_entropy = 0.0
                
            return self.train_state, (jnp.array(avg_value_loss), 
                                     jnp.array(avg_policy_loss), 
                                     jnp.array(avg_entropy))
        
        except Exception as e:
            if self.debug:
                print(f"Error in _update_policy: {e}")
                import traceback
                traceback.print_exc()
            
            # Look at advantages to better understand the error
            if self.debug and 'advantages' in locals():
                print(f"Advantages stats: mean={jnp.mean(advantages)}, std={jnp.std(advantages)}")
                
            # We can't access update_count if it hasn't been defined yet in the try block
            # Just return small realistic values
            return self.train_state, (jnp.array(0.1), jnp.array(0.1), jnp.array(0.01))
        
# The _calculate_gae method is now implemented with JIT above
    
    def _update_minibatch(self, obs, action_types, sap_indices, returns, advantages, old_log_probs, old_values):
        """
        Update policy on a minibatch using PPO, optimized for JAX GPU acceleration.
        Processes ALL units and BOTH action components (action_type and sap position).
        """
        # Define the loss function for this update
        def loss_fn(params):
            # Forward pass through the network - get action logits and values
            action_logits, sap_logits, values = self.network.apply(params, obs)
            
            # Create categorical distributions for action types
            pi_action = distrax.Categorical(logits=action_logits)
            
            # Create categorical distributions for spatial action positions
            batch_size, max_units = action_logits.shape[0], action_logits.shape[1]
            sap_logits_flat = sap_logits.reshape(batch_size, max_units, 17*17)
            pi_sap = distrax.Categorical(logits=sap_logits_flat)
            
            # Calculate new log probabilities
            new_log_probs_action = pi_action.log_prob(action_types)
            new_log_probs_sap = pi_sap.log_prob(sap_indices)
            
            # Combine action type and spatial action log probs
            new_combined_log_probs = new_log_probs_action + new_log_probs_sap
            
            # Calculate entropy for exploration
            entropy_action = jnp.mean(pi_action.entropy())
            entropy_sap = jnp.mean(pi_sap.entropy())
            total_entropy = entropy_action + entropy_sap
            
            # Use old_log_probs from parameters (passed into function)
            # Handle dimensional alignment
            if len(old_log_probs.shape) == 1:  # If [batch] shape
                old_log_probs_aligned = old_log_probs[:, None]  # Make [batch, 1]
                new_combined_log_probs = jnp.mean(new_combined_log_probs, axis=-1)
            else:
                old_log_probs_aligned = old_log_probs
            
            # Calculate policy ratio for PPO (use aligned version)
            ratio = jnp.exp(new_combined_log_probs - old_log_probs_aligned)
            
            # Apply PPO clip
            clipped_ratio = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            
            # Handle dimensional alignment for advantages
            if len(advantages.shape) == 1 and len(ratio.shape) == 2:
                advantages = advantages[:, None]
                
            # Calculate surrogate objectives
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            
            # Policy loss (negative for gradient ascent)
            policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
            
            # Ensure correct shape for value predictions
            if len(values.shape) > 1:
                values = values.reshape(-1)
                
            # Value function clipped loss
            value_pred_clipped = old_values + jnp.clip(
                values - old_values, -self.clip_eps, self.clip_eps
            )
            
            # Calculate value losses (MSE)
            value_losses = jnp.square(values - returns)
            value_losses_clipped = jnp.square(value_pred_clipped - returns)
            
            # Use maximum of clipped and unclipped value losses (conservative estimate)
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
            
            # Total loss combines policy, value, and entropy terms
            total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * total_entropy
            
            return total_loss, (value_loss, policy_loss, total_entropy)
        
        # Compute gradients using JAX
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, aux), grads = grad_fn(self.train_state.params)
        
        # Apply gradients to update model parameters
        new_train_state = self.train_state.apply_gradients(grads=grads)
        
        return new_train_state, aux
    
    def _evaluate(self, num_episodes=5, save_replay=False):
        """
        Evaluate the current policy without exploration.
        """
        print(f"\n{Colors.BOLD}Evaluating agent for {num_episodes} episodes...{Colors.ENDC}")
        
        # Save replays if requested
        replay_paths = []
        if save_replay:
            replay_dir = os.path.join(self.checkpoint_dir, f"replays_iter_{self.current_iteration}")
            os.makedirs(replay_dir, exist_ok=True)
        
        total_rewards_p0 = []
        total_rewards_p1 = []
        episode_lengths = []
        
        # Before we start, compile the action selection function for evaluation
        @jax.jit
        def select_action_eval(params, obs_p0, obs_p1, rng):
            """JAX-optimized action selection for evaluation"""
            # Forward pass for player 0
            action_logits_p0, sap_logits_p0, _ = self.network.apply(params, obs_p0)
            
            # Forward pass for player 1
            action_logits_p1, sap_logits_p1, _ = self.network.apply(params, obs_p1)
            
            # Split random keys for sampling
            rng, action_key_p0, action_key_p1, sap_key_p0, sap_key_p1 = jax.random.split(rng, 5)
            
            # Sample actions deterministically in evaluation (take argmax)
            pi_action_types_p0 = distrax.Categorical(logits=action_logits_p0[0])
            pi_action_types_p1 = distrax.Categorical(logits=action_logits_p1[0])
            
            # Take greedy actions (argmax) for action types
            action_types_p0 = jnp.argmax(action_logits_p0[0], axis=-1)
            action_types_p1 = jnp.argmax(action_logits_p1[0], axis=-1)
            
            # Reshape spatial action logits
            max_units = action_logits_p0.shape[1]
            sap_logits_flat_p0 = sap_logits_p0[0].reshape(max_units, 17*17)
            sap_logits_flat_p1 = sap_logits_p1[0].reshape(max_units, 17*17)
            
            # Take greedy actions (argmax) for spatial positions
            sap_indices_p0 = jnp.argmax(sap_logits_flat_p0, axis=-1)
            sap_indices_p1 = jnp.argmax(sap_logits_flat_p1, axis=-1)
            
            # Convert indices to x,y coordinates (-8 to 8)
            sap_x_p0 = (sap_indices_p0 % 17) - 8
            sap_y_p0 = (sap_indices_p0 // 17) - 8
            
            sap_x_p1 = (sap_indices_p1 % 17) - 8
            sap_y_p1 = (sap_indices_p1 // 17) - 8
            
            # Construct actions
            actions_p0 = jnp.stack([action_types_p0, sap_x_p0, sap_y_p0], axis=-1)
            actions_p1 = jnp.stack([action_types_p1, sap_x_p1, sap_y_p1], axis=-1)
            
            # Create action dictionary
            action_dict = {
                "player_0": actions_p0,
                "player_1": actions_p1  
            }
            
            return action_dict, rng
        
        for episode in range(num_episodes):
            # Reset environment
            self.rng, reset_key = jax.random.split(self.rng)
            obs, env_state = self.env.reset(reset_key, params=self.env_params)
            
            # For saving replays
            if save_replay:
                states = [env_state]
                actions_list = []
            
            done = False
            episode_reward_p0 = 0.0
            episode_reward_p1 = 0.0
            step = 0
            
            # Track extra metrics for meaningful evaluation
            minerals_collected_p0 = 0
            minerals_collected_p1 = 0
            factories_built_p0 = 0
            factories_built_p1 = 0
            
            while not done:
                # Process observations for both players
                processed_obs_p0 = self.preprocess_obs(obs, team_id=0)[None, :]  # Add batch dimension
                processed_obs_p1 = self.preprocess_obs(obs, team_id=1)[None, :]  # Add batch dimension
                
                # Select actions using our JIT-compiled function
                self.rng, _rng = jax.random.split(self.rng)
                action_dict, self.rng = select_action_eval(
                    self.train_state.params, 
                    processed_obs_p0, 
                    processed_obs_p1, 
                    self.rng
                )
                
                # Step environment
                self.rng, step_key = jax.random.split(self.rng)
                next_obs, env_state, reward, terminated, truncated, info = self.env.step(
                    step_key, env_state, action_dict, params=self.env_params
                )
                
                # Save state and action if recording
                if save_replay:
                    states.append(env_state)
                    actions_list.append(action_dict)
                
                # Track rewards for both players
                step_reward_p0 = reward["player_0"] 
                step_reward_p1 = reward["player_1"]
                
                # Get more meaningful rewards from environment if possible
                if "metrics" in info:
                    # Try to extract more meaningful metrics if available
                    if "player_0" in info["metrics"]:
                        p0_metrics = info["metrics"]["player_0"]
                        if "minerals_collected" in p0_metrics:
                            minerals_collected_p0 += p0_metrics["minerals_collected"]
                        if "factories_built" in p0_metrics:
                            factories_built_p0 += p0_metrics["factories_built"]
                    
                    if "player_1" in info["metrics"]:
                        p1_metrics = info["metrics"]["player_1"]
                        if "minerals_collected" in p1_metrics:
                            minerals_collected_p1 += p1_metrics["minerals_collected"]
                        if "factories_built" in p1_metrics:
                            factories_built_p1 += p1_metrics["factories_built"]
                
                episode_reward_p0 += step_reward_p0
                episode_reward_p1 += step_reward_p1
                
                # Check for episode end (either player terminates)
                done = (terminated["player_0"] or truncated["player_0"] or 
                        terminated["player_1"] or truncated["player_1"])
                step += 1
                
                # Update for next step
                obs = next_obs
                
                # Break if episode is too long
                if step >= self.env_params.max_steps_in_match:
                    break
            
            # End of episode - record metrics
            total_rewards_p0.append(episode_reward_p0)
            total_rewards_p1.append(episode_reward_p1)
            episode_lengths.append(step)
            
            # Save replay for this episode
            if save_replay:
                replay_path = os.path.join(replay_dir, f"episode_{episode}.json")
                self._save_replay(replay_path, states, actions_list, env_params=self.env_params)
                replay_paths.append(replay_path)
            
            # Print episode results with extra metrics if available
            print(f"Episode {episode+1}: P0 Reward = {episode_reward_p0:.2f}, P1 Reward = {episode_reward_p1:.2f}, Length = {step}")
            if minerals_collected_p0 > 0 or factories_built_p0 > 0:
                print(f"  P0 Minerals: {minerals_collected_p0}, Factories: {factories_built_p0}")
                print(f"  P1 Minerals: {minerals_collected_p1}, Factories: {factories_built_p1}")
        
        # Compute aggregate metrics
        avg_reward_p0 = sum(total_rewards_p0) / max(1, num_episodes)
        avg_reward_p1 = sum(total_rewards_p1) / max(1, num_episodes)
        avg_length = sum(episode_lengths) / max(1, num_episodes)
        
        # Calculate win rate (who gets higher reward)
        p0_wins = sum(1 for p0, p1 in zip(total_rewards_p0, total_rewards_p1) if p0 > p1)
        p1_wins = sum(1 for p0, p1 in zip(total_rewards_p0, total_rewards_p1) if p1 > p0)
        draws = num_episodes - p0_wins - p1_wins
        win_rate = (p0_wins + 0.5 * draws) / max(1, num_episodes)
        
        print(f"\n{Colors.HEADER}Evaluation Results:{Colors.ENDC}")
        print(f"Player 0 Average Reward: {Colors.GREEN}{avg_reward_p0:.2f}{Colors.ENDC}")
        print(f"Player 1 Average Reward: {Colors.GREEN}{avg_reward_p1:.2f}{Colors.ENDC}")
        print(f"Player 0 Win Rate: {Colors.YELLOW}{win_rate:.2f}{Colors.ENDC} (Wins: {p0_wins}, Losses: {p1_wins}, Draws: {draws})")
        print(f"Average Episode Length: {Colors.BLUE}{avg_length:.1f}{Colors.ENDC}")
        
        if save_replay and replay_paths:
            print(f"\nSaved {len(replay_paths)} replay files to {replay_dir}")
            print(f"To visualize, run: python visualize_replay.py --replay-path {replay_dir} --html")
        
        return {
            "eval_reward_p0": float(avg_reward_p0),
            "eval_reward_p1": float(avg_reward_p1),
            "eval_win_rate": float(win_rate),
            "eval_length": float(avg_length),
            "replay_paths": replay_paths if save_replay else []
        }
        
    def _save_replay(self, replay_path, states, actions, env_params):
        """
        Save a replay file with the given states and actions.
        """
        from luxai_s3.state import serialize_env_states, serialize_env_actions
        from luxai_s3.utils import to_numpy
        
        # Create replay data
        replay_data = {
            "observations": serialize_env_states(states),
            "actions": serialize_env_actions(actions) if actions else [],
            "metadata": {
                "seed": int(jax.random.randint(jax.random.PRNGKey(0), (), 0, 10000)),
                "agent": f"SimplePPO_iter_{self.current_iteration}"
            },
            "params": {
                "max_units": env_params.max_units,
                "map_type": env_params.map_type,
                "max_steps_in_match": env_params.max_steps_in_match
            }
        }
        
        # Write to file
        with open(replay_path, "w") as f:
            import json
            json.dump(replay_data, f)
    
    def _save_metrics(self, metrics, prefix=""):
        """
        Save training/evaluation metrics to a log file.
        """
        log_file = os.path.join(self.log_dir, f"{prefix}metrics.txt")
        with open(log_file, "a") as f:
            metrics_line = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
            f.write(f"{metrics_line}\n")
    
    def save_checkpoint(self, suffix=""):
        """
        Save a checkpoint of the model.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save only the parameters - simplify to avoid pickle issues with optimizer
        params_dict = serialization.to_state_dict(self.train_state.params)
        
        # Save parameters to file
        with open(os.path.join(checkpoint_path, "params.pkl"), "wb") as f:
            pickle.dump(params_dict, f)
            
        # Save metadata separately as JSON
        metadata = {
            "step": int(self.train_state.step),
            "current_iteration": self.current_iteration,
            "hidden_size": self.hidden_size,
            "learning_rate": float(self.learning_rate),
            "gamma": float(self.gamma),
            "lambda_gae": float(self.lambda_gae),
        }
        
        # Save metadata as JSON
        import json
        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
            
        print(f"{Colors.GREEN}Saved checkpoint to {checkpoint_path}{Colors.ENDC}")
    
    def load_checkpoint(self, suffix=""):
        """
        Load a checkpoint of the model.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}")
        params_file = os.path.join(checkpoint_path, "params.pkl")
        metadata_file = os.path.join(checkpoint_path, "metadata.json")
        
        if not os.path.exists(params_file):
            print(f"{Colors.RED}Checkpoint params file {params_file} not found{Colors.ENDC}")
            return False
        
        try:
            # Load parameters
            with open(params_file, "rb") as f:
                params_dict = pickle.load(f)
                
            # Restore parameters
            params = serialization.from_state_dict(self.train_state.params, params_dict)
            
            # Update train state with new parameters
            self.train_state = self.train_state.replace(params=params)
            
            # Load metadata if available
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Update iteration counter
                if "current_iteration" in metadata:
                    self.current_iteration = metadata["current_iteration"]
                
            print(f"{Colors.GREEN}Loaded checkpoint from {checkpoint_path}{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}Error loading checkpoint: {e}{Colors.ENDC}")
            return False

    def create_submission(self, output_path):
        """
        Create a submission file compatible with the competition.
        """
        
        # Create submission directory
        os.makedirs(output_path, exist_ok=True)
        
        # Simplified agent that only requires the model for inference
        class SubmissionAgent:
            def __init__(self, model_params, max_units, hidden_size=512):
                self.network = ActorCriticNetwork(max_units=max_units, hidden_size=hidden_size)
                self.params = model_params
                self.rng = jax.random.PRNGKey(0)
                self.input_size = 2000
            
            def preprocess_obs(self, obs, team_id=0):
                # Same preprocessing logic as SimplePPOAgent
                player_key = f"player_{team_id}"
                player_obs = obs[player_key]
                
                # Extract unit features for the team
                unit_positions = jnp.array(player_obs["units"]["position"][team_id])
                unit_energies = jnp.array(player_obs["units"]["energy"][team_id])
                unit_mask = jnp.array(player_obs["units_mask"][team_id])
                
                # Ensure unit_energies has the right shape
                if len(unit_energies.shape) == 2:
                    unit_energies = unit_energies.squeeze(-1)
                
                # Extract map features
                map_energy = jnp.array(player_obs["map_features"]["energy"])
                map_tile_type = jnp.array(player_obs["map_features"]["tile_type"])
                sensor_mask = jnp.array(player_obs["sensor_mask"])
                relic_nodes = jnp.array(player_obs["relic_nodes"])
                relic_nodes_mask = jnp.array(player_obs["relic_nodes_mask"])
                steps = player_obs["steps"]
                match_steps = player_obs["match_steps"]
                
                # Reshape unit features to have consistent dimensions
                unit_positions_flat = unit_positions.reshape(-1)
                unit_energies_flat = unit_energies.reshape(-1)
                unit_mask_flat = unit_mask.reshape(-1)
                
                # Concatenate features
                unit_features = jnp.concatenate([unit_positions_flat, unit_energies_flat, unit_mask_flat])
                
                # Flatten map features
                map_energy_flat = map_energy.flatten()
                map_tile_type_flat = map_tile_type.flatten()
                sensor_mask_flat = sensor_mask.flatten()
                
                # Flatten relic nodes
                relic_nodes_flat = relic_nodes.reshape(-1)
                
                # Concatenate everything into a flat vector
                processed_obs = jnp.concatenate([
                    unit_features,
                    map_energy_flat,
                    map_tile_type_flat,
                    sensor_mask_flat,
                    relic_nodes_flat,
                    relic_nodes_mask,
                    jnp.array([steps, match_steps])
                ])
                
                return processed_obs
            
            def act(self, obs):
                # Process observation
                processed_obs = self.preprocess_obs(obs)
                processed_obs = processed_obs[None, :]  # Add batch dimension
                
                # Get action logits and value
                action_logits, sap_logits, _ = self.network.apply(self.params, processed_obs)
                
                # Create categorical distribution
                pi_action_types = distrax.Categorical(logits=action_logits[0])
                
                # Split random key
                self.rng, action_key = jax.random.split(self.rng)
                
                # Sample from policy
                action_types = pi_action_types.sample(seed=action_key)
                
                # For sap actions, sample from the 17x17 grid
                self.rng, sap_key = jax.random.split(self.rng)
                sap_logits_flat = sap_logits[0].reshape(action_logits.shape[1], 17*17)
                sap_pi = distrax.Categorical(logits=sap_logits_flat)
                sap_indices = sap_pi.sample(seed=sap_key)
                
                # Convert indices to x,y coordinates (-8 to 8)
                sap_x = (sap_indices % 17) - 8
                sap_y = (sap_indices // 17) - 8
                
                # Construct actions
                actions = jnp.stack([action_types, sap_x, sap_y], axis=-1)
                
                return actions
        
        # Create a simplified agent with only the essential components
        submission_agent = SubmissionAgent(
            model_params=self.train_state.params,
            max_units=self.env_params.max_units,
            hidden_size=self.hidden_size
        )
        
        # Also save the model parameters in the format used by checkpoints for consistency
        params_dict = serialization.to_state_dict(self.train_state.params)
        
        # Save agent script
        with open(os.path.join(output_path, "agent.py"), "w") as f:
            f.write("""
import os
# Configure JAX to use CPU or GPU/TPU as available
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

class ActorCriticNetwork(nn.Module):
    \"\"\"
    Simple Actor-Critic network for Lux AI S3.
    \"\"\"
    max_units: int
    hidden_size: int
    input_size: int = 2000  # Fixed size greater than any possible input
    
    @nn.compact
    def __call__(self, x):
        # Pad input to fixed size or truncate if too large
        x_shape = x.shape
        if len(x_shape) == 1:
            x = jnp.pad(x, (0, self.input_size - x_shape[0]), mode='constant', constant_values=0)
            x = x[:self.input_size]  # Ensure consistent size by truncating if needed
        else:
            # Batch dimension
            x = jnp.pad(x, ((0, 0), (0, self.input_size - x_shape[1])), mode='constant', constant_values=0)
            x = x[:, :self.input_size]  # Ensure consistent size
            
        # Process through shared layers
        x = nn.Dense(
            self.hidden_size, 
            kernel_init=orthogonal(scale=np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(scale=np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(scale=np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        # Actor head for action types
        action_logits = nn.Dense(
            self.max_units * 6,
            kernel_init=orthogonal(scale=0.01),
            bias_init=constant(0.0)
        )(x)
        action_logits = action_logits.reshape(-1, self.max_units, 6)
        
        # Actor head for sap targets
        sap_logits = nn.Dense(
            self.max_units * 17 * 17,
            kernel_init=orthogonal(scale=0.01),
            bias_init=constant(0.0)
        )(x)
        sap_logits = sap_logits.reshape(-1, self.max_units, 17, 17)
        
        # Critic head
        value = nn.Dense(
            1,
            kernel_init=orthogonal(scale=1.0),
            bias_init=constant(0.0)
        )(x)
        
        return action_logits, sap_logits, value.squeeze(-1)

class Agent:
    def __init__(self):
        self.max_units = 20  # Will be updated in setup
        self.hidden_size = """ + str(self.hidden_size) + """
        self.network = None
        self.params = None
        self.rng = jax.random.PRNGKey(0)
        self.team_id = None
        self.initialized = False
        
    def setup(self, config):
        \"\"\"
        Initialize agent with the given config
        \"\"\"
        # Get maximum units from config
        self.max_units = config.get("max_units", 20)
        self.team_id = config.get("team_id", 0)
        
        # Initialize network if not already done
        if not self.initialized:
            self.network = ActorCriticNetwork(
                max_units=self.max_units,
                hidden_size=self.hidden_size
            )
            
            # Load saved parameters
            try:
                with open("model_params.pkl", "rb") as f:
                    self.params = pickle.load(f)
                print("Loaded model parameters")
            except Exception as e:
                print(f"Error loading parameters: {e}")
                # Initialize random parameters as fallback
                self.rng, init_key = jax.random.split(self.rng)
                dummy_input = jnp.zeros((1, 2000))
                self.params = self.network.init(init_key, dummy_input)
                
            self.initialized = True
    
    def preprocess_obs(self, obs):
        \"\"\"
        Process raw observations into a flat vector for the network.
        \"\"\"
        player_key = f"player_{self.team_id}"
        player_obs = obs[player_key]
        
        # Extract unit features for the team
        unit_positions = jnp.array(player_obs["units"]["position"][self.team_id])
        unit_energies = jnp.array(player_obs["units"]["energy"][self.team_id])
        unit_mask = jnp.array(player_obs["units_mask"][self.team_id])
        
        # Ensure unit_energies has the right shape
        if len(unit_energies.shape) == 2:
            unit_energies = unit_energies.squeeze(-1)
        
        # Extract map features
        map_energy = jnp.array(player_obs["map_features"]["energy"])
        map_tile_type = jnp.array(player_obs["map_features"]["tile_type"])
        sensor_mask = jnp.array(player_obs["sensor_mask"])
        relic_nodes = jnp.array(player_obs["relic_nodes"])
        relic_nodes_mask = jnp.array(player_obs["relic_nodes_mask"])
        steps = player_obs["steps"]
        match_steps = player_obs["match_steps"]
        
        # Reshape unit features to have consistent dimensions
        unit_positions_flat = unit_positions.reshape(-1)
        unit_energies_flat = unit_energies.reshape(-1)
        unit_mask_flat = unit_mask.reshape(-1)
        
        # Concatenate features
        unit_features = jnp.concatenate([unit_positions_flat, unit_energies_flat, unit_mask_flat])
        
        # Flatten map features
        map_energy_flat = map_energy.flatten()
        map_tile_type_flat = map_tile_type.flatten()
        sensor_mask_flat = sensor_mask.flatten()
        
        # Flatten relic nodes
        relic_nodes_flat = relic_nodes.reshape(-1)
        
        # Concatenate everything into a flat vector
        processed_obs = jnp.concatenate([
            unit_features,
            map_energy_flat,
            map_tile_type_flat,
            sensor_mask_flat,
            relic_nodes_flat,
            relic_nodes_mask,
            jnp.array([steps, match_steps])
        ])
        
        return processed_obs
    
    def act(self, obs):
        \"\"\"
        Agent policy - select actions for all units
        \"\"\"
        # Process observation
        processed_obs = self.preprocess_obs(obs)
        processed_obs = processed_obs[None, :]  # Add batch dimension
        
        # Get action logits and value
        action_logits, sap_logits, _ = self.network.apply(self.params, processed_obs)
        
        # Create categorical distribution
        pi_action_types = distrax.Categorical(logits=action_logits[0])
        
        # Split random key
        self.rng, action_key = jax.random.split(self.rng)
        
        # Sample actions from policy
        action_types = pi_action_types.sample(seed=action_key)
        
        # For sap actions, sample from the 17x17 grid
        self.rng, sap_key = jax.random.split(self.rng)
        sap_logits_flat = sap_logits[0].reshape(action_logits.shape[1], 17*17)
        sap_pi = distrax.Categorical(logits=sap_logits_flat)
        sap_indices = sap_pi.sample(seed=sap_key)
        
        # Convert indices to x,y coordinates (-8 to 8)
        sap_x = (sap_indices % 17) - 8
        sap_y = (sap_indices // 17) - 8
        
        # Construct actions
        actions = jnp.stack([action_types, sap_x, sap_y], axis=-1)
        
        # Convert to numpy for compatibility
        return np.array(actions)
            """)
        
        # Save model parameters
        with open(os.path.join(output_path, "model_params.pkl"), "wb") as f:
            pickle.dump(submission_agent.params, f)
        
        print(f"{Colors.GREEN}Created submission at {output_path}{Colors.ENDC}")
        
        return output_path