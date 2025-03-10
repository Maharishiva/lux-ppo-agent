#!/usr/bin/env python
import os
import argparse
import pickle
import flax
import jax
import jax.numpy as jnp
from simple_transformer import ActorCriticNetwork, Colors
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams

def parse_args():
    parser = argparse.ArgumentParser(description="Create a submission for Lux AI S3 competition")
    
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="submission", help="Output directory for submission")
    parser.add_argument("--hidden-size", type=int, default=512, help="Size of hidden layers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n{Colors.HEADER}Creating Lux AI S3 Submission{Colors.ENDC}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize environment parameters
    env_params = EnvParams(map_type=0)
    
    # Load parameters from checkpoint if provided
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        params_file = os.path.join(checkpoint_path, "params.pkl")
        
        if os.path.exists(params_file):
            print(f"{Colors.BLUE}Loading parameters from {params_file}{Colors.ENDC}")
            with open(params_file, "rb") as f:
                params_dict = pickle.load(f)
        else:
            print(f"{Colors.RED}Parameters file {params_file} not found, initializing random parameters{Colors.ENDC}")
            
            # Initialize network with random parameters
            network = ActorCriticNetwork(
                max_units=env_params.max_units,
                hidden_size=args.hidden_size
            )
            dummy_input = jnp.zeros((1, 2000))
            params_dict = network.init(jax.random.PRNGKey(0), dummy_input)
    else:
        print(f"{Colors.YELLOW}No checkpoint provided, initializing random parameters{Colors.ENDC}")
        
        # Initialize network with random parameters
        network = ActorCriticNetwork(
            max_units=env_params.max_units,
            hidden_size=args.hidden_size
        )
        dummy_input = jnp.zeros((1, 2000))
        params_dict = network.init(jax.random.PRNGKey(0), dummy_input)
    
    # Create agent script
    agent_script = """
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
        self.hidden_size = %d
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
        sap_x = (sap_indices %% 17) - 8
        sap_y = (sap_indices // 17) - 8
        
        # Construct actions
        actions = jnp.stack([action_types, sap_x, sap_y], axis=-1)
        
        # Convert to numpy for compatibility
        return np.array(actions)
""" % args.hidden_size
    
    # Save agent script
    with open(os.path.join(args.output_dir, "agent.py"), "w") as f:
        f.write(agent_script)
    
    # Save model parameters
    with open(os.path.join(args.output_dir, "model_params.pkl"), "wb") as f:
        pickle.dump(params_dict, f)
    
    print(f"\n{Colors.GREEN}Successfully created submission at {args.output_dir}{Colors.ENDC}")
    print(f"{Colors.BLUE}Files created:{Colors.ENDC}")
    print(f"  - {Colors.YELLOW}agent.py{Colors.ENDC}: Agent implementation")
    print(f"  - {Colors.YELLOW}model_params.pkl{Colors.ENDC}: Model parameters")
    print(f"\n{Colors.BOLD}To test the submission:{Colors.ENDC}")
    print(f"  1. cd {args.output_dir}")
    print(f"  2. Run a local match or upload to competition")

if __name__ == "__main__":
    main()