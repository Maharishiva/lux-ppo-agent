#!/usr/bin/env python
"""
Minimal validation script for testing the PPO agent logic without 
requiring the full luxai environment.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, NamedTuple, Any

# Mock the environment classes/components we need
class EnvParams:
    def __init__(self, map_type=0, max_steps_in_match=100):
        self.map_type = map_type
        self.max_steps_in_match = max_steps_in_match
        self.max_units = 20  # Mock value

class MockLuxAIS3Env:
    def __init__(self):
        pass
        
    def reset(self, key, params=None):
        # Return mock observation and state
        obs = self._create_mock_obs()
        state = "mock_state"
        return obs, state
        
    def step(self, key, state, actions, params=None):
        # Return mock next observation, state, reward, etc.
        next_obs = self._create_mock_obs()
        next_state = "mock_next_state"
        reward = {"player_0": jnp.array(1.0), "player_1": jnp.array(-1.0)}
        terminated = {"player_0": False, "player_1": False}
        truncated = {"player_0": False, "player_1": False}
        info = {"mock_info": True}
        return next_obs, next_state, reward, terminated, truncated, info
        
    def _create_mock_obs(self):
        # Create a mock observation with the expected structure
        obs = {
            "player_0": {
                "units": {
                    "position": {0: jnp.ones((20, 2)), 1: jnp.ones((20, 2))},
                    "energy": {0: jnp.ones((20,)), 1: jnp.ones((20,))}
                },
                "units_mask": {0: jnp.ones((20,)), 1: jnp.ones((20,))},
                "map_features": {
                    "energy": jnp.ones((16, 16)),
                    "tile_type": jnp.ones((16, 16))
                },
                "sensor_mask": jnp.ones((16, 16)),
                "relic_nodes": jnp.ones((5, 2)),
                "relic_nodes_mask": jnp.ones((5,)),
                "steps": 0,
                "match_steps": 0
            },
            "player_1": {
                "units": {
                    "position": {0: jnp.ones((20, 2)), 1: jnp.ones((20, 2))},
                    "energy": {0: jnp.ones((20,)), 1: jnp.ones((20,))}
                },
                "units_mask": {0: jnp.ones((20,)), 1: jnp.ones((20,))},
                "map_features": {
                    "energy": jnp.ones((16, 16)),
                    "tile_type": jnp.ones((16, 16))
                },
                "sensor_mask": jnp.ones((16, 16)),
                "relic_nodes": jnp.ones((5, 2)),
                "relic_nodes_mask": jnp.ones((5,)),
                "steps": 0,
                "match_steps": 0
            }
        }
        return obs

# Transition class for our tests
class Transition(NamedTuple):
    done: jnp.ndarray
    action: Dict[str, jnp.ndarray]
    value: jnp.ndarray
    reward: Dict[str, jnp.ndarray]
    log_prob: jnp.ndarray
    obs: Dict[str, Any]
    info: Dict[str, Any]

# Mock just enough of SimplePPOAgent for testing
def test_action_selection():
    """Test the action selection logic"""
    from simple_transformer import SimplePPOAgent
    
    # Create a mock environment
    env = MockLuxAIS3Env()
    env_params = EnvParams()
    
    # Create agent with tiny params for quick testing
    agent = SimplePPOAgent(
        env=env,
        hidden_size=16,  # Tiny for fast init/test
        learning_rate=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=1,
        num_minibatches=1,
        num_envs=1,
        num_steps=2,  # Very short
        anneal_lr=False,
        debug=True,
        checkpoint_dir="test_checkpoints",
        log_dir="test_logs",
        seed=0,
        env_params=env_params
    )
    
    # Test action selection
    print("Testing action selection...")
    obs, _ = env.reset(jax.random.PRNGKey(0))
    
    action_dict, log_probs, value, _ = agent.select_action(
        obs, jax.random.PRNGKey(1)
    )
    
    # Check that the action dict has entries for both players
    assert "player_0" in action_dict, "player_0 missing from action_dict"
    assert "player_1" in action_dict, "player_1 missing from action_dict"
    
    # Check that the actions are different for the two players
    assert not jnp.array_equal(action_dict["player_0"], action_dict["player_1"]), "Actions for both players are identical"
    
    # Check shapes
    assert action_dict["player_0"].shape == (20, 3), f"Wrong shape for player_0 actions: {action_dict['player_0'].shape}"
    assert action_dict["player_1"].shape == (20, 3), f"Wrong shape for player_1 actions: {action_dict['player_1'].shape}"
    
    print("Action selection test passed!")
    
    return True

def test_ppo_update():
    """Test the PPO update logic"""
    from simple_transformer import SimplePPOAgent
    
    # Create a mock environment
    env = MockLuxAIS3Env()
    env_params = EnvParams()
    
    # Create agent with tiny params for quick testing
    agent = SimplePPOAgent(
        env=env,
        hidden_size=16,  # Tiny for fast init/test
        learning_rate=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=1,
        num_minibatches=1,
        num_envs=1,
        num_steps=2,  # Very short for test
        anneal_lr=False,
        debug=True,
        checkpoint_dir="test_checkpoints",
        log_dir="test_logs",
        seed=0,
        env_params=env_params
    )
    
    # Create some mock trajectories
    # For a simple test, we'll manually create trajectory data
    mock_dones = jnp.zeros((2,))  # 2 steps
    
    # Create mock actions for both players
    mock_action_p0 = jnp.ones((20, 3)) * 2  # action_type=2, sap_x=2, sap_y=2
    mock_action_p1 = jnp.ones((20, 3)) * 3  # action_type=3, sap_x=3, sap_y=3
    
    mock_actions = {
        "player_0": jnp.stack([mock_action_p0, mock_action_p0]),  # 2 steps
        "player_1": jnp.stack([mock_action_p1, mock_action_p1])   # 2 steps
    }
    
    mock_values = jnp.ones((2,))  # 2 steps
    mock_rewards = jnp.ones((2,))  # 2 steps
    mock_log_probs = jnp.zeros((2, 20))  # 2 steps, 20 units
    
    # Create mock observations
    mock_obs = []
    for _ in range(2):  # 2 steps
        mock_obs.append(env._create_mock_obs())
    
    # Create mock infos
    mock_infos = [{"mock": True}, {"mock": True}]
    
    # Create the trajectory
    trajectories = Transition(
        mock_dones, 
        mock_actions, 
        mock_values, 
        mock_rewards, 
        mock_log_probs, 
        mock_obs, 
        mock_infos
    )
    
    # Test the PPO update
    print("Testing PPO update...")
    try:
        new_state, loss_info = agent._update_policy(trajectories)
        print("PPO update test passed!")
        return True
    except Exception as e:
        print(f"PPO update test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the tests
    print("Starting validation tests...")
    action_test_passed = test_action_selection()
    ppo_update_passed = test_ppo_update()
    
    if action_test_passed and ppo_update_passed:
        print("\n✅ All tests passed! The fixes appear to be working correctly.")
    else:
        print("\n❌ Some tests failed. The fixes need more work.")