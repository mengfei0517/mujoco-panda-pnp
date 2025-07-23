import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class ResidualActionWrapper(gym.ActionWrapper):
    """
    env.action = baseline_action + residual_scale * action_raw
    where action_raw ∈ [-1, 1]^n is produced by the RL policy.
    """

    def __init__(self, env: gym.Env, residual_scale=0.2):
        super().__init__(env)
        self.residual_scale = np.asarray(residual_scale, dtype=np.float32)

        # residual 动作空间：与原动作同维，但范围 [-1, 1]
        low = -np.ones_like(env.action_space.low, dtype=np.float32)
        high = np.ones_like(env.action_space.high, dtype=np.float32)
        self.action_space = Box(low, high, dtype=np.float32)

    # 核心：合成动作
    def action(self, residual: np.ndarray):
        a_base = self.env.compute_baseline_action()
        delta = self.residual_scale * residual
        return np.clip(
            a_base + delta,
            self.env.action_space.low,
            self.env.action_space.high,
        )
