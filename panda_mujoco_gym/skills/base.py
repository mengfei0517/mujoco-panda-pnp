from __future__ import annotations

import numpy as np
import abc


class Skill(abc.ABC):
    """Abstract scripted skill that outputs one 7‑D action each `step()`."""

    def __init__(self, env):
        self.env = env  # FrankaEnv
        self.done = False

    # ------------------------------------------------------------------
    # Mandatory overrides
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def reset(self):
        """Reset internal counters / pre‑compute trajectories."""
        self.done = False

    @abc.abstractmethod
    def step(self) -> np.ndarray:
        """Return a single low‑level action (shape==(7,))."""

    # ------------------------------------------------------------------
    def is_done(self) -> bool:
        return self.done

    # Convenience -------------------------------------------------------
    def zero_action(self) -> np.ndarray:
        return np.zeros_like(self.env.action_space.low, dtype=np.float32)


    def _step_sim(self, n: int = 1):
        """Advance physics `n` mj-step(s) then render (if viewer exists)."""
        mujoco = self.env.unwrapped
        for _ in range(n):
            mujoco._mujoco.mj_step(mujoco.model, mujoco.data, nstep=1)
        if hasattr(self.env, "render"):
            self.env.render()


    # -----------------------------------------------------------------------------
    # Common perception helpers for skill termination conditions
    # -----------------------------------------------------------------------------
    @staticmethod
    def pos_close(pos1: np.ndarray, pos2: np.ndarray, thresh: float = 0.01) -> bool:
        """Whether two positions are within a threshold."""
        return np.linalg.norm(pos1 - pos2) < thresh


    @staticmethod
    def quat_close(q1: np.ndarray, q2: np.ndarray, thresh: float = 0.01) -> bool:
        """Whether two quaternions are approximately aligned."""
        return 1.0 - abs(np.dot(q1, q2)) < thresh


    @staticmethod
    def fingers_closed(width: float, thresh: float = 0.2) -> bool:
        """Whether gripper is (almost) fully closed."""
        return width < thresh


    @staticmethod
    def fingers_open(width: float, thresh: float = 0.08) -> bool:
        """Whether gripper is (almost) fully open."""
        return width > thresh


    @staticmethod
    def lifted_enough(z_now: float, z_init: float, dz: float, thresh: float = 0.005) -> bool:
        """Whether object is lifted by at least dz from initial z."""
        return z_now > z_init + dz - thresh


    @staticmethod
    def retreated_enough(p_now: np.ndarray, p_target: np.ndarray, thresh: float = 0.01) -> bool:
        """Whether current position is close to retreat target."""
        return np.linalg.norm(p_now - p_target) < thresh
