from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .base import Skill


class RotateSkill(Skill):
    """Rotate the ee *in place* by `delta_quat` over `steps` ticks."""

    def __init__(
        self,
        env,
        delta_quat: np.ndarray,
        steps: int = 50,
        err_thresh: float = 0.01,
    ):
        super().__init__(env)
        assert len(delta_quat) == 4, "delta_quat must be xyzw quaternion"
        self.delta_quat = np.asarray(delta_quat, dtype=float)
        self.steps = max(1, steps)
        self.err_thresh = err_thresh

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------
    def reset(self):
        self.i = 0
        self.done = False

        self.start_pos = self.env.get_ee_position().copy()
        self.start_quat = self.env.get_ee_orientation().copy()

        self.target_quat = (
            R.from_quat(self.start_quat) * R.from_quat(self.delta_quat)
        ).as_quat()

        # Pre-compute quaternion trajectory (inclusive of both ends)
        key_rots = R.from_quat([self.start_quat, self.target_quat])
        self.quat_traj = Slerp([0, 1], key_rots)(
            np.linspace(0, 1, self.steps, endpoint=True)
        ).as_quat()

    # ------------------------------------------------------------------
    # One control tick
    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        """Return a 7-DoF action; zero vector except we optionally open/close gripper."""
        if self.done:
            return self.zero_action()

        # Guard against index overflow
        if self.i >= self.steps:
            self.done = True
            return self.zero_action()

        # 1) command pose
        quat_cmd = self.quat_traj[self.i]
        self.env.set_mocap_pose(self.start_pos, quat_cmd)

        self._step_sim(n=5)

        # 2) progress index
        self.i += 1

        # 3) success check every tick – useful on real robot with drift
        if Skill.quat_close(self.env.get_ee_orientation(), self.target_quat, self.err_thresh):
            self.done = True

        return self.zero_action()
