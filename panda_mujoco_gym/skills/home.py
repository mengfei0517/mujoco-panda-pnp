from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .base import Skill


def interpolate_pose(p0, p1, q0, q1, n):
    pos = np.linspace(p0, p1, n)
    key = R.from_quat([q0, q1])
    quat = Slerp([0, 1], key)(np.linspace(0, 1, n)).as_quat()
    return pos, quat


class HomeSkill(Skill):
    """Smoothly return to env.home_pos/quat and open gripper."""

    def __init__(self, env, steps: int = 30):
        super().__init__(env)
        self.steps = steps

    def reset(self):
        self.i = 0
        self.done = False
        self.start_pos = self.env.get_ee_position().copy()
        self.start_quat = self.env.get_ee_orientation().copy()

        # fallback if home_pos not set
        self.goal_pos = getattr(self.env, "home_pos", self.start_pos)
        self.goal_quat = getattr(self.env, "home_quat", self.start_quat)

        self.pos_traj, self.quat_traj = interpolate_pose(
            self.start_pos, self.goal_pos, self.start_quat, self.goal_quat, self.steps
        )

    def step(self):
        if self.done:
            return np.zeros(7, dtype=np.float32)

        pos = self.pos_traj[self.i]
        quat = self.quat_traj[self.i]
        self.env.set_mocap_pose(pos, quat)
        self._step_sim(n=5)

        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0  # open gripper

        self.i += 1
        self.done = self.i >= self.steps
        return action

