from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from .base import Skill

class MoveSkill(Skill):
    """Move EE from current to target position in a straight line (fixed orientation)."""

    def __init__(self, env, target_pos: np.ndarray, steps: int = 30, pos_thresh: float = 0.02):
        super().__init__(env)
        assert pos_thresh > 0, "pos_thresh must be positive"

        self.target_pos = np.asarray(target_pos, float)
        self.steps = steps
        self.pos_thresh = pos_thresh

        # runtime state
        self.i = 0
        self.done = False

    def reset(self):
        self.i = 0
        self.done = False
        self.start_pos = self.env.get_ee_position().copy()
        self.quat = self.env.get_ee_orientation().copy()
        # 动态调整插值步数
        dist = np.linalg.norm(self.start_pos - self.target_pos)
        if dist > 1.0:
            steps = 120
        elif dist > 0.5:
            steps = 60
        else:
            steps = 20
        self.steps = steps
        self.pos_traj = np.linspace(self.start_pos, self.target_pos, self.steps)

    def step(self):
        if self.done:
            return self.zero_action()

        if self.i < self.steps:
            pos = self.pos_traj[self.i]
            self.env.set_mocap_pose(pos, self.quat)
            self._step_sim(n=5)
            self.i += 1
        else:
            self.env.set_mocap_pose(self.target_pos, self.quat)
            if Skill.pos_close(self.env.get_ee_position(), self.target_pos, self.pos_thresh):
                self.done = True

        return self.zero_action()

