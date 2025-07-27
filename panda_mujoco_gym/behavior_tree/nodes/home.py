"""
HomeNode – Control robot to return to home pose (only move position, not rotation).

Uses MoveSkill to move from current position to env.home_pos.
"""


from __future__ import annotations
from typing import Any

import py_trees

from panda_mujoco_gym.skills.move import MoveSkill


class HomeNode(py_trees.behaviour.Behaviour):
    def __init__(self, env: Any, name: str = "Home") -> None:
        super().__init__(name=name)
        self.env = env
        self.skill = None

    def initialise(self) -> None:
        # Get home point coordinates, or use current position if not set
        home_pos = getattr(self.env, "home_pos", self.env.get_ee_position())
        self.skill = MoveSkill(self.env, target_pos=home_pos, steps=30)
        self.skill.reset()

    def update(self) -> py_trees.common.Status:
        self.skill.step()
        if self.skill.done:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID and self.skill is not None:
            self.skill.reset()

    @property
    def done(self) -> bool:
        return self.status == py_trees.common.Status.SUCCESS
