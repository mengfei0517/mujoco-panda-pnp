# panda_mujoco_gym/behavior_tree/nodes/home.py

from __future__ import annotations
from typing import Any

import py_trees

from panda_mujoco_gym.skills.move import MoveSkill


class HomeNode(py_trees.behaviour.Behaviour):
    """HomeNode – 控制机器人回到 home 姿态（只移动位置，不旋转）。

    使用 MoveSkill，从当前位置移动到 env.home_pos。
    """

    def __init__(self, env: Any, name: str = "Home") -> None:
        super().__init__(name=name)
        self.env = env
        self.skill = None

    def initialise(self) -> None:
        # 获取home点坐标，若无则用当前位置
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
