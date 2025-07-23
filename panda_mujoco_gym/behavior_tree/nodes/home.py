# panda_mujoco_gym/behavior_tree/nodes/home.py

from __future__ import annotations
from typing import Any

import py_trees

from panda_mujoco_gym.skills.home import HomeSkill


class HomeNode(py_trees.behaviour.Behaviour):
    """HomeNode – 控制机器人回到 home 姿态。

    使用 HomeSkill，从当前位置移动到 home_wpt，自动判断是否完成。

    Args:
        env: 环境对象，必须提供 set_mocap_pose 等接口。
        name: 节点名（可选）
    """

    def __init__(self, env: Any, name: str = "Home") -> None:
        super().__init__(name=name)
        self.env = env
        self.skill = HomeSkill(env)

    def initialise(self) -> None:
        self.skill.reset()

    def update(self) -> py_trees.common.Status:
        self.skill.step()

        if self.skill.done:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self.skill.reset()

    @property
    def done(self) -> bool:
        return self.status == py_trees.common.Status.SUCCESS
