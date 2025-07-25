from __future__ import annotations
from typing import List, Dict, Any

import py_trees
from panda_mujoco_gym.skills.move import MoveSkill
from panda_mujoco_gym.skills.rotate import RotateSkill
from panda_mujoco_gym.skills.gripper import GripperSkill


class PickNode(py_trees.behaviour.Behaviour):
    """
    Pick sequence:
        Rotate →
        Move to approach_wpt1 →
        Move to obj_pos →
        Grasp →
        Move to approach_wpt2
    """

    def __init__(
        self,
        env: Any,
        meta: Dict[str, Any],
        name: str | None = None,
    ) -> None:
        super().__init__(name or f"Pick-{meta.get('id', 'obj')}")
        self.env = env
        self.meta = meta
        self.skills: List[py_trees.behaviour.Behaviour] = [
            RotateSkill(env, meta["delta_q"]),
        ]
        self.phase = 0
        self.curr: py_trees.behaviour.Behaviour | None = None

    def initialise(self) -> None:
        self.phase = 0
        for sk in self.skills:
            sk.reset()
        self.curr = self.skills[0]

    def update(self) -> py_trees.common.Status:
        assert self.curr is not None
        self.curr.step()

        if getattr(self.curr, "done", False):
            self.phase += 1

            if self.phase == 1:
                self.skills.append(MoveSkill(self.env, self.meta["approach_wpt1"]))
            elif self.phase == 2:
                self.skills.append(MoveSkill(self.env, self.meta["obj_pos"]))
            elif self.phase == 3:
                self.skills.append(GripperSkill.close(self.env))
            elif self.phase == 4:
                self.skills.append(MoveSkill(self.env, self.meta["approach_wpt2"]))

            if self.phase >= len(self.skills):
                # if self.env.is_holding_object(self.meta.get("id", None)):
                #     return py_trees.common.Status.SUCCESS
                # else:
                #     return py_trees.common.Status.FAILURE
                return py_trees.common.Status.SUCCESS

            self.curr = self.skills[self.phase]
            self.curr.reset()

        return py_trees.common.Status.RUNNING

    @property
    def done(self) -> bool:
        return self.status == py_trees.common.Status.SUCCESS
