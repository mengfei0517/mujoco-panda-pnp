from __future__ import annotations
from typing import Dict, Any, List

import py_trees
from panda_mujoco_gym.skills.move import MoveSkill, MoveIKSkill
from panda_mujoco_gym.skills.rotate import RotateSkill
from panda_mujoco_gym.skills.gripper import GripperSkill


class PlaceNode(py_trees.behaviour.Behaviour):
    """
    Place sequence (fully lazy built):
        Phase 0 → Move to approach_wpt1
        Phase 1 → Move to home_wpt
        Phase 2 → Rotate to rotate_back_quat
        Phase 3 → Move to approach_wpt2
        Phase 4 → Open gripper
    """

    def __init__(
        self,
        env,
        meta: Dict[str, Any],
        name: str = "Place"
    ) -> None:
        super().__init__(name=name)
        self.env = env
        self.meta = meta

        self.skills: List[py_trees.behaviour.Behaviour] = []
        self.phase = 0
        self.curr: py_trees.behaviour.Behaviour | None = None

    def initialise(self) -> None:
        self.skills.clear()
        self.phase = 0
        self.curr = self._build_skill(self.phase)
        self.curr.reset()

    def update(self) -> py_trees.common.Status:
        assert self.curr is not None
        self.curr.step()

        if getattr(self.curr, "done", False):
            self.phase += 1

            if self.phase >= 5:
                return py_trees.common.Status.SUCCESS

            self.curr = self._build_skill(self.phase)
            self.curr.reset()
            self.skills.append(self.curr)

        return py_trees.common.Status.RUNNING

    def _build_skill(self, phase: int) -> py_trees.behaviour.Behaviour:
        if phase == 0:
            return MoveIKSkill(self.env, self.meta["approach_wpt1"])
        elif phase == 1:
            return MoveIKSkill(self.env, self.meta["home_wpt"])
        elif phase == 2:
            return RotateSkill(self.env, self.meta["rotate_back_quat"])
        elif phase == 3:
            return MoveIKSkill(self.env, self.meta["approach_wpt2"])
        elif phase == 4:
            return GripperSkill.open(self.env)
        else:
            raise ValueError(f"[PlaceNode] Invalid phase {phase}")

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            for sk in self.skills[self.phase:]:
                sk.reset()

    @property
    def done(self) -> bool:
        return self.status == py_trees.common.Status.SUCCESS
