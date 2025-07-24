"""test/bt_test.py

Pytest 用例：验证 Pick‑n‑Place 行为树可在有限 tick 内执行完成；
同时确保 HomeSkill 至少走过 >1 个插值步，而非一步即成功。
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, List

import numpy as np
import py_trees
import pytest

import gymnasium as gym
import py_trees
import numpy as np
from panda_mujoco_gym.behavior_tree.nodes.pick import PickNode
from panda_mujoco_gym.behavior_tree.nodes.place import PlaceNode
from scipy.spatial.transform import Rotation as R
import time


# -----------------------------------------------------------------------------
# Dummy 环境 —— 满足各 Skill / Node 需要的最小接口
# -----------------------------------------------------------------------------
class _DummyEnv:
    """模拟机器人环境，记录末端位姿与抓取状态。"""

    def __init__(self):
        # 起始位姿并非 home 位姿，确保 HomeSkill 需要多步插值
        self._pos: np.ndarray = np.array([0.3, 0.0, 0.4])
        self._quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

        # 设定 home 位姿，与 start_pos 不同
        self.home_pos: np.ndarray = np.array([0.5, 0.0, 0.3])
        self.home_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

        self._holding = False

    # ---- 机器人接口 ----
    def set_mocap_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self._pos = np.asarray(pos)
        self._quat = np.asarray(quat)

    def get_ee_position(self) -> np.ndarray:  # noqa: D401
        return self._pos

    def get_ee_orientation(self) -> np.ndarray:  # noqa: D401
        return self._quat

    # PickNode 会查询是否抓住物体
    def is_holding_object(self, _obj_id: int) -> bool:  # noqa: D401, unused arg
        return True

    # 桩函数：供技能返回 action 长度一致
    def step(self, *_: Any) -> None:  # noqa: D401, unused arg
        pass


# -----------------------------------------------------------------------------
#  Dummy Skill —— 立即完成；满足 Node 的构造签名
# -----------------------------------------------------------------------------
class _DummySkill:
    """RotateSkill / MoveSkill / GripperSkill 的替身，`step()` 一次完成。"""

    def __init__(self, *_, **__):
        self.done = False

    # 单步完成
    def reset(self):
        self.done = False

    def step(self):
        self.done = True  # 立即完成
        return True

# -----------------------------------------------------------------------------
# Monkey‑patch 原子技能为 DummySkill，使测试聚焦 BT 逻辑
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_skills(monkeypatch):
    """自动将 Rotate/Move/Gripper Skill 替换成 _DummySkill。"""

    import importlib

    skill_modules = [
        "panda_mujoco_gym.skills.rotate",
        "panda_mujoco_gym.skills.move",
        "panda_mujoco_gym.skills.gripper",
    ]
    for mod_name in skill_modules:
        mod: ModuleType = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "RotateSkill", _DummySkill, raising=False)
        monkeypatch.setattr(mod, "MoveSkill", _DummySkill, raising=False)
        monkeypatch.setattr(mod, "GripperSkill", _DummySkill, raising=False)
    yield  # 测试结束后自动恢复


# -----------------------------------------------------------------------------
# 主测试：行为树应在有限 tick 内运行到 SUCCESS
# -----------------------------------------------------------------------------

def _build_dummy_tasks() -> List[dict]:
    """生成两个虚拟抓取任务。"""
    return [
        {
            "obj_meta": {"id": 1, "delta_q": [0, 0, 0, 1], "approach_wpts": []},
            "target_pos": [0.6, 0.1, 0.3],
            "rotate_back_quat": [0, 0, 0, 1],
        },
        {
            "obj_meta": {"id": 2, "delta_q": [0, 0, 0, 1], "approach_wpts": []},
            "target_pos": [0.55, -0.1, 0.3],
            "rotate_back_quat": [0, 0, 0, 1],
        },
    ]


def test_pnp_tree_runs_to_success():
    """行为树应在 ≤100 tick 内返回 SUCCESS。"""

    from panda_mujoco_gym.behavior_tree.trees.pnp_tree import build_pnp_tree

    env = _DummyEnv()
    tree = build_pnp_tree(env, tasks=_build_dummy_tasks(), retry_pick=2)

    max_tick = 100
    for _ in range(max_tick):
        tree.tick()
        env.step()
        if tree.root.status == py_trees.common.Status.SUCCESS:
            break

    assert tree.root.status == py_trees.common.Status.SUCCESS

    # 断言 HomeSkill 至少运行了 2 步（插值 >1）
    home_node = tree.root.children[-1]
    assert hasattr(home_node, "_home_skill")
    assert home_node._home_skill.i > 1


def build_pick_task(env):
    name = getattr(env.unwrapped, "task_sequence", ["sphere"])[0]
    utils = env.unwrapped._utils  # type: ignore[attr-defined]
    obj_pos = utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, f"{name}_site").copy()
    return {
        "meta": {
            "id": hash(name) % 10000,
            "delta_q": R.from_euler("y", -90, degrees=True).as_quat().tolist(),
            "approach_wpt1": obj_pos + np.array([-0.20, 0.0, 0.05]),
            "obj_pos": obj_pos,
            "approach_wpt2": obj_pos + np.array([0.0, 0.0, 0.06]),
        }
    }

def build_pick_place_task(env):
    name = getattr(env.unwrapped, "task_sequence", ["sphere"])[0]
    utils = env.unwrapped._utils  # type: ignore[attr-defined]
    obj_pos = utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, f"{name}_site").copy()
    target_pos = np.array([1.0, 0.0, 0.3])
    return {
        "pick_meta": {
            "id": hash(name) % 10000,
            "delta_q": R.from_euler("y", -90, degrees=True).as_quat().tolist(),
            "approach_wpt1": obj_pos + np.array([-0.20, 0.0, 0.05]),
            "obj_pos": obj_pos,
            "approach_wpt2": obj_pos + np.array([0.0, 0.0, 0.06]),
        },
        "place_meta": {
            "approach_wpt1": obj_pos + np.array([-0.20, 0.0, 0.05]),
            "home_wpt": np.array([1.23843967, 0.0, 0.49740014]),
            "rotate_back_quat": R.from_euler("y", 90, degrees=True).as_quat().tolist(),
            "approach_wpt2": target_pos + np.array([0.0, 0.0, 0.06]),
        }
    }


def test_pick_node_runs_to_success():
    """PickNode 节点应在有限 tick 内返回 SUCCESS。"""
    env = gym.make("FrankaShelfPNPDense-v0")
    env.reset()
    # 夹爪初始张开
    open_act = np.zeros(env.action_space.shape, dtype=np.float32)
    open_act[-1] = 1.0
    for _ in range(10):
        env.step(open_act)
    task = build_pick_task(env)
    pick = PickNode(env, task["meta"], name="PickDebug")
    tree = py_trees.trees.BehaviourTree(pick)
    for t in range(1200):
        tree.tick()
        for _ in range(5):
            env.unwrapped._mujoco.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=1)
        if pick.status == py_trees.common.Status.SUCCESS:
            break
    assert pick.status == py_trees.common.Status.SUCCESS
    env.close()
    time.sleep(0.3)


def test_pick_and_place_node_runs_to_success():
    """Pick+Place 节点组合应在有限 tick 内返回 SUCCESS。"""
    env = gym.make("FrankaShelfPNPDense-v0")
    env.reset()
    open_act = np.zeros(env.action_space.shape, dtype=np.float32)
    open_act[-1] = 1.0
    for _ in range(10):
        env.step(open_act)
    task = build_pick_place_task(env)
    pick_node = PickNode(env, task["pick_meta"], name="Pick")
    place_node = PlaceNode(env, task["place_meta"], name="Place")
    root = py_trees.composites.Sequence(name="Pick-Then-Place", memory=True)
    root.add_children([pick_node, place_node])
    tree = py_trees.trees.BehaviourTree(root)
    for t in range(2000):
        tree.tick()
        for _ in range(5):
            env.unwrapped._mujoco.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=1)
        if root.status == py_trees.common.Status.SUCCESS:
            break
    assert root.status == py_trees.common.Status.SUCCESS
    env.close()
    time.sleep(0.3)
