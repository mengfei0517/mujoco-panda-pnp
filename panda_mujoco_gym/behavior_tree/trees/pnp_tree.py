"""panda_mujoco_gym.behavior_tree.trees.pnp_tree

Pick‑and‑Place 行为树构建器。
组合 PickNode / PlaceNode / HomeNode，支持对每次抓取自动重试。
"""
from __future__ import annotations

from typing import Any, Dict, List

import py_trees

from panda_mujoco_gym.behavior_tree.nodes.pick import PickNode
from panda_mujoco_gym.behavior_tree.nodes.place import PlaceNode
from panda_mujoco_gym.behavior_tree.nodes.home import HomeNode

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_pnp_tree(
    env: Any,
    tasks: List[Dict[str, Any]],
    retry_pick: int = 3,
) -> py_trees.trees.BehaviourTree:
    """创建一棵按顺序抓取并放置多物体的行为树。

    Args:
        env: 仿真或真实机器人环境实例。
        tasks: 每个元素包含：
            {
                "obj_meta": {...},      # 传给 PickNode
                "target_pos": [x,y,z],
                "rotate_back_quat": [x,y,z,w],
            }
        retry_pick: 每件物体抓取失败后自动重试次数（`Retry` Decorator）。
    """

    root = py_trees.composites.Sequence(name="PnP-Root", memory=True)

    for i, task in enumerate(tasks):
        # ---- Pick ----
        pick = PickNode(env, obj_meta=task["obj_meta"], name=f"Pick-{i}")
        if retry_pick > 1:
            pick = py_trees.decorators.Retry(
                name=f"RetryPick-{i}", child=pick, num_failures=retry_pick
            )

        # ---- Place ----
        place = PlaceNode(
            env,
            target_pos=task["target_pos"],
            rotate_back_quat=task.get("rotate_back_quat", [0, 0, 0, 1]),
            name=f"Place-{i}",
        )

        root.add_children([pick, place])

    # 所有物体完成后回到 Home
    root.add_child(HomeNode(env))

    return py_trees.trees.BehaviourTree(root)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    class _DummyEnv:
        """非常简陋的示例环境，仅用于演示树可以 tick 完成。"""

        def get_ee_orientation(self):
            return [0, 0, 0, 1]

        def is_holding_object(self, _):
            return True

    tasks_demo = [
        {
            "obj_meta": {
                "id": 1,
                "delta_q": [0, 0, 0, 1],
                "approach_wpts": [],
            },
            "target_pos": [0.5, 0.0, 0.3],
            "rotate_back_quat": [0, 0, 0, 1],
        }
    ]

    tree = build_pnp_tree(_DummyEnv(), tasks_demo, retry_pick=1)
    for _ in range(10):
        tree.tick()
    print("Tree status:", tree.root.status)
