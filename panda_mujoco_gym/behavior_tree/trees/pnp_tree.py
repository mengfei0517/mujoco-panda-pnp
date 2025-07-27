"""panda_mujoco_gym.behavior_tree.trees.pnp_tree

Pick‑and‑Place behavior tree builder.
Combines PickNode / PlaceNode / HomeNode, supports automatic retry for each pick.
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
    root = py_trees.composites.Sequence(name="PnP-Root", memory=True)

    for i, task in enumerate(tasks):
        pick = PickNode(env, meta=task["obj_meta"], name=f"Pick-{i}")
        if retry_pick > 1:
            pick = py_trees.decorators.Retry(
                name=f"RetryPick-{i}", child=pick, num_failures=retry_pick
            )
        place = PlaceNode(env, meta=task["place_meta"], name=f"Place-{i}")
        home = HomeNode(env, name=f"Home-{i}")

        # Process for each object: Pick → Place → Home
        sub_seq = py_trees.composites.Sequence(name=f"PnP-Task-{i}", memory=True)
        sub_seq.add_children([pick, place, home])

        root.add_child(sub_seq)

    return py_trees.trees.BehaviourTree(root)

