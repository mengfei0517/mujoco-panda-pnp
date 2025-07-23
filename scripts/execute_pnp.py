from __future__ import annotations

import argparse
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import py_trees
from scipy.spatial.transform import Rotation as R

from panda_mujoco_gym.behavior_tree.nodes.pick import PickNode
from panda_mujoco_gym.behavior_tree.nodes.place import PlaceNode
from panda_mujoco_gym.behavior_tree.nodes.home import HomeNode  # ← 新增

def build_pick_place_task(env) -> Dict[str, Any]:
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
            "approach_wpt1": obj_pos + np.array([-0.20, 0.0, 0.05]),  # 可选与 pick 共用
            "home_wpt": np.array([1.23843967, 0.0, 0.49740014]),
            "rotate_back_quat": R.from_euler("y", 90, degrees=True).as_quat().tolist(),
            "approach_wpt2": target_pos + np.array([0.0, 0.0, 0.06]),
        }
    }


def main():
    parser = argparse.ArgumentParser("Debug Pick and Place and Home")
    parser.add_argument("--env", default="FrankaShelfPNPDense-v0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--max-tick", type=int, default=3000)
    parser.add_argument("--sim-steps", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human" if args.render else None)
    env.reset()

    open_act = np.zeros(env.action_space.shape, dtype=np.float32)
    open_act[-1] = 1.0
    for _ in range(10):
        env.step(open_act)

    if args.render:
        env.render()

    task = build_pick_place_task(env)
    pick_node = PickNode(env, task["pick_meta"], name="Pick")
    place_node = PlaceNode(env, task["place_meta"], name="Place")
    home_node = HomeNode(env, name="ReturnHome")  # ← 新增

    root = py_trees.composites.Sequence(name="Pick-Place-Home", memory=True)
    root.add_children([pick_node, place_node, home_node])
    tree = py_trees.trees.BehaviourTree(root)

    dt = 1.0 / args.fps

    for t in range(args.max_tick):
        start = time.time()

        tree.tick()

        for _ in range(args.sim_steps):
            env.unwrapped._mujoco.mj_step(
                env.unwrapped.model,
                env.unwrapped.data,
                nstep=1,
            )

        if args.render:
            env.render()

        if root.status == py_trees.common.Status.SUCCESS:
            print(f"[✓] Pick + Place + Home SUCCESS after {t+1} ticks")
            break

        if args.render:
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    else:
        print("[✗] Pick + Place + Home did not succeed within limit")

    env.close()
    time.sleep(0.3)


if __name__ == "__main__":
    main()
