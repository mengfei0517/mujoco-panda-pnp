from __future__ import annotations

import argparse
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import py_trees
from scipy.spatial.transform import Rotation as R

from panda_mujoco_gym.behavior_tree.nodes.pick import PickNode

def build_task(env) -> Dict[str, Any]:
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

def main():  # noqa: D401 – CLI entry
    p = argparse.ArgumentParser("Debug single PickNode")
    p.add_argument("--env", default="FrankaShelfPNPDense-v0")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--max-tick", type=int, default=1200)
    p.add_argument("--sim-steps", type=int, default=5,
                   help="MuJoCo steps per BT tick")
    p.add_argument("--fps", type=int, default=30,
                   help="Target viewer frame‑rate when --render")
    args = p.parse_args()

    env = gym.make(args.env, render_mode="human" if args.render else None)
    env.reset()

    # Gripper open
    open_act = np.zeros(env.action_space.shape, dtype=np.float32)
    open_act[-1] = 1.0  # assuming last dim is gripper open
    for _ in range(10):
        env.step(open_act)
    if args.render:
        env.render()

    # Build BT
    task = build_task(env)
    pick = PickNode(env, task["meta"], name="PickDebug")
    tree = py_trees.trees.BehaviourTree(pick)

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

        if pick.status == py_trees.common.Status.SUCCESS:
            print(f"[✓] Pick SUCCESS after {t+1} ticks")
            break

        if args.render:
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    else:
        print("[✗] Pick did not succeed within limit")

    env.close()
    time.sleep(0.3)

if __name__ == "__main__":
    main()
