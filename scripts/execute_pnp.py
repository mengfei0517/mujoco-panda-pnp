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
from panda_mujoco_gym.behavior_tree.trees.pnp_tree import build_pnp_tree  # 新增

def build_pick_place_tasks(env):
    tasks = []
    for name in env.unwrapped.task_sequence:
        utils = env.unwrapped._utils  # type: ignore[attr-defined]
        obj_pos = utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, f"{name}_site").copy()
        target_pos = utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, f"target_{name}").copy()
        pick_meta = {
            "id": hash(name) % 10000,
            "delta_q": R.from_euler("y", -90, degrees=True).as_quat().tolist(),
            "approach_wpt1": obj_pos + np.array([-0.20, 0.0, 0.05]),
            "obj_pos": obj_pos,
            "approach_wpt2": obj_pos + np.array([0.0, 0.0, 0.06]),
        }
        place_meta = {
            "approach_wpt1": obj_pos + np.array([-0.20, 0.0, 0.05]),
            "home_wpt": np.array([1.23843967, 0.0, 0.49740014]),
            "rotate_back_quat": R.from_euler("y", 90, degrees=True).as_quat().tolist(),
            "approach_wpt2": target_pos + np.array([0.0, 0.0, 0.06]),
        }
        tasks.append({"pick_meta": pick_meta, "place_meta": place_meta})
    return tasks


def main():
    parser = argparse.ArgumentParser("Debug Pick and Place and Home")
    parser.add_argument("--env", default="FrankaShelfPNPDense-v0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--max-tick", type=int, default=3000)
    parser.add_argument("--sim-steps", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--task-sequence",
        type=str,
        default=None,
        help="用逗号分隔的物体名序列，如 sphere,cylinder,cube"
    )
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human" if args.render else None)
    env.reset()

    # 这里赋值任务序列
    env.unwrapped.task_sequence = ["cube1", "cube2", "cube3"]  # 你想要的顺序

    # 或者用命令行参数
    if args.task_sequence is not None:
        env.unwrapped.task_sequence = [name.strip() for name in args.task_sequence.split(",")]

    # 后续直接用 env.unwrapped.task_sequence
    # 打开渲染窗口，持续刷新10秒，等待用户调整视角
    if args.render:
        print("请调整好视频视角，10秒后自动开始任务...")
        t0 = time.time()
        while time.time() - t0 < 10:
            env.render()
            time.sleep(0.03)  # 约30fps，窗口流畅

    open_act = np.zeros(env.action_space.shape, dtype=np.float32)
    open_act[-1] = 1.0
    for _ in range(10):
        env.step(open_act)

    if args.render:
        env.render()

    tasks = build_pick_place_tasks(env)
    # 适配pnp_tree的接口
    tasks_for_tree = [
        {"obj_meta": t["pick_meta"], "place_meta": t["place_meta"]} for t in tasks
    ]
    tree = build_pnp_tree(env, tasks_for_tree, retry_pick=1)
    root = tree.root

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
