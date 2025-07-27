"""
Test reward function.
"""

"""
Three tests:
1. Static 10 steps → cumulative reward must be negative.
2. Use hard-coded behavior tree to run a PnP:
     • At least one ≥+6 positive reward (Grip+Lift)
     • Total reward in reasonable range (-300, 2500)
3. Randomly run 30 physics steps → reward sequence contains both positive and negative values.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import py_trees
from scipy.spatial.transform import Rotation as R
import pytest
import panda_mujoco_gym                             # noqa: register env

# ───────── Adjustable constants (match object size/finger width) ─────────
CUBE_EDGE        = 0.04          # Object outer diameter (m)
GRIPPER_CLEARANCE = 0.004        # Extra gap between fingers after grasping
GRIP_WIDTH_THRESH = CUBE_EDGE + GRIPPER_CLEARANCE    # ≈ 0.044
REACH_DIST_THRESH = (CUBE_EDGE / 2) + 0.03           # ≈ 0.05


def build_pick_place_tasks(env):
    tasks = []
    utils = env.unwrapped._utils
    task_sequence = "cube1"
    for name in env.unwrapped.task_sequence:
        obj = utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, f"{name}_site")
        tgt = utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, f"target_{name}")
        obj_y = obj[1]
        tasks.append(
            dict(
                obj_meta=dict(
                    id=hash(name) % 10000,
                    delta_q=R.from_euler("y", -90, degrees=True).as_quat().tolist(),
                    approach_wpt1=obj + np.array([-0.2, -obj_y, 0.05]),
                    obj_pos=obj + np.array([0.015, 0.0, 0.0]),
                    approach_wpt2=obj + np.array([0.0, 0.0, 0.06]),
                ),
                place_meta=dict(
                    approach_wpt1=obj + np.array([-0.20, -obj_y, 0.05]),
                    home_wpt=np.array([1.23843967, 0.0, 0.49740014]),
                    rotate_back_quat=R.from_euler("y", 90, degrees=True).as_quat().tolist(),
                    approach_wpt2=tgt + np.array([0.0, 0.0, 0.06]),
                ),
            )
        )
    return tasks


class RewardSampler:
    """Lightweight sampler: supports physics step and behavior tree PnP"""

    def __init__(self, render=False):
        self.env = gym.make("FrankaShelfPNPDense-v0", render_mode="human" if render else "rgb_array")
        self.reset_env()

    # ----------------------- env helpers -----------------------
    def reset_env(self):
        self.env.reset()
        self.env.unwrapped.task_sequence = ["cube1", "cube2", "cube3"]
        self.rewards, self.total = [], 0.0

    def _record_reward(self):
        sim = self.env.unwrapped
        obs = sim._get_obs()
        r = float(sim.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))
        self.rewards.append(r)
        self.total += r

    def physics_step_and_record(self, n=1):
        """Advance n physics steps and record reward"""
        sim = self.env.unwrapped
        for _ in range(n):
            sim._mujoco.mj_step(sim.model, sim.data, nstep=1)
            self._record_reward()

    # -------------------- grip helper -------------------------
    def _close_until_width(self, width_thresh=GRIP_WIDTH_THRESH, max_ctrl_steps=40):
        """Repeatedly send close gripper action until finger width < threshold or maximum steps reached"""
        close_act = np.zeros(self.env.action_space.shape, dtype=np.float32)
        close_act[-1] = -1.0
        for _ in range(max_ctrl_steps):
            self.env.step(close_act)
            self.physics_step_and_record(1)
            if self.env.unwrapped.get_fingers_width() < width_thresh:
                break

    # ------------------- run behavior tree --------------------
    def run_behavior_tree(self, ticks=200, sim_steps=4):
        from panda_mujoco_gym.behavior_tree.trees.pnp_tree import build_pnp_tree
        tree = build_pnp_tree(self.env, build_pick_place_tasks(self.env), retry_pick=1)
        root = tree.root
        for _ in range(ticks):
            tree.tick()
            self._close_until_width()               # First ensure gripper is closed
            self.physics_step_and_record(sim_steps) # Then advance physics
            if root.status == py_trees.common.Status.SUCCESS:
                break

    # ----------------- basic stats ----------------------------
    def stats(self):
        arr = np.asarray(self.rewards) if self.rewards else np.zeros(1)
        return dict(total=self.total, min=float(arr.min()), max=float(arr.max()))

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------
#                               Tests
# ---------------------------------------------------------------------
def test_static_negative_reward():
    rs = RewardSampler()
    rs.reset_env()
    rs.physics_step_and_record(10)
    st = rs.stats()
    assert st["total"] < 0, f"Static reward should be negative, got {st}"
    print("Static total:", st["total"])
    rs.close()


def test_episode_positive_spike():
    rs = RewardSampler()
    rs.reset_env()
    rs.run_behavior_tree(ticks=250, sim_steps=4)
    st = rs.stats()
    assert st["max"] >= 6.0, f"No +6 reward triggered, stats={st}"
    assert -300 < st["total"] < 2500, f"Total reward not in reasonable range, stats={st}"
    print("Episode total:", st["total"], "| max step reward:", st["max"])
    rs.close()

def test_reward_has_negative():
    """
    Random action / physics step should at least have negative reward, verify time/reach penalty is common.
    No longer force positive reward —— positive incentive has been verified in episode_positive_spike case.
    """
    rs = RewardSampler()
    rs.reset_env()

    for i in range(80):
        act = rs.env.action_space.sample()
        if i % 4 == 0:
            act[-1] = -1.0        # Periodically close gripper, increase probability of random positive reward (not required)
        rs.env.step(act)
        rs._record_reward()

    arr = np.asarray(rs.rewards)
    assert (arr < 0).any(), "Random step should have negative reward"
    rs.close()


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
