# test/skill_test.py
import sys
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
from panda_mujoco_gym.skills.rotate import RotateSkill
from panda_mujoco_gym.skills.move import MoveSkill
from panda_mujoco_gym.skills.gripper import GripperSkill


def make_env():
    env = gym.make("FrankaShelfPNPDense-v0")
    env.reset()
    return env

def run_skill(skill, env, max_steps=100, use_env_step=False):
    skill.reset()
    for i in range(max_steps):
        action = skill.step()
        if use_env_step:
            env.step(action)
        else:
            env.unwrapped._mujoco.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=1)
        if skill.is_done():
            sys.stdout.write(f"[PASS] {skill.__class__.__name__} finished in {i+1} steps.\n")
            sys.stdout.flush()
            break
    else:
        raise AssertionError(f"{skill.__class__.__name__} did not finish in {max_steps} steps")
    env.close()


def test_rotate_skill():
    """test rotate skill: -90° about Y"""
    env = make_env()
    delta_quat = R.from_euler("y", -90, degrees=True).as_quat()
    skill = RotateSkill(env, delta_quat=delta_quat, steps=30)
    run_skill(skill, env)

def test_move_via():
    """Arbitrary poly-line path."""
    env = make_env()
    pos  = env.unwrapped.get_ee_position().copy()
    quat = env.unwrapped.get_ee_orientation().copy()
    wpts = [
        pos + np.array([0.0, -0.10, 0.0]),
        pos + np.array([0.15, -0.10, 0.0]),
    ]
    skill = MoveSkill(env, waypoints=wpts, quat=quat, steps=30)
    run_skill(skill, env, max_steps=150)
    
def test_move_retreat():
    """Retreat: +X then -Z."""
    env = make_env()
    skill = MoveSkill.retreat(env, retreat_x=-0.30, retreat_z=-0.20, steps=20)
    run_skill(skill, env)

def test_move_place_linear():
    """Place: clearance above target, then descend."""
    env = make_env()
    pos = env.unwrapped.get_ee_position().copy()
    target = pos + np.array([0.0, 0.0, -0.08])
    skill = MoveSkill.place_linear(env, target_pos=target, clearance=0.10, steps=30)
    run_skill(skill, env)

def test_move_lift():
    """Lift via MoveSkill.lift()."""
    env   = make_env()
    skill = MoveSkill.lift(env, dz=0.06, steps=30, pos_thresh=0.005)
    run_skill(skill, env)

def test_gripper_close():
    env   = make_env()
    skill = GripperSkill.close(env, duration=40, thresh=0.02)
    run_skill(skill, env, max_steps=100)


def test_gripper_open():
    env = make_env()
    # 先闭合一下，确保随后 open 有意义
    env.step(np.concatenate([np.zeros(6), [-1.0]]))
    skill = GripperSkill.open(env, duration=30, thresh=0.08)
    run_skill(skill, env, max_steps=100)