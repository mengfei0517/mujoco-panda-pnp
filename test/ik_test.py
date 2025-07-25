import numpy as np
import gymnasium as gym
from panda_mujoco_gym.skills.ik_solver import JacobianIKController

def test_solve_ik_direct():
    env = gym.make("FrankaShelfPNPDense-v0", render_mode="human")
    env.reset()

    model = env.unwrapped.model
    data = env.unwrapped.data
    site_name = "ee_center_site"

    # 初始位姿
    start_pos = env.unwrapped.get_ee_position().copy()
    start_quat = env.unwrapped.get_ee_orientation().copy()
    print("Initial EE position:", start_pos)
    print("Initial EE quat:", start_quat)

    # 设置目标位置（+X方向偏移）
    offset = np.array([0.1, 0.0, 0.0])
    target_pos = start_pos + offset
    q_init = data.qpos[:7].copy()

    # IK 控制器调用（注意：此版本不处理 orientation）
    ik = JacobianIKController(model, data, site_name)
    q_sol = ik.solve(
        target_pos=target_pos,
        q_init=q_init,
        max_iters=100,
        pos_thresh=1e-4,
        damping=0.05,
    )

    # 执行并 forward，获取末端位置
    env.unwrapped.set_joint_angles(q_sol)
    env.forward()

    final_pos = env.unwrapped.get_ee_position()
    print("Target pos:", target_pos)
    print("Final  pos:", final_pos)
    print("Position error:", np.linalg.norm(final_pos - target_pos))

    assert np.linalg.norm(final_pos - target_pos) < 0.05, "Position error too large"
