"""
Franka / Panda pick-and-place environment (base class)
"""

from __future__ import annotations

import mujoco                                   # type: ignore
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat, Sequence


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}


class FrankaEnv(MujocoRobotEnv):
    """Multi-tier shelf Pick-and-Place environment (multi-object version)"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    # ───────────────── Pose Constants ──────────────────
    VERTICAL_QUAT = rotations.euler2quat(np.zeros(3))                      # z-down
    HORIZONTAL_QUAT = rotations.euler2quat(np.array([-np.pi / 2, 0, 0]))   # x-down

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_substeps: int = 50,
        reward_type: str = "dense",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        obj_x_range: float = 0.05,
        obj_y_range: float = 0.2,
        task_sequence: Optional[Sequence[str]] = None,
        # ── Reward-related parameters ──────────────────────────────
        orientation_weight: float = 0.2,
        orientation_threshold: float = 0.15,
        high_pick_z: float = 0.35,
        **kwargs,
    ):
        # ── Multi-object task list ──────────────────────
        self.task_sequence = list(task_sequence) if task_sequence is not None else [
            "cube1",
            "cube2",
            "cube3",
        ]
        self.current_task_index = 0
        self.current_target_object = self.task_sequence[0]
        self.goal: Optional[np.ndarray] = None

        # ── Other members ───────────────────────────────
        self.block_gripper = block_gripper
        self.model_path = model_path
        self.reward_type = reward_type

        action_size = 6 + (0 if self.block_gripper else 1)   # Δxyz + Δrpy + gripper
        self.neutral_joint_values = np.array(
            [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]
        )

        # ── Reward-related parameters ──────────────────────────────
        self.orientation_weight = orientation_weight
        self.orientation_threshold = orientation_threshold
        self.high_pick_z = high_pick_z

        # ── Sampling range parameters ──────────────────────────────
        self.distance_threshold = distance_threshold
        self.obj_x_range = obj_x_range
        self.obj_y_range = obj_y_range

        # ── Initialize parent class (load MJCF) ──────────
        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # ── MuJoCo meta info ────────────────────────────
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ctrl_range = self.model.actuator_ctrlrange

        # ── Initialize multi-object task ────────────────
        self._initialize_multi_object_task()

    # ══════════════════════════════════════════════════════
    # Initialize / Reset
    # ══════════════════════════════════════════════════════
    def _initialize_multi_object_task(self):
        """Initialize multi-object sequential task"""
        self.current_task_index = 0
        self.current_target_object = self.task_sequence[0]
        self.goal = self._sample_goal()

    def _initialize_simulation(self) -> None:
        """Initialize MuJoCo simulation and model"""
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)          # type: ignore
        self.data = self._mujoco.MjData(self.model)                             # type: ignore
        self._model_names = self._utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # ── Joint grouping ──────────────────────────────
        free_joint_index = self._model_names.joint_names.index("obj_joint")
        self.arm_joint_names = self._model_names.joint_names[:free_joint_index][0:7]
        self.gripper_joint_names = self._model_names.joint_names[:free_joint_index][7:9]

        # ── Environment setup ─────────────────────────────
        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        """Setup environment with neutral joint configuration"""
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[:7]
        self.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)                          # type: ignore

        # ── Record initial end-effector pose ──────────────
        self.initial_mocap_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._mujoco_step()
        self.initial_object_height = self._utils.get_joint_qpos(
            self.model, self.data, "obj_joint"
        )[2].copy()

    # ══════════════════════════════════════════════════════
    # Sampling (keep original logic, z is determined by XML)
    # ══════════════════════════════════════════════════════
    def _sample_object(self):
        """Sample object positions with random offsets"""
        for obj in self.task_sequence:
            site_name = f"{obj}_site"
            center_pos = self._utils.get_site_xpos(self.model, self.data, site_name).copy()
            x = center_pos[0] + np.random.uniform(-self.obj_x_range, self.obj_x_range)
            y = center_pos[1] + np.random.uniform(-self.obj_y_range, self.obj_y_range)
            z = center_pos[2]                                   # Fixed z position
            joint_name = f"{obj}_joint"
            self._utils.set_joint_qpos(
                self.model, self.data, joint_name, np.array([x, y, z, 1, 0, 0, 0])
            )
        self._mujoco.mj_forward(self.model, self.data)

    # ══════════════════════════════════════════════════════
    # Main interface
    # ══════════════════════════════════════════════════════
    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:  # type: ignore
        """Execute one step in the environment"""
        if np.asarray(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)           # type: ignore
        self._set_action(action)

        self._mujoco_step(action)
        self._step_callback()
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        if self.goal is not None:
            assert np.allclose(obs["desired_goal"], self.goal), "goal mismatch"

        info = {"is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])}
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

        # ── Multi-object sequential task processing ────────
        terminated = False
        if info["is_success"]:
            self.current_task_index += 1
            if self.current_task_index < len(self.task_sequence):
                self.current_target_object = self.task_sequence[self.current_task_index]
                self.goal = self._utils.get_site_xpos(
                    self.model, self.data, f"target_{self.current_target_object}"
                ).copy()
            else:
                terminated = True

        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        return obs, reward, terminated, truncated, info                                   # type: ignore

    # ══════════════════════════════════════════════════════
    # Reward function
    # ══════════════════════════════════════════════════════
    # ------------------------------------------------------------------
    # Event-driven dense reward - small negative when no action, 
    # positive reward when correct grasp-lift-place
    # ------------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward based on achieved and desired goals"""
        achieved_goal = np.asarray(achieved_goal)
        desired_goal  = np.asarray(desired_goal)

        # ── Basic quantities ──────────────────────────────
        d_reach = float(self.goal_distance(self.get_ee_position(), achieved_goal))
        d_place = float(self.goal_distance(achieved_goal, desired_goal))

        ee_width = float(self.get_fingers_width())
        GRIP_WIDTH_THRESH = 0.045     # 4 cm cube + ≈5 mm clearance
        gripped = (ee_width < GRIP_WIDTH_THRESH) and (d_reach < 0.05)

        # ── Lift detection ──────────────────────────────────────────
        lifted   = gripped and (achieved_goal[2] - self.initial_object_height > 0.04)
        placed   = d_place < self.distance_threshold

        ee_q   = self.get_ee_orientation()
        need_q = self.HORIZONTAL_QUAT if achieved_goal[2] > self.high_pick_z else self.VERTICAL_QUAT
        ori_err = float(1.0 - abs(np.dot(ee_q, need_q)))        # [0,1]

        # ── Sparse reward ────────────────────────────────────────
        if self.reward_type == "sparse":
            return np.float32(-float(not placed))

        # ── Dense reward ──────────────────────────────────────────
        reward  = -0.003                          # Time penalty
        reward += -min(d_reach, 0.05)             # Reach negative gradient (max 0.05)

        if gripped:
            reward += 2.0                         # Grip reward
            reward += (1.0 - ori_err)             # Alignment reward (0~1)

        if lifted:
            reward += 4.0                         # Lift reward

        if placed:
            reward += 10.0                        # Place reward

        reward += 0.5 * (self.current_task_index / len(self.task_sequence))
        return np.float32(reward)

    # ══════════════════════════════════════════════════════
    # Low-level control / Observation
    # ══════════════════════════════════════════════════════
    def _set_action(self, action) -> None:
        """Set action for end-effector and gripper"""
        action = action.copy()
        if not self.block_gripper:
            pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:6], action[6]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(
                fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1]
            )
        else:
            pos_ctrl, rot_ctrl = action[:3], action[3:6]
            fingers_half_width = 0.0

        # ── Gripper control ──────────────────────────────
        self.data.ctrl[-2:] = fingers_half_width

        # ── End-effector position (Δposition scaling) ──────────────
        pos_ctrl = self.get_ee_position().copy() + 0.05 * pos_ctrl
        pos_ctrl[2] = max(0.0, pos_ctrl[2])        # Prevent ground penetration

        # ── End-effector orientation (Euler→Quat accumulation) ──────────────
        current_quat = self.get_ee_orientation().copy()
        delta_euler = np.clip(rot_ctrl, -1.0, 1.0) * 0.1
        delta_quat = rotations.euler2quat(delta_euler)
        target_quat = rotations.quat_mul(delta_quat, current_quat)

        self.set_mocap_pose(pos_ctrl, target_quat)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get current observation"""
        current_obj = self.current_target_object
        goal = self.goal.copy() if self.goal is not None else None

        # ── End-effector state ──────────────────────────────────
        ee_pos = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_vel = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        # ── Object state ───────────────────────────────────────
        site = f"{current_obj}_site"
        obj_pos = self._utils.get_site_xpos(self.model, self.data, site).copy()
        obj_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, site)).copy()
        obj_velp = self._utils.get_site_xvelp(self.model, self.data, site).copy() * self.dt
        obj_velr = self._utils.get_site_xvelr(self.model, self.data, site).copy() * self.dt

        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()
            obs = np.concatenate([ee_pos, ee_vel, fingers_width, obj_pos, obj_rot, obj_velp, obj_velr])
        else:
            obs = np.concatenate([ee_pos, ee_vel, obj_pos, obj_rot, obj_velp, obj_velr])

        return {"observation": obs, "achieved_goal": obj_pos.copy(), "desired_goal": goal}

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        """Check if task is successful"""
        d = float(self.goal_distance(achieved_goal, desired_goal))
        return np.float32(1.0 if d < self.distance_threshold else 0.0)

    # ══════════════════════════════════════════════════════
    # Helper methods
    # ══════════════════════════════════════════════════════
    def goal_distance(self, a, b):
        """Calculate distance between goals (supports batch)"""
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b, axis=-1)

    def set_mocap_pose(self, pos, quat) -> None:
        """Set mocap pose for end-effector"""
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", pos)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", quat)

    def set_joint_neutral(self) -> None:
        """Set joints to neutral configuration"""
        for name, val in zip(self.arm_joint_names, self.neutral_joint_values[:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, val)
        for name, val in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, val)

    def reset_mocap_welds(self, model, data) -> None:
        """Reset mocap welds to default configuration"""
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:                 # type: ignore
                    model.eq_data[i, 3:10] = np.array([0, 0, 0, 1, 0, 0, 0])
        self._mujoco.mj_forward(model, data)                                   # type: ignore

    def get_ee_orientation(self):
        """Get end-effector orientation as quaternion"""
        mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        quat = np.empty(4)
        self._mujoco.mju_mat2Quat(quat, mat)                                   # type: ignore
        return quat

    def get_ee_position(self):
        """Get end-effector position"""
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_fingers_width(self):
        """Get gripper fingers width"""
        f1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        f2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return f1 + f2

    # ── Public interface ─────────────────────────────────
    def _mujoco_step(self, *_):
        """Execute MuJoCo simulation step"""
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps) # type: ignore

    def _sample_goal(self):
        """Sample goal position for current target object"""
        return self._utils.get_site_xpos(
            self.model, self.data, f"target_{self.current_target_object}"
        ).copy()

    def _reset_sim(self) -> bool:
        """Reset simulation state"""
        # 1) Reset state
        self.data.time = self.initial_time
        self.data.qvel[:] = self.initial_qvel
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        # 2) Resample object positions
        self._sample_object()

        # 3) Multi-object synchronization
        self._initialize_multi_object_task()

        self._mujoco.mj_forward(self.model, self.data)
        return True

    # ── Convenient interface ────────────────────────────────────
    def reset(self, *args, **kwargs):
        """Reset environment and return initial observation"""
        out = super().reset(*args, **kwargs)
        self.home_pos = self.get_ee_position().copy()
        return out

    def set_joint_angles(self, q: np.ndarray):
        """Directly write joint angles (for simulation debugging only)"""
        assert q.shape == (7,)
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, target_pos, target_quat, q_init=None):
        """Thin wrapper for global IK solver"""
        from panda_mujoco_gym.skills.ik_solver import solve_ik as _ik
        return _ik(
            model=self.model,
            data=self.data,
            site_name="ee_center_site",
            target_pos=target_pos,
            target_quat=target_quat,
            q_init=q_init if q_init is not None else self.data.qpos[:7].copy(),
        )

    @property
    def utils(self):
        """Get MuJoCo utilities"""
        return self._utils
