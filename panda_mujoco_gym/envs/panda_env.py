import mujoco  # type: ignore
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}


class FrankaEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_substeps: int = 50,
        reward_type: str = "sparse",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        obj_x_range: float = 0.05,
        obj_y_range: float = 0.2,
        **kwargs,
    ):
        # multi-object version variables initialization
        self.task_sequence = ["sphere", "cylinder"]
        # self.task_sequence = ["sphere"]
        self.current_task_index = 0
        self.current_target_object = self.task_sequence[0]
        self.goal = None

        # initialize other variables
        self.block_gripper = block_gripper
        self.model_path = model_path

        action_size = 6 # 3 for position, 3 for orientation
        action_size += 0 if self.block_gripper else 1

        self.reward_type = reward_type

        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.distance_threshold = distance_threshold

        # sample areas for the object 
        self.obj_x_range = obj_x_range
        self.obj_y_range = obj_y_range

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        self.ctrl_range = self.model.actuator_ctrlrange

        # Multi-object task initialization
        self._initialize_multi_object_task()

        # override the methods in MujocoRobotEnv
        # -----------------------------

    def _initialize_multi_object_task(self):
        self.current_task_index = 0
        self.current_target_object = self.task_sequence[0]
        self.goal = self._sample_goal()


    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)  # type: ignore
        self.data = self._mujoco.MjData(self.model)  # type: ignore
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # index used to distinguish arm and gripper joints
        free_joint_index = self._model_names.joint_names.index("obj_joint")
        self.arm_joint_names = self._model_names.joint_names[:free_joint_index][0:7]
        self.gripper_joint_names = self._model_names.joint_names[:free_joint_index][7:9]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)  # type: ignore

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco_step()

        self.initial_object_height = self._utils.get_joint_qpos(self.model, self.data, "obj_joint")[2].copy()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:  # type: ignore
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)  # type: ignore
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        if self.goal is not None:
            assert np.allclose(obs["desired_goal"], self.goal), "Mismatch between obs and internal goal"

        info = {"is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])}

        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

        # multi-object version
        terminated = False
        if info["is_success"]:
            self.current_task_index += 1
            if self.current_task_index < len(self.task_sequence):
                self.current_target_object = self.task_sequence[self.current_task_index]
                self.goal = self._utils.get_site_xpos(self.model, self.data, f"target_{self.current_target_object}").copy()
            else:
                terminated = True  # all tasks completed

        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        return obs, reward, terminated, truncated, info  # type: ignore

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        compute the reward for the multi-object task
        """
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        d = self.goal_distance(achieved_goal, desired_goal)
        task_progress = self.current_task_index / len(self.task_sequence)

        if self.reward_type == "sparse":
            # 稀疏奖励：只有达到目标时才给奖励
            reward = -(d < self.distance_threshold).astype(np.float32)
        else:
            # 稠密奖励：结合距离奖励和任务进度奖励
            reward = -d + 0.5 * task_progress

        return reward

    def _set_action(self, action) -> None:
        action = action.copy()
        if not self.block_gripper:
            pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:6], action[6]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])
        else:
            pos_ctrl, rot_ctrl = action[:3], action[3:6]
            fingers_half_width = 0

        # control gripper 
        self.data.ctrl[-2:] = fingers_half_width

        # control EE position
        pos_ctrl *= 0.05
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        # control EE orientation: euler → quaternion
        current_quat = self.get_ee_orientation().copy()
        delta_euler = np.clip(rot_ctrl, -1.0, 1.0) * 0.1  # reduce rotation step
        delta_quat = rotations.euler2quat(delta_euler)
        target_quat = rotations.quat_mul(delta_quat, current_quat)

        self.set_mocap_pose(pos_ctrl, target_quat)


    def _get_obs(self) -> dict:
        current_obj = self.current_target_object
        goal = self.goal.copy() if self.goal is not None else None

        # End-effector state
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        # Object state (only for current target object)
        current_site = f"{current_obj}_site"
        object_position = self._utils.get_site_xpos(self.model, self.data, current_site).copy()
        object_rotation = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, current_site)).copy()
        object_velp = self._utils.get_site_xvelp(self.model, self.data, current_site).copy() * self.dt
        object_velr = self._utils.get_site_xvelr(self.model, self.data, current_site).copy() * self.dt

        # Combine into observation
        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()
            observation = np.concatenate([
                ee_position,
                ee_velocity,
                fingers_width,
                object_position,
                object_rotation,
                object_velp,
                object_velr,
            ])
        else:
            observation = np.concatenate([
                ee_position,
                ee_velocity,
                object_position,
                object_rotation,
                object_velp,
                object_velr,
            ])

        return {
            "observation": observation,
            "achieved_goal": object_position.copy(),
            "desired_goal": goal,  # use self.goal, ensure consistency with step()
        }

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = float(self.goal_distance(achieved_goal, desired_goal))
        return np.float32(1.0 if d < self.distance_threshold else 0.0)


    def _reset_sim(self) -> bool:
        # 1. reset the simulation state
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        # 2. sample/reset all object positions
        self._sample_object()  # you can let it sample all objects

        # 3. multi-object version task variables synchronization
        self.current_task_index = 0
        self.current_target_object = self.task_sequence[0]
        self.goal = self._utils.get_site_xpos(
            self.model, self.data, f"target_{self.current_target_object}"
        ).copy()

        self._mujoco.mj_forward(self.model, self.data)  # type: ignore
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)  # type: ignore

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:  # type: ignore
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)  # type: ignore

    def goal_distance(self, goal_a, goal_b):
        # 支持批量和单个输入
        goal_a = np.array(goal_a)
        goal_b = np.array(goal_b)
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        # assign value to finger joints
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
            
    def _sample_goal(self):
        # return the target site coordinates of the current target object
        return self._utils.get_site_xpos(
            self.model, self.data, f"target_{self.current_target_object}"
        ).copy()

    def _sample_object(self):
        # not sample object, just return the current object position
        return self._utils.get_site_xpos(
            self.model, self.data, f"{self.current_target_object}_site"
        ).copy()

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)  # type: ignore
        return current_quat

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_body_state(self, name) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        body_xpos = self.data.xpos[body_id]
        body_xquat = self.data.xquat[body_id]
        body_state = np.concatenate([body_xpos, body_xquat])
        return body_state

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2

