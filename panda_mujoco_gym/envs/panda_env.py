import mujoco
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
        model_path: str = None,
        n_substeps: int = 50,
        reward_type: str = "sparse",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        obj_x_range: float = 0.05,
        obj_y_range: float = 0.2,
        **kwargs,
    ):
        # multi-object version variables initialization
        self.task_sequence = ["sphere", "cube", "cylinder"]
        self.current_task_index = 0
        self.current_target_object = self.task_sequence[0]
        self.goal = None

        # initialize other variables
        self.block_gripper = block_gripper
        self.model_path = model_path

        action_size = 3
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

        # override the methods in MujocoRobotEnv
    # -----------------------------
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
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

        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco_step()

        self.initial_object_height = self._utils.get_joint_qpos(self.model, self.data, "obj_joint")[2].copy()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()

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
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        dist = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return 1.0 if dist < self.distance_threshold else 0.0
        else:
            return -dist

    def _set_action(self, action) -> None:
        action = action.copy()
        # for the pick and place task
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])

        elif self.block_gripper:
            pos_ctrl = action
            fingers_half_width = 0

        # control the gripper
        self.data.ctrl[-2:] = fingers_half_width

        # control the end-effector with mocap body
        pos_ctrl *= 0.05
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)

    def _get_obs(self) -> dict:
        # if the attribute does not exist, set a default value
        if not hasattr(self, "current_target_object"):
            self.current_target_object = "sphere"  # or other default value
        # current target object name
        current_obj = self.current_target_object
        current_site = f"{current_obj}_site"

        # end-effector state
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()

        # current object state
        object_position = self._utils.get_site_xpos(self.model, self.data, current_site).copy()
        object_rotation = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, current_site)).copy()
        object_velp = self._utils.get_site_xvelp(self.model, self.data, current_site).copy() * self.dt
        object_velr = self._utils.get_site_xvelr(self.model, self.data, current_site).copy() * self.dt

        # current target point
        desired_goal = self._utils.get_site_xpos(self.model, self.data, f"target_{current_obj}").copy()

        if not self.block_gripper:
            obs = {
                "observation": np.concatenate([
                    ee_position,
                    ee_velocity,
                    fingers_width,
                    object_position,
                    object_rotation,
                    object_velp,
                    object_velr,
                ]).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": desired_goal.copy(),
            }
        else:
            obs = {
                "observation": np.concatenate([
                    ee_position,
                    ee_velocity,
                    object_position,
                    object_rotation,
                    object_velp,
                    object_velr,
                ]).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": desired_goal.copy(),
            }

        return obs

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    # def _render_callback(self) -> None:
    #     # visualize goal site for multi-object version
    #     sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
    #     site_id = self._model_names.site_name2id["target"]
    #     self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
    #     self._mujoco.mj_forward(self.model, self.data)

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

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
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

    def _sample_object(self) -> None:
        """
        Sample each object's position within a rectangle defined by:
        - x in [x0 - dx, x0 + dx]
        - y in [y0 - dy, y0 + dy]
        where (x0, y0, z0) is from XML (via get_joint_qpos)
        """
        joint_names = ["sphere_joint", "cube_joint", "cylinder_joint"]
        for joint_name in joint_names:
            base_qpos = self._utils.get_joint_qpos(self.model, self.data, joint_name).copy()
            base_pos = base_qpos[:3]
            quat = base_qpos[3:]
            # sample xy, keep z unchanged
            base_pos[0] += self.np_random.uniform(-self.obj_x_range, self.obj_x_range)
            base_pos[1] += self.np_random.uniform(-self.obj_y_range, self.obj_y_range)
            new_qpos = np.concatenate([base_pos, quat])
            self._utils.set_joint_qpos(self.model, self.data, joint_name, new_qpos)

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
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

