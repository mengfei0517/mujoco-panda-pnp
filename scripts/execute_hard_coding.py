import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import panda_mujoco_gym  # noqa: register env


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def interpolate_pose(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_quat: np.ndarray,
    end_quat: np.ndarray,
    num_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate linear position trajectory + SLERP orientation."""
    pos_traj = np.linspace(start_pos, end_pos, num_steps)
    key_rots = R.from_quat([start_quat, end_quat])
    slerp = Slerp([0, 1], key_rots)
    quat_traj = slerp(np.linspace(0, 1, num_steps)).as_quat()
    return pos_traj, quat_traj


# -----------------------------------------------------------------------------
# Structured controller
# -----------------------------------------------------------------------------

class PandaPickPlace:
    """Structured pick‑and‑place controller for Franka Panda (MuJoCo)."""

    # ------------------------------ init ---------------------------------- #
    def __init__(
        self,
        env: gym.Env,
        *,
        sim_steps_per_action: int = 5,
        motion_steps: int = 30,
        render: bool = True,
    ) -> None:
        self.env = env
        self.sim_steps_per_action = sim_steps_per_action
        self.motion_steps = motion_steps
        self.render_enabled = render

        # Home pose gets captured after first reset
        self.home_pos: np.ndarray | None = None
        self.home_quat: np.ndarray | None = None

        # Internal counter for pacing (not critical here but kept for extensibility)
        self._tick = 0

    # --------------------------- shortcuts --------------------------------- #
    @property
    def robot(self):
        """Access to the underlying unwrapped environment (with MuJoCo handles)."""
        return self.env.unwrapped

    def _step_simulation(self, n: int = 1) -> None:
        """Advance simulation *n* physics steps and render if enabled."""
        for _ in range(n):
            self.robot._mujoco.mj_step(self.robot.model, self.robot.data, nstep=1)
            if self.render_enabled:
                self.env.render()
        self._tick += n

    def _set_mocap(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Directly set mocap body pose (controls EE via MuJoCo constraints)."""
        self.robot.set_mocap_pose(pos, quat)

    # ---------------------------- state IO --------------------------------- #
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.robot.get_ee_position().copy()
        quat = self.robot.get_ee_orientation().copy()
        return pos, quat

    def capture_home_pose(self) -> None:
        """Remember the current EE pose as *home* (return‑to) pose."""
        self.home_pos, self.home_quat = self.get_ee_pose()
        print("[Controller] Captured home pose.")

    def move_to_home(self, steps: int | None = None) -> None:
        """Return EE to the recorded home pose and确保夹爪张开."""
        if self.home_pos is None or self.home_quat is None:
            raise RuntimeError("Home pose has not been captured yet.")
        self.move_ee_linear(self.home_pos, self.home_quat, steps)
        self.release(duration_steps=30)

    # ------------------------- motion primitives --------------------------- #
    def move_ee_to_pose(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        steps: int | None = None,
    ) -> None:
        steps = steps or self.motion_steps
        pos_traj, quat_traj = interpolate_pose(
            start_pos, end_pos, start_quat, end_quat, steps
        )
        for pos, quat in zip(pos_traj, quat_traj):
            self._set_mocap(pos, quat)
            self._step_simulation(self.sim_steps_per_action)

    def move_ee_linear(
        self,
        end_pos: np.ndarray,
        quat: np.ndarray,
        steps: int | None = None,
    ) -> None:
        start_pos, start_quat = self.get_ee_pose()
        self.move_ee_to_pose(start_pos, end_pos, start_quat, quat, steps)

    def move_ee_via(
        self,
        waypoints: List[np.ndarray],
        quat: np.ndarray,
        steps_per_seg: int | None = None,
    ) -> None:
        curr_pos, curr_quat = self.get_ee_pose()
        pts = [curr_pos] + list(waypoints)
        for p_start, p_end in zip(pts[:-1], pts[1:]):
            self.move_ee_to_pose(p_start, p_end, curr_quat, quat, steps_per_seg)

    def rotate_ee_in_place(self, delta_quat: np.ndarray, steps: int | None = None) -> None:
        """Apply *delta_quat* rotation relative to current orientation."""
        pos, quat = self.get_ee_pose()
        target_quat = (R.from_quat(quat) * R.from_quat(delta_quat)).as_quat()
        self.move_ee_to_pose(pos, pos, quat, target_quat, steps)

    # --------------------- synchronisation helpers ------------------------ #
    def wait_until_close(
        self,
        target_pos: np.ndarray,
        threshold: float = 0.005,
        timeout: float = 2.0,
    ) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            ee_pos, _ = self.get_ee_pose()
            if np.linalg.norm(ee_pos - target_pos) < threshold:
                return True
            self._step_simulation(1)
        return False

    # ------------------------ gripper control ----------------------------- #
    def set_gripper(self, open_ratio: float, duration_steps: int = 30) -> None:
        """*open_ratio* ∈ [-1, 1]; -1 = fully close, +1 = fully open."""
        for _ in range(duration_steps):
            action = np.concatenate([np.zeros(6), [open_ratio]])
            self.env.step(action)
            if self.render_enabled:
                self.env.render()
            time.sleep(0.02)

    def grasp(self, duration_steps: int = 40) -> None:
        self.set_gripper(-1.0, duration_steps)

    def release(self, duration_steps: int = 30) -> None:
        self.set_gripper(1.0, duration_steps)

    # ------------------------ high‑level behaviours ----------------------- #
    def pick(self, obj_name: str) -> None:
        """Approach, grasp and lift *obj_name*."""
        # 1) rotate EE -90° around Y so gripper fingers align with object
        delta_quat = R.from_euler("y", -90, degrees=True).as_quat()
        self.rotate_ee_in_place(delta_quat, steps=30)

        # 2) target positions
        object_site = f"{obj_name}_site"
        obj_pos = self.robot._utils.get_site_xpos(
            self.robot.model, self.robot.data, object_site
        ).copy()

        obj_y = obj_pos[1]

        mid_offset = np.array([-0.20, -obj_y, 0.05])
        grasp_offset = np.array([0.24, -obj_y, -0.06])

        mid_pos = obj_pos + mid_offset
        grasp_pos = mid_pos + grasp_offset
        _, grasp_quat = self.get_ee_pose()

        # 3) move, grasp, lift
        self.move_ee_via([mid_pos, grasp_pos], grasp_quat)
        self.wait_until_close(grasp_pos)
        self.grasp(duration_steps=40)

        lift_pos = grasp_pos + np.array([0.0, 0.0, 0.06])
        self.move_ee_linear(lift_pos, grasp_quat)

    def place(self, obj_name: str) -> None:
        """Place currently held object onto its *target_site*."""
        target_site = f"target_{obj_name}"
        target_pos = self.robot._utils.get_site_xpos(
            self.robot.model, self.robot.data, target_site
        ).copy()

        ee_pos, ee_quat = self.get_ee_pose()

        # 1) retreat for clearance (backwards in +X of world, then back along -Z)
        retreat_x = ee_pos + np.array([-0.30, 0.0, 0.0])
        # 如果z方向距离目标位置小于0.1m，则不进行z方向的移动
        if np.linalg.norm(target_pos - ee_pos) < 0.5:
            retreat_z = retreat_x
        else:
            retreat_z = retreat_x + np.array([0.0, 0.0, -0.2])
        self.move_ee_via([retreat_x, retreat_z], ee_quat)

        # 2) rotate EE back +90° about Y to original orientation
        inv_delta_quat = R.from_euler("y", 90, degrees=True).as_quat()
        self.rotate_ee_in_place(inv_delta_quat, steps=60)

        # 3) move above target, descend, release
        target_above = target_pos + np.array([0.0, 0.0, 0.10])
        self.move_ee_linear(target_above, self.get_ee_pose()[1])
        self.move_ee_linear(target_pos, self.get_ee_pose()[1])
        self.wait_until_close(target_pos)
        self.release(duration_steps=40)

        # 4) slight retract upward so gripper clears object
        retract_up = target_pos + np.array([0.0, 0.0, 0.05])
        self.move_ee_linear(retract_up, self.get_ee_pose()[1])

    # ----------------------------- routines ------------------------------- #
    def execute_pick_and_place(self, obj_name: str) -> None:
        self.pick(obj_name)
        self.place(obj_name)

    def run_task_sequence(self, task_sequence: List[str]) -> None:
        """Iterate over *task_sequence* [obj_name, ...] performing pick→place."""
        for obj in task_sequence:
            print(f"\n=== Handling '{obj}' ===")
            self.execute_pick_and_place(obj)
            self.move_to_home()


# -----------------------------------------------------------------------------
# Main script entry
# -----------------------------------------------------------------------------

def main() -> None:
    env = gym.make("FrankaShelfPNPDense-v0", render_mode="human")
    controller = PandaPickPlace(env, render=True)

    # Reset env and open gripper
    env.reset()
    controller.release(duration_steps=30)

    # Capture home pose just after reset (arm still at default position)
    controller.capture_home_pose()

    # Determine object sequence from env or fallback list
    task_sequence = getattr(env.unwrapped, "task_sequence", ["sphere", "cube", "cylinder"])

    # Run multi‑object pick & place routine
    controller.run_task_sequence(task_sequence)

    time.sleep(1)
    env.close()


if __name__ == "__main__":
    main()