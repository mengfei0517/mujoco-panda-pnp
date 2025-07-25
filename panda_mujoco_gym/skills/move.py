from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import mujoco
from .base import Skill
from .ik_solver import JacobianIKController

class MoveSkill(Skill):
    """Move EE from current to target position in a straight line (fixed orientation)."""

    def __init__(self, env, target_pos: np.ndarray, steps: int = 30, pos_thresh: float = 0.02):
        super().__init__(env)
        assert pos_thresh > 0, "pos_thresh must be positive"

        self.target_pos = np.asarray(target_pos, float)
        self.steps = steps
        self.pos_thresh = pos_thresh

        # runtime state
        self.i = 0
        self.done = False

    def reset(self):
        self.i = 0
        self.done = False
        self.start_pos = self.env.get_ee_position().copy()
        self.quat = self.env.get_ee_orientation().copy()
        # 动态调整插值步数
        dist = np.linalg.norm(self.start_pos - self.target_pos)
        if dist > 1.0:
            steps = 120
        elif dist > 0.5:
            steps = 60
        else:
            steps = 20
        self.steps = steps
        self.pos_traj = np.linspace(self.start_pos, self.target_pos, self.steps)

    def step(self):
        if self.done:
            return self.zero_action()

        if self.i < self.steps:
            pos = self.pos_traj[self.i]
            self.env.set_mocap_pose(pos, self.quat)
            self._step_sim(n=5)
            self.i += 1
        else:
            self.env.set_mocap_pose(self.target_pos, self.quat)
            if Skill.pos_close(self.env.get_ee_position(), self.target_pos, self.pos_thresh):
                self.done = True

        return self.zero_action()


class MoveIKSkill(Skill):
    """
    Adaptive IK trajectory planning: generate intermediate points using IK solver,
    considering robot constraints, instead of pre-interpolation.
    """
    def __init__(self, env, target_pos: np.ndarray, pos_thresh: float = 0.01, 
                 max_traj_points: int = 200, step_size: float = 0.01):
        super().__init__(env)
        self.target_pos = np.asarray(target_pos, float)
        self.pos_thresh = pos_thresh
        self.max_traj_points = max_traj_points  # Maximum trajectory points to prevent infinite loops
        self.step_size = step_size  # Adaptive step size for trajectory generation
        self.i = 0
        self.done = False

    def reset(self):
        self.i = 0
        self.done = False
        
        # Initialize IK controller
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        import copy
        self.tmp_data = copy.deepcopy(data)
        self.ik_controller = JacobianIKController(model, self.tmp_data)
        
        # Generate adaptive IK trajectory
        self.pos_traj = []
        self.quat_traj = []
        
        start_pos = self.env.get_ee_position().copy()
        start_quat = self.env.get_ee_orientation().copy()
        q_current = data.qpos[:7].copy()
        pos_current = start_pos.copy()
        quat_current = start_quat.copy()
        
        # Add starting point
        self.pos_traj.append(pos_current.copy())
        self.quat_traj.append(quat_current.copy())
        
        # Generate trajectory points using IK
        point_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while (np.linalg.norm(pos_current - self.target_pos) > self.pos_thresh and 
               point_count < self.max_traj_points):
            
            # Calculate direction to target
            direction = self.target_pos - pos_current
            distance = np.linalg.norm(direction)
            
            # Adaptive step size: smaller steps when closer to target or after failures
            adaptive_step = min(self.step_size, distance * 0.1)
            # Set maximum step size to prevent mocap instability
            max_step_size = 0.02  # Maximum step size for stable mocap control
            adaptive_step = min(adaptive_step, max_step_size)
            if consecutive_failures > 0:
                adaptive_step *= 0.5  # Reduce step size after failures
            
            # Calculate next target position
            if distance > adaptive_step:
                next_pos = pos_current + direction * adaptive_step / distance
            else:
                next_pos = self.target_pos.copy()
            
            # Use IK to solve for joint angles
            ik_result = self.ik_controller.solve(next_pos, q_current)
            
            # Check IK result and apply fallback strategies
            if ik_result.success and ik_result.pos_error < self.step_size * 2:
                # IK succeeded, add point to trajectory
                self.pos_traj.append(ik_result.final_pos.copy())
                self.quat_traj.append(quat_current.copy())  # Keep orientation constant for now
                
                pos_current = ik_result.final_pos.copy()
                q_current = ik_result.q.copy()
                consecutive_failures = 0  # Reset failure counter
                
            else:
                # IK failed, try fallback strategies
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    # Too many consecutive failures, try alternative approaches
                    print(f"IK failed {consecutive_failures} times, trying fallback strategies...")
                    
                    # Strategy 1: Try with smaller step size
                    smaller_step = adaptive_step * 0.1
                    if distance > smaller_step:
                        fallback_pos = pos_current + direction * smaller_step / distance
                        fallback_result = self.ik_controller.solve(fallback_pos, q_current)
                        
                        if fallback_result.success:
                            self.pos_traj.append(fallback_result.final_pos.copy())
                            self.quat_traj.append(quat_current.copy())
                            pos_current = fallback_result.final_pos.copy()
                            q_current = fallback_result.q.copy()
                            consecutive_failures = 0
                            continue
                    
                    # Strategy 2: Try moving in a different direction (e.g., only X and Z)
                    alt_direction = direction.copy()
                    alt_direction[1] = 0  # Keep Y constant
                    if np.linalg.norm(alt_direction) > 0.001:
                        alt_direction = alt_direction / np.linalg.norm(alt_direction)
                        alt_pos = pos_current + alt_direction * adaptive_step
                        alt_result = self.ik_controller.solve(alt_pos, q_current)
                        
                        if alt_result.success:
                            self.pos_traj.append(alt_result.final_pos.copy())
                            self.quat_traj.append(quat_current.copy())
                            pos_current = alt_result.final_pos.copy()
                            q_current = alt_result.q.copy()
                            consecutive_failures = 0
                            continue
                    
                    # Strategy 3: If all fallbacks fail, stop trajectory generation
                    print(f"All fallback strategies failed, stopping at point {point_count}")
                    break
                else:
                    # Just skip this point and try next iteration
                    consecutive_failures += 1
                    continue
            
            point_count += 1
        
        # Add final target point if not reached
        if np.linalg.norm(pos_current - self.target_pos) > self.pos_thresh:
            self.pos_traj.append(self.target_pos.copy())
            self.quat_traj.append(quat_current.copy())
        
        # Only print final summary, no detailed trajectory info

    def step(self):
        if self.done:
            return self.zero_action()
        
        if self.i < len(self.pos_traj):
            pos = self.pos_traj[self.i]
            quat = self.quat_traj[self.i]
            self.env.set_mocap_pose(pos, quat)
            self._step_sim(n=5)
            self.i += 1
        else:
            self.done = True
        
        return self.zero_action()
