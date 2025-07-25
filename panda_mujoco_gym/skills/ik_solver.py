from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from .base import Skill
import mujoco
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IKResult:
    """Result of IK solving with detailed information."""
    success: bool  # Whether IK converged successfully
    q: np.ndarray  # Final joint angles (7,)
    final_pos: np.ndarray  # Final end-effector position (3,)
    pos_error: float  # Final position error (distance)
    iterations: int  # Number of iterations used
    converged: bool  # Whether converged within threshold

class JacobianIKController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, site_name: str = "ee_center_site"):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.joint_ids = np.arange(7)
        self.lower = model.jnt_range[:7, 0].copy()
        self.upper = model.jnt_range[:7, 1].copy()

    def solve(self, target_pos: np.ndarray, q_init: np.ndarray,
              max_iters: int = 100, pos_thresh: float = 1e-3,
              damping: float = 1e-2, step_limit: float = 0.1) -> IKResult:
        """
        Jacobian-based IK solver for end-effector position.
        Args:
            target_pos: Desired end-effector position (3,)
            q_init: Initial joint angles (7,)
            max_iters: Maximum number of iterations
            pos_thresh: Position error threshold for convergence
            damping: Damping factor for pseudo-inverse
            step_limit: Maximum step size per iteration
        Returns:
            IKResult: Structured result containing success status, joint angles, and error info
        """
        q = q_init.copy()
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)

        converged = False
        iterations = 0
        
        for i in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            curr_pos = self.data.site_xpos[self.site_id].copy()
            pos_err = target_pos - curr_pos
            pos_error_norm = np.linalg.norm(pos_err)

            # Check for convergence
            if pos_error_norm < pos_thresh:
                converged = True
                iterations = i + 1
                break

            # Compute Jacobian for the end-effector position
            J_pos = np.zeros((3, self.model.nv))
            _ = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, J_pos, _, self.site_id)

            J = J_pos[:3, :]
            err = pos_err[:3]

            # Damped least-squares solution for joint update
            JT = J.T
            delta_q_full = JT @ np.linalg.solve(J @ JT + damping * np.eye(3), err)
            delta_q = np.clip(delta_q_full[:7], -step_limit, step_limit)
            q = np.clip(q + delta_q, self.lower, self.upper)
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            
            iterations = i + 1

        # Get final position and error
        final_pos = self.data.site_xpos[self.site_id].copy()
        final_error = np.linalg.norm(final_pos - target_pos)
        
        # Determine success based on convergence and final error
        success = converged and final_error < pos_thresh * 2  # Allow some tolerance
        
        return IKResult(
            success=success,
            q=q.copy(),
            final_pos=final_pos.copy(),
            pos_error=final_error,
            iterations=iterations,
            converged=converged
        )
