"""
GripperSkill – Unified gripper primitive that covers both 'close' and 'open'.

End condition is consistent with the old version:
    done = (step ≥ duration) AND (width satisfies fingers_(closed/open)).

If the environment lacks `get_gripper_width()`:
    • close → width defaults to 0.0 ⇒ fingers_closed() always True
    • open  → width defaults to np.inf ⇒ fingers_open() always True
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from .base import Skill


class GripperSkill(Skill):

    def __init__(
        self,
        env,
        mode: Literal["close", "open"],
        *,
        duration: int | None = None,
        thresh: float | None = None,
    ):
        super().__init__(env)

        assert mode in ("close", "open"), "mode must be 'close' or 'open'"
        self.mode      = mode
        self.duration  = duration if duration is not None else (10 if mode == "close" else 15)
        self.thresh    = thresh   if thresh   is not None else (0.02 if mode == "close" else 0.08)
        self.i         = 0        # step counter
        self.done      = False

    # ── Convenient factory ──────────────────────────────────────────────
    @classmethod
    def close(cls, env, **kw):   # noqa: D401
        return cls(env, "close", **kw)

    @classmethod
    def open(cls, env, **kw):    # noqa: D401
        return cls(env, "open", **kw)

    # --------------------------------------------------------------- #
    def reset(self):
        self.i    = 0
        self.done = False

    # --------------------------------------------------------------- #
    def _current_width(self) -> float:
        """Get gripper width; return default value if environment lacks interface."""
        default = 0.0 if self.mode == "close" else np.inf
        get_w   = getattr(self.env, "get_gripper_width", None)
        if callable(get_w):
            try:
                w = float(get_w())
                return w if np.isfinite(w) else default
            except Exception:
                return default
        return default

    # --------------------------------------------------------------- #
    def step(self):
        if self.done:
            return np.zeros(7, dtype=np.float32)

        # Construct action (only control gripper channel)
        action       = np.zeros(7, dtype=np.float32)
        action[-1]   = -1.0 if self.mode == "close" else 1.0
        self.env.step(action)
        self._step_sim(n=5)
        self.i += 1

        width = self._current_width()

        if self.mode == "close":
            cond_width = Skill.fingers_closed(width, self.thresh)
        else:  # open
            cond_width = Skill.fingers_open(width, self.thresh)

        # ── End condition ──────────────────────────────────────────────
        if (self.i >= self.duration) and cond_width:
            self.done = True

        return action
