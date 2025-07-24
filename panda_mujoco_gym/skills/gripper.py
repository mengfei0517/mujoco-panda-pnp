from __future__ import annotations

import numpy as np
from typing import Literal

from .base import Skill


class GripperSkill(Skill):
    """
    Unified gripper primitive that covers both 'close' and 'open'.

    结束条件与旧版保持一致：
        done = (step ≥ duration) AND (宽度满足 fingers_(closed/open)).

    若环境缺少 `get_gripper_width()`：
        • close → width 默认为 0.0 ⇒ fingers_closed() 恒 True
        • open  → width 默认为 np.inf ⇒ fingers_open()  恒 True
    """

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

    # -------- 便捷工厂 -------- #
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
        """获取夹爪宽度；若环境无接口则返回默认值."""
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

        # 构造动作（仅控制 gripper channel）
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

        # 与旧版保持一致：计时 AND 宽度
        if (self.i >= self.duration) and cond_width:
            self.done = True

        return action
