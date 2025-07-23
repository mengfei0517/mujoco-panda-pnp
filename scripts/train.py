# -*- coding: utf-8 -*-
"""train_multi_object_rl.py
===========================================
Train a Truncated Quantile Critics (TQC) agent (wrapped with HER) on the
multi‑object *Franka Shelf Pick‑and‑Place* task.

This revision creates the environment **exactly** the same way you would in a
one‑liner:

```python
env = gym.make("FrankaShelfPNPDense-v0", render_mode="human")
```

but with optional sparse/dense switching, statistics collection and a short
time‑limit wrapper for faster exploration.

Usage
-----
Train for 3 M steps on 8 parallel envs with dense rewards and verbose prints:

```bash
python train_multi_object_rl.py --timesteps 3e6 --vec-envs 8 \
    --dense --print-every 5000
```

Evaluate a saved model:

```bash
python train_multi_object_rl.py --eval-only runs/tqc_franka_sparse_0_20250722_153200/best_model.zip
```
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MultiInputPolicy
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import panda_mujoco_gym

# -----------------------------------------------------------------------------
# Environment factory (now mirrors the simple gym.make() call) -----------------
# -----------------------------------------------------------------------------

ENV_SPARSE_ID = "FrankaShelfPNPSparse-v0"
ENV_DENSE_ID = "FrankaShelfPNPDense-v0"


def make_franka_env(
    *,                      # ← 只接受显式关键字参数
    reward_type: str = "sparse",
    render_mode: str | None = None,
):
    env_id = ENV_DENSE_ID if reward_type == "dense" else ENV_SPARSE_ID

    env = gym.make(env_id, render_mode=render_mode)      # ✅ 不再传 distance_threshold
    env = TimeLimit(env, max_episode_steps=300)
    env = RecordEpisodeStatistics(env)
    return env


# -----------------------------------------------------------------------------
# Helper callback to print training progress -----------------------------------
# -----------------------------------------------------------------------------

class ProgressPrinter(BaseCallback):
    """Print useful stats every *N* environment steps."""

    def __init__(self, print_every: int = 5000):
        super().__init__()
        self.print_every = print_every

    def _on_step(self) -> bool:  # type: ignore[override]
        # 直接用 name_to_value
        ep_rew = self.logger.name_to_value.get("rollout/ep_rew_mean", np.nan)
        succ_rate = self.logger.name_to_value.get("rollout/success_rate", np.nan)
        q_std = self.locals.get("qf0_std", np.nan)
        alpha = self.locals.get("entropy_coef", np.nan)
        if self.num_timesteps % self.print_every == 0:
            print(
                f"step={self.num_timesteps:>8}  R̄={ep_rew:>6.2f}  succ={succ_rate:>5.1%}  "
                f"Qstd={q_std if isinstance(q_std, float) else np.nan:.3f}  "
                f"α={alpha if isinstance(alpha, float) else np.nan:.3f}"
            )
        return True


# -----------------------------------------------------------------------------
# Training routine -------------------------------------------------------------
# -----------------------------------------------------------------------------

def train(
    total_timesteps: int = 2_000_000,
    seed: int = 0,
    logdir: str = "runs",
    use_dense_reward: bool = False,
    n_envs: int = 4,
    print_every: int = 5000,
) -> Path:
    """Train TQC agent with HER and return the path to the saved model."""

    os.makedirs(logdir, exist_ok=True)
    run_name = (
        f"tqc_franka_{'dense' if use_dense_reward else 'sparse'}_{seed}_"
        f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(logdir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Create vectorised environment -----------------------------------------
    env_kwargs: Dict[str, Any] = {
        "reward_type": "dense" if use_dense_reward else "sparse",
    }

    vec_env = make_vec_env(
        make_franka_env,  # type: ignore[arg-type]
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        # wrappers already applied inside factory
    )

    # 2) Evaluation environment ------------------------------------------------
    eval_env = make_franka_env(reward_type=env_kwargs["reward_type"], render_mode=None)

    # 3) HER replay buffer -----------------------------------------------------
    goal_selection_strategy = GoalSelectionStrategy.FUTURE

    replay_buffer_kwargs: Dict[str, Any] = dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        # online_sampling=True, # because terminal error
        # max_episode_length=50, # because terminal error
    )

    # 4) Instantiate TQC -------------------------------------------------------
    model = TQC(
        policy=MultiInputPolicy,
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=50_000,
        batch_size=512,
        tau=0.02,
        gamma=0.95,
        train_freq=256,
        gradient_steps=256,
        target_entropy="auto",
        top_quantiles_to_drop_per_net=2,
        # n_critics=2, # because terminal error
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        seed=seed,
    )

    # 5) Callbacks -------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_log"),
        eval_freq=10_000,
        n_eval_episodes=15,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=0.0, verbose=1),
    )

    printer_callback = ProgressPrinter(print_every=print_every)

    # 6) Train ---------------------------------------------------------------
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, printer_callback],
        progress_bar=True,
    )

    # 7) Save model ----------------------------------------------------------
    model_path = out_dir / "tqc_franka_final.zip"
    model.save(model_path)
    return model_path


# -----------------------------------------------------------------------------
# CLI interface ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train TQC on multi‑object Franka PnP task")
    parser.add_argument("--timesteps", type=float, default=2e6, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dense", action="store_true", help="Use dense reward")
    parser.add_argument("--logdir", type=str, default="runs", help="Logging directory")
    parser.add_argument("--vec-envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--print-every", type=int, default=5000, help="Print interval in env steps")
    parser.add_argument("--eval-only", type=str, default=None, help="Path to a saved model to evaluate only")
    return parser.parse_args()


def evaluate(model_path: str, n_episodes: int = 20):
    from stable_baselines3.common.evaluation import evaluate_policy

    env = make_franka_env(render_mode="human")
    model = TQC.load(model_path, env=env)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, render=True, deterministic=True
    )
    print(f"Evaluation over {n_episodes} episodes — mean R: {mean_reward:.2f} ± {std_reward:.2f}")


def main():
    args = parse_args()

    if args.eval_only is not None:
        evaluate(args.eval_only)
        return

    model_path = train(
        total_timesteps=int(args.timesteps),
        seed=args.seed,
        logdir=args.logdir,
        use_dense_reward=args.dense,
        n_envs=args.vec_envs,
        print_every=args.print_every,
    )
    print(f"\nTraining finished. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
