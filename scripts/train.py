
#!/usr/bin/env python
"""
TQC(+HER) training script, process-safe version
"""

import os
import sys
from pathlib import Path
import argparse
import multiprocessing as mp

import torch
import gymnasium as gym
import panda_mujoco_gym                 # noqa: register env

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# linear_schedule compatible import

try:
    from stable_baselines3.common.utils import linear_schedule
except Exception:
    from stable_baselines3.common.utils import get_schedule_fn

    def linear_schedule(initial_value: float):
        try:
            return get_schedule_fn(f"linear_{initial_value}")
        except Exception:
                # Homemade constant→0 linear decay
                def _fn(progress_remaining: float):
                    return progress_remaining * initial_value
                return _fn

# ────────── Constants/CLI ──────────
parser = argparse.ArgumentParser()
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

ENV_ID = "FrankaShelfPNPSparse-v0" if args.sparse else "FrankaShelfPNPDense-v0"
DEVICE = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

N_ENVS      = 4
TOTAL_STEPS = 2_000_000
SAVE_EVERY  = 200_000
LOG_DIR     = Path("./logs/tqc_franka")
CKPT_DIR    = Path("./checkpoints"); CKPT_DIR.mkdir(exist_ok=True)

# ────────── Env factory ──────────
def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(ENV_ID)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.task_sequence = ["cube1"]
        return env
    return _init


def main():
    print(f"==> Training on {ENV_ID} | device={DEVICE}")

    # Create VecEnv
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    eval_env  = DummyVecEnv([lambda: gym.make(ENV_ID)])
    eval_env  = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # Model
    model = TQC(
        "MultiInputPolicy",
        train_env,
        device=DEVICE,
        verbose=1,
        learning_rate=linear_schedule(3e-4),
        buffer_size=500_000,
        batch_size=512,
        gamma=0.95,
        tau=0.005,
        ent_coef="auto",
        target_entropy="auto",
        top_quantiles_to_drop_per_net=2,
        policy_kwargs=dict(
            log_std_init=-3,
            net_arch=[256, 256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=str(LOG_DIR),
    )

    # Callback
    chkpt_cb = CheckpointCallback(
        save_freq=SAVE_EVERY // N_ENVS,
        save_path=str(CKPT_DIR),
        name_prefix=f"tqc_{'sparse' if args.sparse else 'dense'}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=SAVE_EVERY // N_ENVS,
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(total_timesteps=TOTAL_STEPS, callback=[chkpt_cb, eval_cb])

    # Save
    model.save(CKPT_DIR / f"tqc_final_{'sparse' if args.sparse else 'dense'}")
    train_env.save(CKPT_DIR / "vecnormalize.pkl")
    train_env.close(); eval_env.close()
    print("✓ Training finished & model saved")


# ──────────────────────────
if __name__ == "__main__":
    # Manually specify fork for Linux/mac to avoid spawn importing itself in main process
    if os.name == "posix":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:  # Already set
            pass
    main()
