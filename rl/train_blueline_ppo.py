"""PPO training for DroneBlueLineEnv (blue-line corridor, no YOLO).

uv run -m rl.train_blueline_ppo              # 1M steps, curriculum
uv run -m rl.train_blueline_ppo --quick      # smoke
uv run -m rl.train_blueline_ppo --selftest
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.blueline_env import CURRICULUM, DroneBlueLineEnv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BEST_DIR = os.path.join(DATA_DIR, "blueline_best")
ZIP_PATH = os.path.join(DATA_DIR, "blueline_ppo.zip")

TOTAL_TIMESTEPS = 1_000_000
NET_ARCH = [64, 64, 64]


def _make_env(stage: int, seed: int, rank: int = 0, monitor: bool = False):
    def _thunk():
        env = DroneBlueLineEnv(stage=stage, seed=seed + rank)
        if monitor:
            env = Monitor(env)
        return env

    return _thunk


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = 8,
    quick: bool = False,
    seed: int = 0,
):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    if quick:
        total_timesteps, n_envs = 4_000, 2

    # Split budget across curriculum stages (3 -> 15 gates).
    n_stages = len(CURRICULUM)
    per_stage = max(total_timesteps // n_stages, 1)

    train_env = DummyVecEnv(
        [_make_env(0, seed, i, monitor=True) for i in range(n_envs)]
    )
    eval_env = DummyVecEnv([_make_env(0, seed + 10_000, monitor=True)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=BEST_DIR,
        eval_freq=max(5_000 // n_envs, 500),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)),
        verbose=1,
        n_steps=1024 if not quick else 256,
        batch_size=256 if not quick else 64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.005,
        learning_rate=3e-4,
        clip_range=0.2,
        n_epochs=10,
        seed=seed,
        device="cpu",
    )

    for stage in range(n_stages):
        cfg = CURRICULUM[stage]
        train_env = DummyVecEnv(
            [
                _make_env(stage, seed + 1000 * stage, i, monitor=True)
                for i in range(n_envs)
            ]
        )
        eval_env = DummyVecEnv([_make_env(stage, seed + 20_000 + stage, monitor=True)])
        eval_cb.eval_env = eval_env
        model.set_env(train_env)
        print(
            f"[blueline-ppo] stage {stage} ({cfg['num_gates']} gates) "
            f"x {per_stage} steps",
            flush=True,
        )
        model.learn(
            total_timesteps=per_stage,
            callback=eval_cb,
            reset_num_timesteps=False,
            progress_bar=False,
        )

    model.save(ZIP_PATH)
    print(f"[blueline-ppo] saved -> {ZIP_PATH}", flush=True)
    print(f"[blueline-ppo] best weights -> {BEST_DIR}", flush=True)
    return model


def _selftest():
    train(total_timesteps=2_000, n_envs=2, quick=True, seed=0)
    assert os.path.exists(ZIP_PATH) or os.path.exists(
        os.path.join(BEST_DIR, "best_model.zip")
    )
    print("[selftest] OK — blueline PPO smoke train finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        train(
            total_timesteps=args.steps,
            n_envs=args.envs,
            quick=args.quick,
        )
