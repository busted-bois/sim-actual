"""Train (P2) and evaluate (P3) the residual-RL speed policy with PPO.

Single real-time env (n_envs=1), VecNormalize obs, 3x64 MLP. A curriculum callback drives
the reverse-curriculum start gate and the reliability->speed time-weight schedule.

Usage (via Makefile):
  uv run python -m simulator.rl_train train   # -> policy.pt
  uv run python -m simulator.rl_train eval     # finish-rate + lap-time vs bare pilot

Requires the RL dependency group:  uv sync --group rl
And the live simulator running (real-time training).
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np

from simulator.rl_core import (
    RL_DT_S,
    identity_residual,
    time_weight_schedule,
)

TOTAL_TIMESTEPS = 300_000
RELIABILITY_FRAC = 0.5  # first half = reliability phase (time_weight 0)
HARDEST_GATE = 2
POLICY_PATH = "policy.pt"
VECNORM_PATH = "vecnormalize.pkl"


def _make_env():
    from simulator.rl_env import ResidualRacingEnv

    return ResidualRacingEnv(hardest_gate=HARDEST_GATE)


def train(total_timesteps: int = TOTAL_TIMESTEPS) -> None:
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from simulator.policy_runtime import export_from_sb3

    venv = DummyVecEnv([_make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    class CurriculumCallback(BaseCallback):
        def _on_step(self) -> bool:
            p = min(1.0, self.num_timesteps / total_timesteps)
            w = time_weight_schedule(p, RELIABILITY_FRAC)
            # VecNormalize -> DummyVecEnv -> ResidualRacingEnv
            self.training_env.env_method("set_curriculum", p, w)
            return True

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh
        ),
        verbose=1,
    )
    print(
        f"[train] PPO start: {total_timesteps} steps, reliability_frac={RELIABILITY_FRAC}",
        flush=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=CurriculumCallback())

    venv.save(VECNORM_PATH)
    export_from_sb3(model, venv, POLICY_PATH)
    print(f"[train] saved deploy policy -> {POLICY_PATH}", flush=True)


def _run_episode(env, policy_runtime=None, max_steps: int = 1500):
    """Run one full-track episode (start gate 0). Returns (finished, lap_time_s)."""
    env.set_curriculum(progress01=1.0, time_weight=0.0)  # full track, start gate 0
    obs, _ = env.reset()
    t0 = time.monotonic()
    for _ in range(max_steps):
        if policy_runtime is None:
            env.pilot.residual = identity_residual()
            action = np.zeros(3, dtype=np.float32)
            time.sleep(RL_DT_S)
            env._steps += 1
            obs = env._build_obs()
            terminated = (env.shared.get("collision") is not None) or (
                int(env.shared.get("active_gate_index", 0) or 0) >= len(env._gates())
                and len(env._gates()) > 0
            )
            truncated = env._steps >= max_steps
            info = {
                "finished": int(env.shared.get("active_gate_index", 0) or 0)
                >= len(env._gates())
            }
        else:
            action = policy_runtime.act(obs)
            obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return bool(info.get("finished", False)), time.monotonic() - t0
    return False, time.monotonic() - t0


def evaluate(n_runs: int = 20) -> None:
    from simulator.policy_runtime import PolicyRuntime
    from simulator.rl_env import ResidualRacingEnv

    env = ResidualRacingEnv(hardest_gate=HARDEST_GATE)

    # Bare-pilot baseline.
    base_finishes, base_times = 0, []
    for i in range(n_runs):
        fin, t = _run_episode(env, policy_runtime=None)
        base_finishes += int(fin)
        if fin:
            base_times.append(t)
        print(
            f"[eval] baseline run {i + 1}/{n_runs}: finished={fin} time={t:.1f}s",
            flush=True,
        )

    # Policy.
    pol = PolicyRuntime.load(POLICY_PATH)
    env.pilot.external_residual = True
    pol_finishes, pol_times = 0, []
    for i in range(n_runs):
        fin, t = _run_episode(env, policy_runtime=pol)
        pol_finishes += int(fin)
        if fin:
            pol_times.append(t)
        print(
            f"[eval] policy run {i + 1}/{n_runs}: finished={fin} time={t:.1f}s",
            flush=True,
        )

    env.close()

    base_rate = base_finishes / n_runs
    pol_rate = pol_finishes / n_runs
    base_med = statistics.median(base_times) if base_times else float("inf")
    pol_med = statistics.median(pol_times) if pol_times else float("inf")
    improve = (
        (base_med - pol_med) / base_med * 100
        if base_med not in (0, float("inf"))
        else 0.0
    )

    print("\n==== EVAL GATE (P3) ====", flush=True)
    print(f"baseline: finish-rate {base_rate:.0%}, median {base_med:.1f}s", flush=True)
    print(f"policy:   finish-rate {pol_rate:.0%}, median {pol_med:.1f}s", flush=True)
    print(f"time improvement: {improve:.1f}%", flush=True)
    deploy_ok = pol_rate >= base_rate and improve >= 10.0
    print(
        f"DEPLOY GATE {'PASS' if deploy_ok else 'FAIL'} (need finish-rate >= baseline AND >=10% faster)",
        flush=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "eval":
        evaluate()
    else:
        print(f"unknown command '{cmd}' (use: train | eval)", flush=True)
        sys.exit(2)
