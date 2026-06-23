"""Module 8 (training) — BC warm-start + PPO with a 3x64 MLP + curriculum.

Trains an SB3 PPO agent over the 24-D observation with hybrid outer-loop
actions (v_des NED + yaw rate -> geo_control inner loop in env.py).

Optional BC warm-start from fly2 --log demos:
    uv run -m rl.fly2 --mode course --log rl/data/demo.jsonl   # live sim
    uv run -m rl.bc_dataset --demo rl/data/demo.jsonl
    uv run -m rl.train_ppo --bc rl/data/demo.jsonl

Full curriculum PPO:
    uv run -m rl.train_ppo                 # full curriculum
    uv run -m rl.train_ppo --quick         # tiny smoke run (verifies pipeline)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import spec
from rl.bc_dataset import train_bc
from rl.env import CURRICULUM, GateRacingEnv
from rl.policy import NET_ARCH, POLICY_PT, StandalonePolicy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "policy_ppo.zip")


def _vec_env(stage, n_envs=8, seed=0):
    return DummyVecEnv(
        [lambda i=i: GateRacingEnv(stage=stage, seed=seed + i) for i in range(n_envs)]
    )


def load_bc_into_ppo(model: PPO, bc_path: str) -> None:
    """Copy StandalonePolicy weights from BC checkpoint into SB3 actor."""
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    std = StandalonePolicy(
        obs_dim=ckpt.get("obs_dim", spec.OBS_DIM),
        act_dim=ckpt.get("act_dim", spec.ACTION_DIM),
        arch=ckpt.get("arch", NET_ARCH),
    )
    std.load_state_dict(ckpt["state_dict"])
    model.policy.mlp_extractor.policy_net.load_state_dict(std.body.state_dict())
    model.policy.action_net.load_state_dict(std.head.state_dict())
    print(f"[ppo] loaded BC weights from {bc_path}", flush=True)


def export_policy(model: PPO, out: str = POLICY_PT):
    """Copy SB3 PPO actor weights into a StandalonePolicy and save."""
    sb3 = model.policy
    std = StandalonePolicy()
    # SB3 MlpExtractor policy_net + action_net -> our body + head.
    src_body = sb3.mlp_extractor.policy_net.state_dict()
    std.body.load_state_dict(src_body)
    std.head.load_state_dict(sb3.action_net.state_dict())
    std.eval()
    torch.save(
        {
            "state_dict": std.state_dict(),
            "arch": NET_ARCH,
            "obs_dim": spec.OBS_DIM,
            "act_dim": spec.ACTION_DIM,
            "action_space": "velocity",
        },
        out,
    )
    print(f"[ppo] exported standalone policy -> {out}", flush=True)
    return std


def _verify_export(model, std, n=64):
    """Deterministic SB3 action == StandalonePolicy output (within tol)."""
    obs = np.random.uniform(-1, 1, (n, spec.OBS_DIM)).astype(np.float32)
    sb3_act, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        mine = std(torch.from_numpy(obs)).numpy()
    mine = np.clip(mine, -1, 1)
    sb3_act = np.clip(sb3_act, -1, 1)
    err = np.abs(sb3_act - mine).max()
    print(f"[ppo] export parity max_abs_action_diff={err:.5f}")
    return err


def train(
    total_per_stage=300_000,
    n_envs=8,
    quick=False,
    seed=0,
    bc_demo: str | None = None,
    bc_epochs: int = 30,
):
    os.makedirs(DATA_DIR, exist_ok=True)
    if quick:
        total_per_stage, n_envs = 4000, 4

    if bc_demo and os.path.isfile(bc_demo):
        train_bc(bc_demo, POLICY_PT, epochs=bc_epochs if not quick else 5)

    policy_kwargs = dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh)
    env0 = _vec_env(0, n_envs, seed)
    model = PPO(
        "MlpPolicy",
        env0,
        policy_kwargs=policy_kwargs,
        verbose=0,
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

    if bc_demo and os.path.isfile(bc_demo) and os.path.isfile(POLICY_PT):
        load_bc_into_ppo(model, POLICY_PT)

    for stage in range(len(CURRICULUM)):
        model.set_env(_vec_env(stage, n_envs, seed + 1000 * stage))
        print(
            f"[ppo] === stage {stage} ({CURRICULUM[stage]['num_gates']} gates) "
            f"x {total_per_stage} steps ===",
            flush=True,
        )
        model.learn(
            total_timesteps=total_per_stage,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        # Quick eval.
        rew = evaluate(model, stage=stage, episodes=5)
        print(f"[ppo] stage {stage} mean eval reward={rew:.1f}", flush=True)

    model.save(ZIP_PATH)
    print(f"[ppo] saved SB3 model -> {ZIP_PATH}", flush=True)
    std = export_policy(model)
    err = _verify_export(model, std)
    assert err < 1e-4, (
        f"export parity failed (max_abs_action_diff={err:.5f}); policy.pt would "
        f"not match the trained SB3 actor"
    )
    return model


def evaluate(model, stage=2, episodes=10, seed=999):
    env = GateRacingEnv(stage=stage, seed=seed)
    rews = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done = trunc = False
        tot = 0.0
        while not (done or trunc):
            a, _ = model.predict(o, deterministic=True)
            o, r, done, trunc, info = env.step(a)
            tot += r
        rews.append(tot)
    return float(np.mean(rews))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000, help="steps per stage")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--bc",
        default=None,
        help="fly2 --log JSONL for BC warm-start (also runs bc_dataset first)",
    )
    ap.add_argument("--bc-epochs", type=int, default=30)
    args = ap.parse_args()
    train(
        total_per_stage=args.steps,
        n_envs=args.envs,
        quick=args.quick,
        bc_demo=args.bc,
        bc_epochs=args.bc_epochs,
    )
