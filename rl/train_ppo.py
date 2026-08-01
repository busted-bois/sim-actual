"""Module 8 (training) — PPO with a 3x64 MLP policy + curriculum.

Trains an SB3 PPO agent over the 24-D observation across the 3 curriculum
stages (single gate -> two gates -> full 6-gate course), reusing weights
between stages. Exports a dependency-light ``policy.pt`` (pure-torch
deterministic actor) for deployment, plus the full SB3 zip for resuming.

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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import spec
from rl.env import CURRICULUM, GateRacingEnv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "policy_ppo.zip")
POLICY_PT = os.path.join(DATA_DIR, "policy.pt")
POLICY_BC_PT = os.path.join(DATA_DIR, "policy_bc.pt")
TB_DIR = os.path.join(DATA_DIR, "tb")
CKPT_DIR = os.path.join(DATA_DIR, "ckpts")
BEST_DIR = os.path.join(DATA_DIR, "best")

NET_ARCH = [64, 64, 64]  # 3x64 MLP (shared depth for pi and vf)
# Per-stage step budget: the short early stages are quickly solved; the long
# chaining stages (6- and 17-gate) get the bulk of the compute — that's where
# completion is won.
STEPS_SCHEDULE = [150_000, 150_000, 400_000, 600_000]


def _tb_dir():
    """TB_DIR when tensorboard is importable, else None (SB3 hard-errors on a
    tensorboard_log path if the package is missing — never let logging ground
    a training run)."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("[ppo] tensorboard not installed — skipping TB logs", flush=True)
        return None
    return TB_DIR


def resolve_device(pref: str = "auto") -> str:
    """'auto' -> cuda when available else cpu; else honor the explicit choice."""
    if pref == "cpu":
        return "cpu"
    if pref in ("cuda", "gpu") and not torch.cuda.is_available():
        print("[ppo] WARNING: cuda requested but unavailable — falling back to cpu")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _vec_env(stage, n_envs=8, seed=0, randomize=True):
    # Training envs randomize the plant (sim-to-real robustness); eval envs pass
    # randomize=False for a deterministic, honest metric.
    return DummyVecEnv(
        [
            lambda i=i: GateRacingEnv(stage=stage, seed=seed + i, randomize=randomize)
            for i in range(n_envs)
        ]
    )


class RaceMetricsCallback(BaseCallback):
    """Log per-episode race outcomes to TensorBoard: gates passed, completion %,
    and collisions by cause (ground / out_of_bounds / flipped)."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "gates_passed_total" not in info:
                continue  # only episode-end infos carry the summary
            self.logger.record_mean("race/gates_passed", float(info["gates_passed_total"]))
            self.logger.record_mean(
                "race/completion_pct", 100.0 * float(info.get("completed", False))
            )
            cause = info.get("crash")
            for c in ("ground", "out_of_bounds", "flipped"):
                self.logger.record_mean(f"race/crash_{c}", 1.0 if cause == c else 0.0)
        return True


class StandalonePolicy(nn.Module):
    """Pure-torch deterministic actor: obs(24) -> 3x64 tanh -> action(4)."""

    def __init__(self, obs_dim=spec.POLICY_OBS_DIM, act_dim=spec.ACTION_DIM, arch=NET_ARCH):
        super().__init__()
        layers, last = [], obs_dim
        for h in arch:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, act_dim)

    def forward(self, x):
        return self.head(self.body(x))


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
            "obs_dim": spec.POLICY_OBS_DIM,
            "act_dim": spec.ACTION_DIM,
            "obs_stack": spec.OBS_STACK,
            # Training-plant contract: deploy must interpret [-1,1] actions
            # with the scales the policy was trained against, not whatever
            # spec says at load time.
            "train_hover_thrust": spec.HOVER_THRUST,
            "action_scale": [
                spec.MAX_ROLL_RATE,
                spec.MAX_PITCH_RATE,
                spec.MAX_YAW_RATE,
            ],
        },
        out,
    )
    print(f"[ppo] exported standalone policy -> {out}", flush=True)
    return std


def _verify_export(model, std, n=64):
    """Deterministic SB3 action == StandalonePolicy output (within tol)."""
    obs = np.random.uniform(-1, 1, (n, spec.POLICY_OBS_DIM)).astype(np.float32)
    sb3_act, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        mine = std(torch.from_numpy(obs)).numpy()
    mine = np.clip(mine, -1, 1)
    sb3_act = np.clip(sb3_act, -1, 1)
    err = np.abs(sb3_act - mine).max()
    print(f"[ppo] export parity max_abs_action_diff={err:.5f}")
    return err


def load_bc_init(model: PPO, path: str) -> None:
    """Warm-start the PPO actor from a BC-pretrained StandalonePolicy."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt.get("arch") == NET_ARCH, f"BC arch {ckpt.get('arch')} != {NET_ARCH}"
    std = StandalonePolicy()
    std.load_state_dict(ckpt["state_dict"])
    # Inverse of export_policy: our body + head -> SB3 policy_net + action_net.
    model.policy.mlp_extractor.policy_net.load_state_dict(std.body.state_dict())
    model.policy.action_net.load_state_dict(std.head.state_dict())
    print(f"[ppo] warm-started actor from BC weights {path}", flush=True)


def train(
    total_per_stage=None,
    n_envs=8,
    quick=False,
    seed=0,
    bc_init=None,
    device="auto",
):
    os.makedirs(DATA_DIR, exist_ok=True)
    # Per-stage step schedule: explicit --steps overrides it uniformly; else the
    # hard long stages get more (STEPS_SCHEDULE).
    if total_per_stage is not None:
        schedule = [int(total_per_stage)] * len(CURRICULUM)
    else:
        schedule = list(STEPS_SCHEDULE)
    if quick:
        schedule, n_envs = [4000] * len(CURRICULUM), 4

    device = resolve_device(device)
    print(f"[ppo] training on device={device}", flush=True)

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
        ent_coef=0.01,  # was 0.005 — more exploration to escape the gate-1 optimum
        learning_rate=3e-4,
        clip_range=0.2,
        n_epochs=10,
        seed=seed,
        device=device,
        tensorboard_log=_tb_dir(),
    )

    if bc_init is None and os.path.exists(POLICY_BC_PT):
        bc_init = POLICY_BC_PT  # make train-bc output found -> warm start
    if bc_init:
        load_bc_init(model, bc_init)

    ckpt_freq = max(1, (100_000 if not quick else 2000) // n_envs)
    eval_freq = max(1, (50_000 if not quick else 1000) // n_envs)
    for stage in range(len(CURRICULUM)):
        total_per_stage = schedule[stage]
        model.set_env(_vec_env(stage, n_envs, seed + 1000 * stage))
        # Held-out, NON-randomized eval env at this stage (separate seed range).
        eval_env = _vec_env(stage, 1, seed + 500_000 + stage, randomize=False)
        callbacks = [
            RaceMetricsCallback(),
            CheckpointCallback(
                save_freq=ckpt_freq,
                save_path=CKPT_DIR,
                name_prefix=f"ppo_s{stage}",
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(BEST_DIR, f"s{stage}"),
                log_path=os.path.join(BEST_DIR, f"s{stage}"),
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=0,
            ),
        ]
        print(
            f"[ppo] === stage {stage} ({CURRICULUM[stage]['num_gates']} gates) "
            f"x {total_per_stage} steps ===",
            flush=True,
        )
        model.learn(
            total_timesteps=total_per_stage,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=callbacks,
            tb_log_name=f"stage{stage}",
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
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="uniform steps/stage override (default: per-stage STEPS_SCHEDULE)",
    )
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--bc-init",
        default=None,
        help="BC checkpoint to warm-start the actor (default: "
        "rl/data/policy_bc.pt when it exists)",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="auto (default) uses the GPU when available",
    )
    args = ap.parse_args()
    train(
        total_per_stage=args.steps,
        n_envs=args.envs,
        quick=args.quick,
        bc_init=args.bc_init,
        device=args.device,
    )
