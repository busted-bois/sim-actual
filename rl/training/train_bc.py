"""BC pretrain — clone demos into the PPO actor network.

MSE-regresses the StandalonePolicy (same 3x64 tanh MLP the PPO actor uses)
onto demonstration data, saving ``policy_bc.pt`` in the same dict schema as
``policy.pt`` so ``rl.training.train_ppo`` can warm-start from it.

    uv run -m rl.training.train_bc --config configs/default.yaml
    uv run -m rl.training.train_bc --config configs/default.yaml --smoke 200
    uv run -m rl.training.train_bc --selftest
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from rl.core import spec
from rl.core.config import RLConfig, load_config
from rl.training.train_ppo import NET_ARCH, StandalonePolicy

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEMOS_PATH = os.path.join(DATA_DIR, "gp_demos.npz")
POLICY_BC_PT = os.path.join(DATA_DIR, "policy_bc.pt")

EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3


def load_demos(path: str = DEMOS_PATH) -> tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(path):
        raise SystemExit(f"[bc] no demos at {path} — run `make log-demos` first")
    with np.load(path) as d:
        cur = [spec.MAX_ROLL_RATE, spec.MAX_PITCH_RATE, spec.MAX_YAW_RATE]
        stamped = "action_scale" in d.files and np.allclose(d["action_scale"], cur)
        if stamped and "hover_thrust" in d.files:
            stamped = np.allclose(d["hover_thrust"], spec.HOVER_THRUST)
        if not stamped:
            got = (
                d["action_scale"].tolist() if "action_scale" in d.files else "unstamped"
            )
            raise SystemExit(
                f"[bc] demos at {path} were logged under action scales {got}, "
                f"but this build uses {cur} — re-run `make log-demos` (stale "
                "demos would clone the wrong normalization into the policy)"
            )
        obs = torch.from_numpy(d["obs"].astype(np.float32))
        act = torch.from_numpy(d["act"].astype(np.float32))
    assert obs.shape[1] == spec.OBS_DIM and act.shape[1] == spec.ACTION_DIM
    return obs, act


def train_bc(
    obs: torch.Tensor,
    act: torch.Tensor,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = 0,
    arch: list[int] | None = None,
) -> tuple[StandalonePolicy, list[float]]:
    if arch is None:
        arch = NET_ARCH
    torch.manual_seed(seed)
    policy = StandalonePolicy(arch=arch)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(obs)
    losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(policy(obs[idx]), act[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / n)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[bc] epoch {epoch + 1}/{epochs} loss={losses[-1]:.5f}", flush=True)
    policy.eval()
    return policy, losses


def save_policy(policy: StandalonePolicy, out: str = POLICY_BC_PT) -> None:
    arch = [layer.out_features for layer in policy.body if isinstance(layer, nn.Linear)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "arch": arch,
            "obs_dim": spec.OBS_DIM,
            "act_dim": spec.ACTION_DIM,
            "train_hover_thrust": spec.HOVER_THRUST,
            "action_scale": [
                spec.MAX_ROLL_RATE,
                spec.MAX_PITCH_RATE,
                spec.MAX_YAW_RATE,
            ],
        },
        out,
    )
    print(f"[bc] saved BC policy -> {out}", flush=True)


def _make_synthetic_demos(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(42)
    obs = torch.from_numpy(rng.uniform(-1, 1, (n, spec.OBS_DIM)).astype(np.float32))
    w = torch.from_numpy(
        rng.uniform(-0.2, 0.2, (spec.OBS_DIM, spec.ACTION_DIM)).astype(np.float32)
    )
    act = torch.clamp(obs @ w, -1, 1)
    return obs, act


def run(cfg: RLConfig, *, smoke_n: int | None = None) -> StandalonePolicy:
    if smoke_n is not None:
        if smoke_n < 2:
            raise ValueError(f"smoke_n must be >= 2, got {smoke_n}")
        obs, act = _make_synthetic_demos(smoke_n)
        epochs = min(cfg.bc.epochs, 3)
        lr = cfg.bc.lr
        seed = cfg.seed
        arch = cfg.ppo.net_arch
        print(f"[bc] smoke mode: {smoke_n} synthetic transitions, {epochs} epochs")
        policy, losses = train_bc(
            obs,
            act,
            epochs=epochs,
            batch_size=min(32, smoke_n),
            lr=lr,
            seed=seed,
            arch=arch,
        )
        assert losses[-1] < losses[0], (
            f"smoke loss did not drop: {losses[0]}->{losses[-1]}"
        )
        print(f"[bc] smoke OK — loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        return policy

    obs, act = load_demos()
    print(f"[bc] {len(obs)} demo transitions from {DEMOS_PATH}", flush=True)
    policy, _ = train_bc(
        obs,
        act,
        epochs=cfg.bc.epochs,
        lr=cfg.bc.lr,
        seed=cfg.seed,
        arch=cfg.ppo.net_arch,
    )
    save_policy(policy)
    return policy


def _selftest():
    rng = np.random.default_rng(3)
    obs = torch.from_numpy(rng.uniform(-1, 1, (256, spec.OBS_DIM)).astype(np.float32))
    w = torch.from_numpy(
        rng.uniform(-0.2, 0.2, (spec.OBS_DIM, spec.ACTION_DIM)).astype(np.float32)
    )
    act = torch.clamp(obs @ w, -1, 1)
    policy, losses = train_bc(obs, act, epochs=15, batch_size=64)
    assert losses[-1] < 0.5 * losses[0], f"loss did not drop: {losses[0]}->{losses[-1]}"
    with torch.no_grad():
        out = policy(obs[:8])
    assert out.shape == (8, spec.ACTION_DIM)
    print(f"[selftest] OK — BC loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Config-driven BC pretraining runner.",
    )
    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config path (default: configs/default.yaml)",
    )
    ap.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Smoke mode: N synthetic transitions, bounded epochs",
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--epochs", type=int, default=None, help="Override config epochs (legacy)"
    )
    args = ap.parse_args()

    if args.selftest:
        _selftest()
    elif args.smoke is not None:
        cfg = load_config(args.config)
        if args.epochs is not None:
            cfg.bc.epochs = args.epochs
        run(cfg, smoke_n=args.smoke)
    else:
        cfg = load_config(args.config)
        if args.epochs is not None:
            cfg.bc.epochs = args.epochs
        run(cfg)
