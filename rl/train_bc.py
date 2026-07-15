"""BC pretrain — clone the GP expert demos into the PPO actor network.

MSE-regresses the StandalonePolicy (same 3x64 tanh MLP the PPO actor uses)
onto rl/data/gp_demos.npz (rl.log_demos), saving rl/data/policy_bc.pt in the
same dict schema as policy.pt so rl.train_ppo can warm-start from it and
rl.deploy could fly it directly.

    uv run -m rl.train_bc               # full pretrain
    uv run -m rl.train_bc --selftest    # tiny synthetic run, asserts loss drop
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from rl import spec
from rl.train_ppo import NET_ARCH, StandalonePolicy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEMOS_PATH = os.path.join(DATA_DIR, "gp_demos.npz")
POLICY_BC_PT = os.path.join(DATA_DIR, "policy_bc.pt")

EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3


def load_demos(path: str = DEMOS_PATH) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as d:
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
) -> tuple[StandalonePolicy, list[float]]:
    torch.manual_seed(seed)
    policy = StandalonePolicy()
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
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "arch": NET_ARCH,
            "obs_dim": spec.OBS_DIM,
            "act_dim": spec.ACTION_DIM,
        },
        out,
    )
    print(f"[bc] saved BC policy -> {out}", flush=True)


def _selftest():
    # Tiny synthetic linear mapping: loss must clearly drop.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        obs, act = load_demos()
        print(f"[bc] {len(obs)} demo transitions from {DEMOS_PATH}", flush=True)
        policy, _ = train_bc(obs, act, epochs=args.epochs)
        save_policy(policy)
