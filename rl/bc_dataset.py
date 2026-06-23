"""Behavior cloning from fly2 --log JSONL demos.

Each JSONL line (from ``uv run -m rl.fly2 --mode course --log path.jsonl``)
contains ``obs`` (24-D), ``v_des_ned`` (3,), and ``yaw_rate_des`` (scalar).
We encode those setpoints with ``spec.encode_velocity_action`` and train a
StandalonePolicy (same architecture as PPO export) via supervised MSE.

    uv run -m rl.bc_dataset --demo rl/data/demo.jsonl
    uv run -m rl.bc_dataset --demo rl/data/demo.jsonl --epochs 40 --out rl/data/policy.pt
    uv run -m rl.bc_dataset --selftest
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rl import spec
from rl.policy import NET_ARCH, POLICY_PT, StandalonePolicy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DEMO = os.path.join(DATA_DIR, "demo.jsonl")


class DemoDataset(Dataset):
    def __init__(self, path: str):
        self.obs = []
        self.act = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = np.asarray(row["obs"], dtype=np.float32)
                if obs.shape != (spec.OBS_DIM,):
                    continue
                v_des = np.asarray(row["v_des_ned"], dtype=float)
                yaw_rd = float(row["yaw_rate_des"])
                act = spec.encode_velocity_action(v_des, yaw_rd)
                self.obs.append(obs)
                self.act.append(act)
        if not self.obs:
            raise ValueError(f"no valid rows in {path}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


def load_demo_pairs(path: str) -> tuple[np.ndarray, np.ndarray]:
    ds = DemoDataset(path)
    obs = np.stack(ds.obs)
    act = np.stack(ds.act)
    return obs, act


def train_bc(
    demo_path: str = DEFAULT_DEMO,
    out_path: str = POLICY_PT,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
) -> StandalonePolicy:
    ds = DemoDataset(demo_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dev = torch.device(device)
    net = StandalonePolicy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(
        f"[bc] training on {len(ds)} samples from {demo_path} "
        f"({epochs} epochs, batch={batch_size})",
        flush=True,
    )
    net.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for obs_b, act_b in loader:
            obs_t = obs_b.to(dev)
            act_t = act_b.to(dev)
            pred = net(obs_t)
            loss = loss_fn(pred, act_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(obs_b)
            n += len(obs_b)
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"[bc] epoch {ep + 1}/{epochs} mse={total / n:.5f}", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    net.eval()
    torch.save(
        {
            "state_dict": net.state_dict(),
            "arch": NET_ARCH,
            "obs_dim": spec.OBS_DIM,
            "act_dim": spec.ACTION_DIM,
            "action_space": "velocity",
            "demo_path": demo_path,
        },
        out_path,
    )
    print(f"[bc] saved -> {out_path}", flush=True)
    return net


def _selftest():
    import tempfile

    obs = np.random.randn(spec.OBS_DIM).astype(np.float32)
    v_des = np.array([2.0, 0.5, -0.3])
    yaw_rd = 0.4
    act = spec.encode_velocity_action(v_des, yaw_rd)
    row = {
        "t": 0.0,
        "active_gate": 0,
        "obs": obs.tolist(),
        "v_des_ned": v_des.tolist(),
        "yaw_rate_des": yaw_rd,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(row) + "\n")
        path = f.name
    try:
        o, a = load_demo_pairs(path)
        assert o.shape == (1, spec.OBS_DIM)
        assert a.shape == (1, spec.ACTION_DIM)
        assert np.allclose(a[0], act, atol=1e-5)
        print("[bc] load_demo_pairs OK", flush=True)
    finally:
        os.unlink(path)
    print("[bc] selftest OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=DEFAULT_DEMO, help="fly2 --log JSONL path")
    ap.add_argument("--out", default=POLICY_PT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        train_bc(args.demo, args.out, epochs=args.epochs, batch_size=args.batch)
