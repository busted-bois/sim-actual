"""Behaviour-clone a PPO policy onto the GP-pilot demos, so real-sim PPO starts
from a flying policy instead of noise.

Builds an SB3 PPO model (same [64,64,64] MLP the trainer uses), then supervised-
fits its action mean to the logged expert actions (MSE). Saves an SB3 zip that
`rl.vq2.train --resume` continues with real-sim PPO. No sim needed here.

    uv run -m rl.vq2.train_bc                       # -> rl/data/vq2/policy_bc.zip
    uv run -m rl.vq2.train --resume rl/data/vq2/policy_bc.zip
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from rl import spec
from rl.vq2.observation import POLICY_OBS_DIM

DATA = os.path.join("rl", "data", "vq2")


class _SpacesEnv(gym.Env):
    """No-sim stub so PPO can build a policy with the right spaces for BC."""

    def __init__(self):
        self.observation_space = spaces.Box(-10.0, 10.0, (POLICY_OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (spec.ACTION_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(POLICY_OBS_DIM, np.float32), {}

    def step(self, action):
        return np.zeros(POLICY_OBS_DIM, np.float32), 0.0, True, False, {}


def train_bc(demos: str = os.path.join(DATA, "demos.npz"),
             epochs: int = 40, batch: int = 512, lr: float = 1e-3,
             out: str = os.path.join(DATA, "policy_bc")) -> None:
    d = np.load(demos)
    obs = d["obs"].astype(np.float32)
    act = d["act"].astype(np.float32)
    assert obs.shape[1] == POLICY_OBS_DIM, (obs.shape, POLICY_OBS_DIM)
    assert act.shape[1] == spec.ACTION_DIM, act.shape
    print(f"[vq2.train_bc] {len(obs)} samples  obs{obs.shape} act{act.shape}", flush=True)

    model = PPO("MlpPolicy", _SpacesEnv(),
                policy_kwargs=dict(net_arch=[64, 64, 64]), device="cpu", verbose=0)
    pol = model.policy
    dev = pol.device
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs, device=dev)
    act_t = torch.as_tensor(act, device=dev)
    n = len(obs)

    for ep in range(epochs):
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        nb = 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            mean = pol.get_distribution(obs_t[b]).distribution.mean  # action mean
            loss = torch.nn.functional.mse_loss(mean, act_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:2d}  mse {tot / max(1, nb):.4f}", flush=True)

    model.save(out)
    print(f"[vq2.train_bc] saved {out}.zip", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=str, default=os.path.join(DATA, "demos.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_bc(args.demos, args.epochs, args.batch, args.lr)
