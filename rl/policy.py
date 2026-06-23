"""Shared policy network (BC + PPO export + deploy)."""

from __future__ import annotations

import os

import torch.nn as nn

from rl import spec

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
POLICY_PT = os.path.join(DATA_DIR, "policy.pt")
NET_ARCH = [64, 64, 64]


class StandalonePolicy(nn.Module):
    """Pure-torch deterministic actor: obs(24) -> 3x64 tanh -> action(4)."""

    def __init__(self, obs_dim=spec.OBS_DIM, act_dim=spec.ACTION_DIM, arch=NET_ARCH):
        super().__init__()
        layers, last = [], obs_dim
        for h in arch:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, act_dim)

    def forward(self, x):
        return self.head(self.body(x))
