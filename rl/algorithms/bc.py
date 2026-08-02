"""BC adapter wrapping :func:`rl.training.train_bc.run`.

Thin adapter: ``train`` delegates to the config-driven runner, ``load``/``save``
use the existing standalone ``.pt`` schema. The loader validates metadata
dimensions and architecture before constructing the policy.
"""

from __future__ import annotations

from typing import Any

import torch

from rl.core import spec
from rl.core.config import RLConfig
from rl.training.train_bc import run as _run_bc
from rl.training.train_bc import save_policy as _save_policy
from rl.training.train_ppo import StandalonePolicy


class BC:
    """Adapter over :func:`rl.training.train_bc.run` (standalone ``.pt`` format)."""

    def train(self, env: Any, cfg: RLConfig) -> StandalonePolicy:
        if env is not None:
            raise ValueError(
                "BC.train does not accept a prebuilt env; the runner loads demos "
                "from its standard data path. Pass env=None."
            )
        return _run_bc(cfg)

    def load(self, path: str) -> StandalonePolicy:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"BC checkpoint at {path} is not a dict (got {type(ckpt).__name__})"
            )

        obs_dim = ckpt.get("obs_dim")
        if (
            not isinstance(obs_dim, int)
            or isinstance(obs_dim, bool)
            or obs_dim != spec.OBS_DIM
        ):
            raise ValueError(
                f"obs_dim mismatch: checkpoint={obs_dim!r}, build={spec.OBS_DIM}"
            )

        act_dim = ckpt.get("act_dim")
        if (
            not isinstance(act_dim, int)
            or isinstance(act_dim, bool)
            or act_dim != spec.ACTION_DIM
        ):
            raise ValueError(
                f"act_dim mismatch: checkpoint={act_dim!r}, build={spec.ACTION_DIM}"
            )

        arch = ckpt.get("arch")
        if not isinstance(arch, list) or not arch:
            raise ValueError(f"arch must be a nonempty list, got {arch!r}")
        for h in arch:
            if not isinstance(h, int) or isinstance(h, bool) or h <= 0:
                raise ValueError(
                    f"arch entries must be positive ints, got {h!r} in {arch}"
                )

        state_dict = ckpt.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"state_dict missing or empty in {path}")

        policy = StandalonePolicy(arch=arch, obs_dim=obs_dim, act_dim=act_dim)
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy

    def save(self, policy: StandalonePolicy, path: str) -> None:
        _save_policy(policy, path)
