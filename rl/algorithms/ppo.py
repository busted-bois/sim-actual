"""PPO adapter wrapping :func:`rl.training.train_ppo.run`.

Thin adapter: ``train`` delegates to the config-driven runner, ``load`` uses
native SB3 archive loading, ``save`` uses the existing atomic SB3 saver.
No new serialization format is introduced.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO as _SB3PPO

from rl.core.config import RLConfig
from rl.training.checkpoint import save_sb3_atomic
from rl.training.train_ppo import run as _run_ppo


class PPO:
    """Adapter over :func:`rl.training.train_ppo.run` (native SB3 PPO format)."""

    def train(self, env: Any, cfg: RLConfig) -> _SB3PPO:
        if env is not None:
            raise ValueError(
                "PPO.train does not accept a prebuilt env; the runner constructs "
                "vectorised envs from cfg. Pass env=None and set curriculum/seed "
                "fields on cfg instead."
            )
        return _run_ppo(cfg)

    def load(self, path: str) -> _SB3PPO:
        return _SB3PPO.load(path, force_reset=True)

    def save(self, policy: _SB3PPO, path: str) -> None:
        save_sb3_atomic(policy, path)
