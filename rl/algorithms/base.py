"""Minimal swappable algorithm interface.

Exactly three public methods: ``train``, ``load``, ``save``. No extra
abstraction methods/properties are added on top of this Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Algorithm(Protocol):
    """Three-method interface for swappable RL algorithms.

    Adapters wrap existing training runners and native policy
    savers/loaders in :mod:`rl.training` — they introduce no new
    serialization format.

    Implementations:

    - :class:`rl.algorithms.ppo.PPO` — SB3 PPO via :mod:`rl.training.train_ppo`
    - :class:`rl.algorithms.bc.BC` — behavior cloning via :mod:`rl.training.train_bc`
    - :class:`rl.algorithms.sac.SAC` — interface stub (raises ``NotImplementedError``)
    """

    def train(self, env: Any, cfg: Any) -> Any:
        """Train and return a policy.

        ``env`` is accepted by the interface but the PPO/BC runners build
        their own vectorised environments from ``cfg``; pass ``env=None``.
        Passing a non-``None`` ``env`` is rejected with ``ValueError``.
        """
        ...

    def load(self, path: str) -> Any:
        """Load a policy from ``path`` using the algorithm's native format."""
        ...

    def save(self, policy: Any, path: str) -> None:
        """Save ``policy`` to ``path`` in the algorithm's native format."""
        ...
