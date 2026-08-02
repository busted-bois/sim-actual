"""SAC interface stub — NOT implemented.

Every method raises ``NotImplementedError`` with the exact contract message.
No SAC implementation is imported; this module exists only to reserve the
registry slot for a future adapter.
"""

from __future__ import annotations

from typing import Any

_MSG = "SAC not implemented; interface slot for future. Use PPO."


class SAC:
    """Interface stub. All three methods raise ``NotImplementedError``."""

    def train(self, env: Any, cfg: Any) -> Any:
        raise NotImplementedError(_MSG)

    def load(self, path: str) -> Any:
        raise NotImplementedError(_MSG)

    def save(self, policy: Any, path: str) -> None:
        raise NotImplementedError(_MSG)
