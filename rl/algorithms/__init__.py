"""Swappable algorithm interface and registry.

Public adapters:

- :class:`PPO`  — :mod:`rl.algorithms.ppo`
- :class:`BC`   — :mod:`rl.algorithms.bc`
- :class:`SAC`  — :mod:`rl.algorithms.sac` (stub; raises ``NotImplementedError``)

    from rl.algorithms import get_algorithm
    algo = get_algorithm("ppo")
    policy = algo().train(env=None, cfg=cfg)
"""

from rl.algorithms.base import Algorithm
from rl.algorithms.bc import BC
from rl.algorithms.ppo import PPO
from rl.algorithms.sac import SAC

__all__ = ["Algorithm", "ALGORITHMS", "PPO", "BC", "SAC", "get_algorithm"]

ALGORITHMS: dict[str, type] = {
    "ppo": PPO,
    "bc": BC,
    "sac": SAC,
}


def get_algorithm(name: str) -> type:
    """Return the algorithm class registered under exact lowercase ``name``.

    Raises ``ValueError`` listing valid names if ``name`` is unknown, and
    ``TypeError`` if ``name`` is not a string.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__}")
    try:
        return ALGORITHMS[name]
    except KeyError:
        valid = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unknown algorithm {name!r}; valid names: {valid}") from None
