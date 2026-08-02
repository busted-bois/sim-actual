"""CLI entry: ``uv run python -m rl.algorithms --selftest``.

The self-test is fast and artifact-free: it proves registry identities,
Protocol conformance (runtime-checkable), and the exact SAC error message.
It performs no real training and writes no files.
"""

from __future__ import annotations

import argparse
import sys

from rl.algorithms import ALGORITHMS, Algorithm, BC, PPO, SAC, get_algorithm

_SAC_MSG = "SAC not implemented; interface slot for future. Use PPO."


def _selftest() -> int:
    assert get_algorithm("ppo") is PPO, "registry: get_algorithm('ppo') must be PPO"
    assert get_algorithm("bc") is BC, "registry: get_algorithm('bc') must be BC"
    assert get_algorithm("sac") is SAC, "registry: get_algorithm('sac') must be SAC"
    assert ALGORITHMS == {"ppo": PPO, "bc": BC, "sac": SAC}, (
        f"registry literal mismatch: {ALGORITHMS}"
    )

    for name, cls in ALGORITHMS.items():
        instance = cls()
        assert isinstance(instance, Algorithm), (
            f"{name} ({cls.__name__}) does not conform to Algorithm Protocol"
        )

    sac = SAC()
    for method_name, args in (
        ("train", (None, None)),
        ("load", ("dummy",)),
        ("save", (None, "dummy")),
    ):
        try:
            getattr(sac, method_name)(*args)
        except NotImplementedError as exc:
            assert str(exc) == _SAC_MSG, (
                f"SAC.{method_name} message mismatch: {str(exc)!r}"
            )
        else:
            raise AssertionError(f"SAC.{method_name} did not raise NotImplementedError")

    try:
        get_algorithm("unknown")
    except ValueError as exc:
        for valid_name in ALGORITHMS:
            assert valid_name in str(exc), (
                f"unknown-name error missing {valid_name}: {exc}"
            )
    else:
        raise AssertionError("get_algorithm('unknown') did not raise ValueError")

    print("[selftest] OK — registry identities, Protocol conformance, SAC stub")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.algorithms",
        description="Swappable algorithm interface — registry + self-test.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run artifact-free self-test (registry, Protocol, SAC stub).",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
