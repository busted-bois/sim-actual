"""Run the complete offline RL training and evaluation pipeline.

Usage:
    uv run scripts/train_rl.py
    uv run scripts/train_rl.py --dry-run
    uv run scripts/train_rl.py --run-name gpu-ppo --steps 300000 --envs 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_SEEDS = (0, 1, 2)
BC_POLICY = Path("rl/data/policy_bc.pt")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _seeds(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated non-negative integers"
        ) from exc
    if (
        not parsed
        or any(seed < 0 for seed in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError(
            "must be unique comma-separated non-negative integers"
        )
    return parsed


def _run_name(value: str) -> str:
    if not value or value.strip() != value:
        raise argparse.ArgumentTypeError("must be non-empty without surrounding spaces")
    if any(char in value for char in ("/", "\\", ".")) or any(
        char.isspace() for char in value
    ):
        raise argparse.ArgumentTypeError(
            "must not contain path separators, dots, or whitespace"
        )
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demos, train BC and PPO, then evaluate both policy and expert."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", type=_run_name, default="ppo")
    parser.add_argument("--steps", type=_positive_int)
    parser.add_argument("--envs", type=_positive_int)
    parser.add_argument("--episodes", type=_positive_int, default=50)
    parser.add_argument("--seeds", type=_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _display(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run(command: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(_display(command), flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except OSError as exc:
        raise SystemExit(f"[train-rl] could not start {command[0]}: {exc}") from exc
    if result.returncode:
        raise SystemExit(result.returncode)


def _module(name: str, *args: str) -> list[str]:
    return [sys.executable, "-m", name, *args]


def _pipeline(args: argparse.Namespace, checkpoint_dir: Path):
    run_dir = checkpoint_dir / args.run_name
    candidate = run_dir / "policy.pt"
    candidate_csv = run_dir / "eval.csv"
    expert_csv = run_dir / "expert-eval.csv"
    seed_args = ["--seeds", *(str(seed) for seed in args.seeds)]

    ppo_args = [
        "--config",
        args.config,
        "--run-name",
        args.run_name,
        "--bc-init",
        str(BC_POLICY),
    ]
    if args.steps is not None:
        ppo_args.extend(("--steps", str(args.steps)))
    if args.envs is not None:
        ppo_args.extend(("--envs", str(args.envs)))

    commands = [
        ("1/6 Generate GP demos", _module("rl.training.log_demos")),
        (
            "2/6 Train behavior cloning",
            _module("rl.training.train_bc", "--config", args.config),
        ),
        ("3/6 Train curriculum PPO", _module("rl.training.train_ppo", *ppo_args)),
        (
            "5/6 Evaluate candidate",
            _module(
                "rl.training.evaluate",
                "--policy",
                str(candidate),
                "--config",
                args.config,
                "--episodes",
                str(args.episodes),
                *seed_args,
                "--out-csv",
                str(candidate_csv),
            ),
        ),
        (
            "6/6 Evaluate GP expert",
            _module(
                "rl.training.evaluate",
                "--expert",
                "gp_expert",
                "--config",
                args.config,
                "--episodes",
                str(args.episodes),
                *seed_args,
                "--out-csv",
                str(expert_csv),
            ),
        ),
    ]
    return commands, candidate, candidate_csv, expert_csv


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from rl.core.config import load_config

    config = load_config(args.config)
    checkpoint_dir = Path(config.checkpoint.dir)
    commands, candidate, candidate_csv, expert_csv = _pipeline(args, checkpoint_dir)

    if args.dry_run:
        print("train-rl [DRY RUN]")
        print(
            f"config={args.config} run={args.run_name} checkpoint.dir={checkpoint_dir}"
        )
        for label, command in commands[:3]:
            print(f"\n=== {label} ===\n{_display(command)}")
        print(f"\n=== 4/6 Verify candidate ===\nrequire exists: {candidate}")
        for label, command in commands[3:]:
            print(f"\n=== {label} ===\n{_display(command)}")
    else:
        for label, command in commands[:3]:
            _run(command, label)
        print("\n=== 4/6 Verify candidate ===", flush=True)
        if not candidate.is_file():
            raise SystemExit(f"[train-rl] expected candidate not found: {candidate}")
        print(f"exists: {candidate}", flush=True)
        for label, command in commands[3:]:
            _run(command, label)

    print("\n=== Pipeline complete ===")
    print(f"candidate:  {candidate}")
    print(f"eval csv:   {candidate_csv}")
    print(f"expert csv: {expert_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
