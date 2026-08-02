"""T11 — deterministic multi-seed evaluation harness.

Reproduces the frozen T0 baseline protocol exactly so a retrained policy can
be compared against the checked-in anchor without ambiguity:

  * raw ``GateRacingEnv`` (no Monitor / VecEnv) — auto-reset ambiguity would
    drift the captured gates/return away from the baseline.
  * final 17-gate stage / default config, base seeds ``[0, 1, 2]``, 50 episodes
    each.
  * per-episode reset seed ``= base_seed * 10000 + episode_index`` (the formula
    recorded in ``rl/data/baseline.json``).
  * deterministic actions, terminate on ``terminated or truncated``.
  * ``gates_cleared`` read from ``info`` with a safe fallback to
    ``env.gate_idx``; success means clearing every gate in ``env.gate_map``.
  * population std (``np.std(..., ddof=0)``) to match SB3 / the baseline.

Standalone ``.pt`` artifacts reuse :func:`rl.deploy.load_policy`; expert mode
reuses :class:`rl.experts.gp_expert.GPExpert`.

    uv run -m rl.training.evaluate --policy rl/data/policy.pt
    uv run -m rl.training.evaluate --expert gp_expert --episodes 50
    uv run -m rl.training.evaluate --policy rl/data/policy.pt \
        --episodes 50 --seeds 0 1 2 --out-csv /tmp/eval-post.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from rl.core.config import EnvConfig, RLConfig
from rl.environment.env import CURRICULUM, DECISION_HZ, GateRacingEnv

# Frozen protocol constants — mirror rl/data/baseline.json["evaluation"].
SUPPORTED_EXPERTS = frozenset({"gp_expert"})
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EPISODES = 50
DEFAULT_STAGE = 3
DEFAULT_MAX_STEPS = 3500

# Exact CSV schema (order matters — checked-in baseline + tests assert it).
CSV_COLUMNS = ["seed", "episode", "gates_cleared", "return", "success", "steps"]


def _resolve_env_settings(env_cfg):
    """Return ``(stage, max_steps)`` from a typed config or the frozen default.

    ``EnvConfig`` and ``RLConfig`` are the only accepted typed carriers — no
    duck-typed dicts, so a malformed mapping cannot silently change the
    evaluation plant.
    """
    if env_cfg is None:
        stage, max_steps = DEFAULT_STAGE, DEFAULT_MAX_STEPS
    elif isinstance(env_cfg, RLConfig):
        stage, max_steps = env_cfg.env.curriculum_stage, env_cfg.env.max_steps
    elif isinstance(env_cfg, EnvConfig):
        stage, max_steps = env_cfg.curriculum_stage, env_cfg.max_steps
    else:
        raise ValueError(
            "env_cfg must be EnvConfig, RLConfig, or None — "
            f"got {type(env_cfg).__name__}"
        )

    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ValueError(f"curriculum_stage must be an integer, got {stage!r}")
    if not 0 <= stage < len(CURRICULUM):
        raise ValueError(
            f"curriculum_stage must be in [0, {len(CURRICULUM) - 1}], got {stage}"
        )
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    return stage, max_steps


def _validate(policy_path, episodes, seeds, expert):
    """Validate inputs and return seeds as a validated ``list[int]``."""
    if policy_path is None and expert is None:
        raise ValueError("either policy_path or expert must be provided")
    if policy_path is not None and expert is not None:
        raise ValueError("policy_path and expert are mutually exclusive")
    # bool is a subclass of int — reject it explicitly so True/False are not
    # silently treated as seed 1/0.
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError(f"episodes must be a positive int, got {episodes!r}")
    seeds_list = list(seeds)
    if not seeds_list:
        raise ValueError("seeds must not be empty")
    if len(set(seeds_list)) != len(seeds_list):
        raise ValueError(f"seeds must be unique, got {seeds_list}")
    for value in seeds_list:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"seeds must be integers, got {value!r}")
        if value < 0:
            raise ValueError(f"seeds must be non-negative, got {value}")
    if expert is not None and expert not in SUPPORTED_EXPERTS:
        raise ValueError(
            f"unsupported expert {expert!r}; supported: {sorted(SUPPORTED_EXPERTS)}"
        )
    if policy_path is not None and not os.path.exists(policy_path):
        raise ValueError(f"policy not found: {policy_path}")
    return seeds_list


def _default_out_csv(policy_path, expert):
    """Default CSV location: ``rl/data/best/<stem>/eval.csv``."""
    if expert is not None:
        name = expert
    else:
        name = Path(policy_path).stem
    return f"rl/data/best/{name}/eval.csv"


def _write_csv_atomic(path, rows):
    """Write CSV to a same-dir temp file then ``os.replace`` for atomicity."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out.name}.",
        suffix=".tmp",
        dir=str(out.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)
        os.replace(tmp_name, out)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def evaluate(
    policy_path=None,
    episodes=DEFAULT_EPISODES,
    seeds=DEFAULT_SEEDS,
    env_cfg=None,
    out_csv=None,
    *,
    expert=None,
):
    """Run a deterministic multi-seed evaluation and write one CSV row per episode.

    Exactly one of ``policy_path`` (a standalone ``.pt`` loaded via
    :func:`rl.deploy.load_policy`) or ``expert`` (a name in
    :data:`SUPPORTED_EXPERTS`) must be provided.

    Parameters mirror the frozen T0 baseline: ``seeds`` default to
    ``[0, 1, 2]``, ``episodes`` to 50, and stage to the 17-gate final course
    (or whatever ``env_cfg`` carries). Each episode resets the env with seed
    ``base_seed * 10000 + episode_index``; the expert is reset every episode.

    Returns a summary dict with population mean/std (``ddof=0``) of gates
    cleared and episodic return, plus the overall success rate.
    """
    seeds_list = _validate(policy_path, episodes, seeds, expert)
    stage, max_steps = _resolve_env_settings(env_cfg)
    max_seconds = max_steps / DECISION_HZ

    if expert is not None:
        from rl.experts.gp_expert import GPExpert

        action_kind = "expert"
        action_src = GPExpert()
    else:
        from rl.deploy import load_policy

        action_kind = "policy"
        action_src = load_policy(policy_path)[0]

    rows = []
    gates_all = []
    return_all = []
    success_all = []

    for base_seed in seeds_list:
        # One env per seed (frozen baseline protocol) — episode seeds are
        # pushed through reset() so the layout is fully reproducible.
        env = GateRacingEnv(stage=stage, max_seconds=max_seconds)
        for ep_idx in range(episodes):
            ep_seed = base_seed * 10000 + ep_idx
            obs, _ = env.reset(seed=ep_seed)
            num_gates = len(env.gate_map)
            if action_kind == "expert":
                action_src.reset()
            total_return = 0.0
            steps = 0
            info = {}
            terminated = truncated = False
            while not (terminated or truncated):
                if action_kind == "expert":
                    action = action_src.act(
                        env.p, env.v, env.q, env.gate_map, env.gate_idx
                    )
                else:
                    action = action_src(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += reward
                steps += 1
            gates = int(info.get("gates_cleared", env.gate_idx))
            success = 1 if gates >= num_gates else 0
            rows.append((base_seed, ep_idx, gates, total_return, success, steps))
            gates_all.append(gates)
            return_all.append(total_return)
            success_all.append(success)

    gates_arr = np.asarray(gates_all, dtype=np.float64)
    return_arr = np.asarray(return_all, dtype=np.float64)
    success_arr = np.asarray(success_all, dtype=np.float64)
    total_episodes = len(rows)
    summary = {
        "episodes": total_episodes,
        "seeds": list(seeds_list),
        "gates_cleared_mean": float(np.mean(gates_arr)),
        "gates_cleared_std": float(np.std(gates_arr, ddof=0)),
        "return_mean": float(np.mean(return_arr)),
        "return_std": float(np.std(return_arr, ddof=0)),
        "success_rate": float(np.mean(success_arr)),
    }

    if out_csv is None:
        out_csv = _default_out_csv(policy_path, expert)
    _write_csv_atomic(out_csv, rows)
    summary["out_csv"] = str(out_csv)

    print(
        f"[evaluate] {total_episodes} episodes over seeds {list(seeds_list)}: "
        f"gates {summary['gates_cleared_mean']:.4f} +/- "
        f"{summary['gates_cleared_std']:.4f}, "
        f"return {summary['return_mean']:.4f} +/- {summary['return_std']:.4f}, "
        f"success {summary['success_rate']:.4f}",
        flush=True,
    )
    print(f"[evaluate] csv -> {out_csv}", flush=True)
    return summary


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rl.training.evaluate",
        description=(
            "Deterministic multi-seed evaluation of a policy.pt or named "
            "expert against the frozen T0 baseline protocol."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--policy", metavar="PATH", help="standalone policy .pt path")
    mode.add_argument(
        "--expert",
        choices=sorted(SUPPORTED_EXPERTS),
        help="named expert controller",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--config",
        help="path to RLConfig YAML (provides curriculum stage + max_steps)",
    )
    parser.add_argument(
        "--out-csv",
        dest="out_csv",
        help="CSV output path (default rl/data/best/<stem>/eval.csv)",
    )
    args = parser.parse_args(argv)

    env_cfg = None
    if args.config is not None:
        from rl.core.config import load_config

        env_cfg = load_config(args.config)

    try:
        summary = evaluate(
            policy_path=args.policy,
            episodes=args.episodes,
            seeds=args.seeds,
            env_cfg=env_cfg,
            out_csv=args.out_csv,
            expert=args.expert,
        )
    except ValueError as exc:
        print(f"[evaluate] error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    return summary


if __name__ == "__main__":
    _main()
