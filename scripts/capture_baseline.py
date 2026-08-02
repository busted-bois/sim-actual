"""Capture the frozen pre-overhaul RL behavior contract."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np

from rl.core import spec
from rl.deploy import (
    LIVE_RATE_CLIP,
    LIVE_THRUST_MAX,
    LIVE_THRUST_MIN,
    live_scale_action,
    load_policy,
)
from rl.environment.env import CURRICULUM, GateRacingEnv
from rl.experts.fly2_course import HOVER_T as LIVE_HOVER_THRUST
from rl.experts.gp_expert import GPExpert
from rl.core.observation import DIST_SCALE, RATE_SCALE, V_SCALE, build_observation
from rl.training.train_ppo import POLICY_PT


DEFAULT_OUTPUT = ROOT / "rl" / "data" / "baseline.json"
DEFAULT_EVIDENCE = ROOT / ".sisyphus" / "evidence" / "task-0-baseline.json"
EVAL_STAGE = 2
EVAL_SEEDS = (0, 1, 2)
EPISODE_SEED_STRIDE = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_seed(seed: int, episode: int) -> int:
    return seed * EPISODE_SEED_STRIDE + episode


def _rollout(env: GateRacingEnv, action_fn, seed: int) -> dict[str, object]:
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    terminated = truncated = False
    info: dict[str, object] = {}
    while not (terminated or truncated):
        action = action_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward
    return {
        "seed": seed,
        "gates_cleared": int(env.gate_idx),
        "return": total_return,
        "steps": int(env.steps),
        "success": bool(info.get("course_complete", False)),
        "terminal_info": info,
    }


def _aggregate(episodes: list[dict[str, object]]) -> dict[str, object]:
    gates = np.asarray([episode["gates_cleared"] for episode in episodes], dtype=float)
    returns = np.asarray([episode["return"] for episode in episodes], dtype=float)
    successes = np.asarray([episode["success"] for episode in episodes], dtype=float)
    return {
        "episodes": len(episodes),
        "gates_cleared_mean": float(gates.mean()),
        "gates_cleared_std": float(gates.std()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "success_rate": float(successes.mean()),
    }


def _evaluate_policy(episodes_per_seed: int) -> dict[str, object]:
    act, metadata = load_policy(str(ROOT / POLICY_PT), device="cpu")
    episodes: list[dict[str, object]] = []
    by_seed: dict[str, object] = {}
    for seed in EVAL_SEEDS:
        seed_episodes = []
        env = GateRacingEnv(stage=EVAL_STAGE)
        for episode in range(episodes_per_seed):
            result = _rollout(
                env,
                lambda obs, _env: act(obs),
                _episode_seed(seed, episode),
            )
            seed_episodes.append(result)
            episodes.append(result)
        env.close()
        by_seed[str(seed)] = _aggregate(seed_episodes)
    return {"metadata": metadata, "summary": _aggregate(episodes), "by_seed": by_seed}


def _evaluate_expert(episodes_per_seed: int) -> dict[str, object]:
    expert = GPExpert()
    episodes: list[dict[str, object]] = []
    by_seed: dict[str, object] = {}
    for seed in EVAL_SEEDS:
        seed_episodes = []
        env = GateRacingEnv(stage=EVAL_STAGE)
        for episode in range(episodes_per_seed):
            expert.reset()
            result = _rollout(
                env,
                lambda _obs, state: expert.act(
                    state.p, state.v, state.q, state.gate_map, state.gate_idx
                ),
                _episode_seed(seed, episode),
            )
            seed_episodes.append(result)
            episodes.append(result)
        env.close()
        by_seed[str(seed)] = _aggregate(seed_episodes)
    return {"summary": _aggregate(episodes), "by_seed": by_seed}


def _golden_contract() -> dict[str, object]:
    gate_map = [
        {"pos": [5.0, 1.0, -2.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        {
            "pos": [11.0, -2.0, -3.5],
            "quat": [0.9800665778412416, 0.0, 0.0, 0.19866933079506122],
        },
    ]
    observation_input = {
        "position": [1.0, -0.5, -1.25],
        "velocity_world": [2.5, -1.0, 0.4],
        "quaternion": [0.9659258262890683, 0.0, 0.0, 0.25881904510252074],
        "angular_velocity": [0.12, -0.18, 0.24],
        "gate_map": gate_map,
        "gate_index": 0,
        "last_action": [0.3, -0.2, 0.1],
    }
    observation = build_observation(
        np.asarray(observation_input["position"]),
        np.asarray(observation_input["velocity_world"]),
        np.asarray(observation_input["quaternion"]),
        np.asarray(observation_input["angular_velocity"]),
        gate_map,
        0,
        np.asarray(observation_input["last_action"]),
    )
    normalized_action = np.array([-1.25, -0.5, 0.25, 1.2], dtype=np.float64)
    physical_action = spec.scale_action(normalized_action)
    deploy_input = np.array([0.75, -0.5, 0.4, 0.0], dtype=np.float64)
    deploy_meta = {"train_hover": 0.5, "action_scale": (4.0, 4.0, 3.0)}
    return {
        "observation_input": observation_input,
        "observation": observation.tolist(),
        "normalized_action_input": normalized_action.tolist(),
        "scaled_action": physical_action.tolist(),
        "unscaled_action": spec.unscale_action(physical_action).tolist(),
        "deploy_remap": {
            "input": deploy_input.tolist(),
            "metadata": deploy_meta,
            "output": live_scale_action(deploy_input, deploy_meta).tolist(),
        },
    }


def _golden_trajectory() -> dict[str, object]:
    env = GateRacingEnv(stage=EVAL_STAGE)
    obs, _ = env.reset(seed=42)
    action = np.array([0.05, -0.1, 0.02, -0.4], dtype=np.float32)
    steps = []
    for index in range(16):
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(
            {
                "index": index + 1,
                "position": env.p.tolist(),
                "velocity": env.v.tolist(),
                "quaternion": env.q.tolist(),
                "angular_velocity": env.omega.tolist(),
                "reward": reward,
                "gate_index": int(env.gate_idx),
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        if terminated or truncated:
            break
    env.close()
    return {
        "stage": EVAL_STAGE,
        "seed": 42,
        "action": action.tolist(),
        "steps": steps,
        "final_observation": obs.tolist(),
    }


def _test_baseline(test_log: Path, rl_test_log: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    if test_log.exists():
        text = test_log.read_text(errors="replace")
        modules: dict[str, dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "failed": 0}
        )
        pattern = re.compile(
            r"\(tests\.(test_[^.]+)\.[^)]+\) \.\.\. (ok|FAIL|ERROR)$", re.M
        )
        for module, status in pattern.findall(text):
            modules[module]["passed" if status == "ok" else "failed"] += 1
        ran = re.search(r"Ran (\d+) tests", text)
        failures = re.findall(r"^(?:FAIL|ERROR): ([^ ]+) \(([^)]+)\)$", text, re.M)
        result["make_test"] = {
            "log": str(test_log),
            "log_sha256": _sha256(test_log),
            "tests_run": int(ran.group(1)) if ran else None,
            "status": "failed" if "FAILED (" in text else "passed",
            "failures": [f"{case} ({qualified})" for case, qualified in failures],
            "by_module": dict(sorted(modules.items())),
        }
    else:
        result["make_test"] = {"log": str(test_log), "status": "not_recorded"}

    if rl_test_log.exists():
        text = rl_test_log.read_text(errors="replace")
        modules = re.findall(r"^uv run -m ([^ ]+) --selftest$", text, re.M)
        result["make_rl_test"] = {
            "log": str(rl_test_log),
            "log_sha256": _sha256(rl_test_log),
            "status": "failed" if "make: ***" in text else "passed",
            "modules": {module: "passed" for module in modules},
        }
    else:
        result["make_rl_test"] = {"log": str(rl_test_log), "status": "not_recorded"}
    return result


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def capture(
    episodes_per_seed: int, test_log: Path, rl_test_log: Path
) -> dict[str, object]:
    policy_path = ROOT / POLICY_PT
    excluded = {ROOT / ".venv", ROOT / ".git"}

    def tracked_files(pattern: str) -> list[str]:
        return sorted(
            str(path.relative_to(ROOT))
            for path in ROOT.rglob(pattern)
            if not any(parent in path.parents for parent in excluded)
        )

    constants = {
        "action_dim": spec.ACTION_DIM,
        "observation_dim": spec.OBS_DIM,
        "gate_size_m": spec.GATE_SIZE_M,
        "gravity_m_s2": spec.GRAVITY,
        "hover_thrust": spec.HOVER_THRUST,
        "max_rates_rad_s": [
            spec.MAX_ROLL_RATE,
            spec.MAX_PITCH_RATE,
            spec.MAX_YAW_RATE,
        ],
        "observation_layout": {
            name: [section.start, section.stop]
            for name, section in spec.OBS_LAYOUT.items()
        },
        "observation_scales": {
            "distance": DIST_SCALE,
            "velocity": V_SCALE,
            "rate": RATE_SCALE,
        },
        "curriculum": CURRICULUM,
        "live_deploy": {
            "hover_thrust": LIVE_HOVER_THRUST,
            "rate_clip": LIVE_RATE_CLIP,
            "thrust_min": LIVE_THRUST_MIN,
            "thrust_max": LIVE_THRUST_MAX,
        },
    }
    policy_evaluation = _evaluate_policy(episodes_per_seed)
    expert_evaluation = _evaluate_expert(episodes_per_seed)
    policy_summary = policy_evaluation["summary"]
    expert_summary = expert_evaluation["summary"]
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "policy_baseline_mean": policy_summary["gates_cleared_mean"],
        "policy_baseline_std": policy_summary["gates_cleared_std"],
        "expert_baseline_mean": expert_summary["gates_cleared_mean"],
        "expert_baseline_std": expert_summary["gates_cleared_std"],
        "gatenet_weights": "absent",
        "calibration_json": "absent_or_runtime",
        "evaluation": {
            "stage": EVAL_STAGE,
            "seeds": list(EVAL_SEEDS),
            "episodes_per_seed": episodes_per_seed,
            "episode_seed_formula": "seed * 10000 + episode_index",
            "policy": policy_evaluation,
            "expert": expert_evaluation,
        },
        "policy_artifact": {
            "path": str(policy_path.relative_to(ROOT)),
            "sha256": _sha256(policy_path),
            "size_bytes": policy_path.stat().st_size,
        },
        "runtime_artifacts": {
            "gatenet_weights": tracked_files("gatenet*.pt"),
            "calibration_files": tracked_files("calibration.json"),
        },
        "constants": constants,
        "golden_contract": _golden_contract(),
        "golden_trajectory": _golden_trajectory(),
        "tests": _test_baseline(test_log, rl_test_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--test-log", type=Path, default=Path("/tmp/baseline-test.log"))
    parser.add_argument(
        "--rl-test-log", type=Path, default=Path("/tmp/baseline-rltest.log")
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    baseline = capture(args.episodes, args.test_log, args.rl_test_log)
    _atomic_json(args.output, baseline)
    _atomic_json(args.evidence, baseline)
    print(f"baseline: {args.output}")
    print(f"evidence: {args.evidence}")


if __name__ == "__main__":
    main()
