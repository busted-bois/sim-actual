"""Roll out the GP expert in GateRacingEnv and log (obs, action) demos.

Runs the AndurilGP guidance expert (rl.gp_expert) across all curriculum
stages and saves the 24-D observations + normalized 4-D actions to
rl/data/gp_demos.npz for BC pretraining (rl.train_bc). Episodes that never
pass a gate are dropped, and each kept episode is truncated at its final
gate pass so crash tails don't poison the dataset.

    uv run -m rl.log_demos              # full demo set
    uv run -m rl.log_demos --selftest   # tiny run, asserts file + shapes
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rl import spec
from rl.env import CURRICULUM, GateRacingEnv
from rl.gp_expert import GPExpert

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEMOS_PATH = os.path.join(DATA_DIR, "gp_demos.npz")

EPISODES_PER_STAGE = 40


def _episode(env: GateRacingEnv, expert: GPExpert) -> tuple[np.ndarray, np.ndarray]:
    """One rollout. Returns (obs, act) arrays truncated at the last gate pass
    (empty arrays when no gate was passed)."""
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    last_pass = -1

    obs, _ = env.reset()
    expert.reset()
    term = trunc = False
    while not (term or trunc):
        a = expert.act(env.p, env.v, env.q, env.gate_map, env.gate_idx)
        obs_buf.append(np.asarray(obs, np.float32))
        act_buf.append(np.asarray(a, np.float32))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("gate_passed"):
            last_pass = len(act_buf) - 1

    if last_pass < 0:
        return np.zeros((0, spec.OBS_DIM), np.float32), np.zeros(
            (0, spec.ACTION_DIM), np.float32
        )
    n = last_pass + 1
    return np.stack(obs_buf[:n]), np.stack(act_buf[:n])


def collect(episodes_per_stage: int = EPISODES_PER_STAGE, seed: int = 0) -> dict:
    expert = GPExpert()
    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    kept = total = 0
    for stage in range(len(CURRICULUM)):
        for ep in range(episodes_per_stage):
            env = GateRacingEnv(stage=stage, seed=seed + stage * 1000 + ep)
            obs, act = _episode(env, expert)
            total += 1
            if len(obs):
                kept += 1
                all_obs.append(obs)
                all_act.append(act)
        print(
            f"[demos] stage {stage}: {kept}/{total} episodes kept so far",
            flush=True,
        )
    obs = (
        np.concatenate(all_obs) if all_obs else np.zeros((0, spec.OBS_DIM), np.float32)
    )
    act = (
        np.concatenate(all_act)
        if all_act
        else np.zeros((0, spec.ACTION_DIM), np.float32)
    )
    return {"obs": obs, "act": act}


def save(demos: dict, path: str = DEMOS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        obs=demos["obs"],
        act=demos["act"],
        # Plant stamp: actions/obs are normalized by these scales. rl.train_bc
        # refuses demos from a different build so a caps change can't silently
        # clone the wrong normalization into the BC policy.
        action_scale=np.array(
            [spec.MAX_ROLL_RATE, spec.MAX_PITCH_RATE, spec.MAX_YAW_RATE]
        ),
        hover_thrust=np.array(spec.HOVER_THRUST),
    )
    print(
        f"[demos] saved {len(demos['obs'])} transitions -> {path}",
        flush=True,
    )


def _selftest():
    import tempfile

    demos = collect(episodes_per_stage=2, seed=7)
    assert demos["obs"].shape[1] == spec.OBS_DIM
    assert demos["act"].shape[1] == spec.ACTION_DIM
    assert len(demos["obs"]) == len(demos["act"]) > 0
    assert np.all(np.abs(demos["act"]) <= 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gp_demos.npz")
        save(demos, path)
        with np.load(path) as loaded:
            assert loaded["obs"].shape == demos["obs"].shape
    print(f"[selftest] OK — {len(demos['obs'])} demo transitions, shapes + bounds sane")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        save(collect())
