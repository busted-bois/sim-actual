"""Per-gate successful-trajectory store -- the heart of the incremental
closed-loop learning pipeline.

The policy flies CLOSED-LOOP in the real sim (VQ2RealEnv). The instant it passes
a gate it hasn't reached before this episode, the segment from the START to that
gate is saved PERMANENTLY under rl/data/vq2/success/gate{N}/. Training then
BC-initializes the policy from the UNION of all saved segments, so it keeps
building on what worked. Segments are NEVER auto-deleted -- the user prunes by
hand (see rl2-list-demos), and the next rl2-train re-reads whatever remains.

Preference when building the BC set (better segments oversampled):
  + more gates reached   + no collisions   (smoothness recorded for inspection)

    uv run -m rl.vq2.success --list      # counts per gate
    uv run -m rl.vq2.success --reset      # delete ALL stored demonstrations
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np

DATA = os.path.join("rl", "data", "vq2")
ROOT = os.path.join(DATA, "success")           # success/gate1/, gate2/, ...
DEMOS = os.path.join(DATA, "demos.npz")          # expert BC bootstrap


def _gate_dir(n: int) -> str:
    return os.path.join(ROOT, f"gate{n}")


def save_segment(gate_n, obs, act, *, reward=None, sim_us=None, gate_index=None,
                 gates_reached=0, collisions=0) -> str:
    """Save one successful segment (start -> gate_n). Auto-numbers within the gate
    dir so nothing is ever overwritten."""
    d = _gate_dir(int(gate_n))
    os.makedirs(d, exist_ok=True)
    obs = np.asarray(obs, np.float32)
    act = np.asarray(act, np.float32)
    n = len(act)
    smooth = float(np.mean(np.abs(np.diff(act, axis=0)))) if n > 1 else 0.0
    seq = len(glob.glob(os.path.join(d, "*.npz")))
    path = os.path.join(d, f"seg_{seq:05d}_{gates_reached}g_{collisions}c_{n}s.npz")
    np.savez(
        path,
        obs=obs, act=act,
        reward=np.asarray(reward if reward is not None else np.zeros(n), np.float32),
        sim_us=np.asarray(sim_us if sim_us is not None else np.zeros(n), np.int64),
        gate_index=np.asarray(gate_index if gate_index is not None else np.zeros(n), np.int32),
        gates_reached=int(gates_reached), collisions=int(collisions), smoothness=smooth,
    )
    return path


def list_segments() -> dict[int, list[str]]:
    """{gate_n: [paths]} for gates that have at least one saved segment."""
    out: dict[int, list[str]] = {}
    if not os.path.isdir(ROOT):
        return out
    for gd in sorted(glob.glob(os.path.join(ROOT, "gate*"))):
        try:
            n = int(os.path.basename(gd)[4:])
        except ValueError:
            continue
        files = sorted(glob.glob(os.path.join(gd, "*.npz")))
        if files:
            out[n] = files
    return out


def counts() -> dict[int, int]:
    return {n: len(f) for n, f in list_segments().items()}


def reset(include_demos: bool = True) -> None:
    """Delete all success segments (and the expert demos.npz, for a clean slate)."""
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    if include_demos and os.path.exists(DEMOS):
        os.remove(DEMOS)


def _reps(gates_reached: int, collisions: int) -> int:
    """Oversample factor for BC: more gates + no collisions -> weighted higher."""
    score = int(gates_reached) + (1 if int(collisions) == 0 else 0)
    return max(1, min(5, score))


def load_for_bc(min_gate: int = 1, weight_by_preference: bool = True):
    """Concatenate (obs, act) over ALL saved segments for behaviour cloning.
    Better segments are repeated (oversampled). Returns (obs, act) or (None, None)."""
    obs_all, act_all = [], []
    for n, files in list_segments().items():
        if n < min_gate:
            continue
        for p in files:
            d = np.load(p)
            reps = (_reps(int(d["gates_reached"]), int(d["collisions"]))
                    if weight_by_preference else 1)
            for _ in range(reps):
                obs_all.append(d["obs"])
                act_all.append(d["act"])
    if not obs_all:
        return None, None
    return np.concatenate(obs_all).astype(np.float32), np.concatenate(act_all).astype(np.float32)


def summary() -> str:
    c = counts()
    if not c:
        return "(no successful segments yet)"
    lines = []
    total_files = total_samples = 0
    for n in sorted(c):
        files = list_segments()[n]
        samples = sum(int(np.load(f)["act"].shape[0]) for f in files)
        best = max(int(np.load(f)["gates_reached"]) for f in files)
        clean = sum(int(np.load(f)["collisions"]) == 0 for f in files)
        total_files += len(files)
        total_samples += samples
        lines.append(f"  gate{n}: {len(files)} segs, {samples} samples, "
                     f"{clean} collision-free, best-run reached gate {best}")
    lines.append(f"  TOTAL: {total_files} segments, {total_samples} samples")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list segments per gate")
    ap.add_argument("--reset", action="store_true", help="delete ALL demos + successes")
    ap.add_argument("--keep-demos", action="store_true", help="reset successes but keep demos.npz")
    args = ap.parse_args()
    if args.reset:
        reset(include_demos=not args.keep_demos)
        print("[success] deleted all stored demonstrations"
              f"{' (kept demos.npz)' if args.keep_demos else ' (incl. demos.npz)'}.")
    else:
        print("[success] stored successful trajectories:")
        print(summary())
