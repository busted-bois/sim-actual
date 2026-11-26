"""Diff a replay trace against its original recording to find WHERE and WHY the
drone made different turns.

    make rl2-diff ARGS="run4"                       # original vs run4_replay
    make rl2-diff ARGS="run4_replay_A run4_replay_B"  # replay vs replay (sim non-determinism!)

Compares two traces on their common signals. No ground truth needed -- under a
deterministic sim, identical commands + timing + start => identical gyro/gate-pose.
The FIRST signal that diverges localizes the cause:
  * commands differ            -> normalization/clipping bug (should never happen)
  * sim-time offsets differ    -> timing/scheduling bug
  * gyro/gate-pose diverge but commands+timing match -> start-state misalignment
    OR the sim itself is non-deterministic.

CRUCIAL check: diff two replays of the SAME save (run rl2-run twice with --tag).
If those differ, the sim is non-deterministic and open-loop replay can NEVER
reproduce a flight -- no amount of recording/scheduling fixes it.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rl.vq2 import controller

SAVES = os.path.join("rl", "data", "vq2", "saves")


def _path(name: str) -> str:
    for cand in (name, name + ".npz", os.path.join(SAVES, name),
                 os.path.join(SAVES, name + ".npz")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"missing trace '{name}' (in {SAVES})")


def _trace(name: str):
    """Load either format (original with 'act', or replay with 'cmd') and
    normalize to a common dict of comparable signals."""
    p = _path(name)
    d = np.load(p)
    if "cmd" in d:
        cmd = d["cmd"].astype(np.float32)
    else:  # original save -> derive the wire command the replay would send
        cmd = np.array([controller.action_to_attitude(a) for a in d["act"]], np.float32)
    return {
        "path": p,
        "sim_us": d["sim_us"].astype(np.int64),
        "cmd": cmd,
        "gyro": d["gyro"].astype(np.float32) if "gyro" in d else None,
        "gate_pose": d["gate_pose"].astype(np.float32) if "gate_pose" in d else None,
        "gate_index": d["gate_index"].astype(np.int32) if "gate_index" in d else None,
    }


def _first_exceed(series: np.ndarray, thr: float):
    idx = np.where(np.nan_to_num(series) > thr)[0]
    return int(idx[0]) if len(idx) else None


def _passes(gate_idx: np.ndarray, sim_us: np.ndarray) -> dict:
    """{gate g: (step, sim-time s)} for the first step active_gate reaches g."""
    gi = gate_idx.astype(int)
    out = {}
    top = int(gi.max()) if len(gi) and gi.max() > 0 else 0
    for g in range(1, top + 1):
        w = np.where(gi >= g)[0]
        if len(w):
            out[g] = (int(w[0]), round(float((sim_us[w[0]] - sim_us[0]) / 1e6), 2))
    return out


def diff(name_a: str, name_b: str | None = None) -> None:
    a = _trace(name_a)
    b = _trace(name_b) if name_b else _trace(name_a.replace(".npz", "") + "_replay")
    # if only one name given and it's an original, compare it to its _replay above;
    # that already resolved. Label both by path.
    print(f"A: {a['path']}\nB: {b['path']}\n", flush=True)

    na, nb = len(a["cmd"]), len(b["cmd"])
    n = min(na, nb)
    if na != nb:
        print(f"[!] step counts differ: A={na} B={nb} "
              f"(one ended early -> drone diverged/crashed) -- comparing first {n}\n")

    o_sim, r_sim = a["sim_us"][:n], b["sim_us"][:n]
    o_off = (o_sim - o_sim[0]) / 1e6
    r_off = (r_sim - r_sim[0]) / 1e6

    # 1) command values (both are the SAME wire command by construction -> must match)
    cdiff = np.abs(a["cmd"][:n] - b["cmd"][:n]).max(axis=1)
    sc = _first_exceed(cdiff, 1e-3)
    print(f"command diff:    max={cdiff.max():.4f}   first >0.001 @ step {sc}   "
          f"(expected: None -- else clipping/normalization bug)")

    # 2) sim-time offset (did each command land at the same sim-time?)
    dt = np.abs(o_off - r_off)
    st = _first_exceed(dt, 0.02)
    print(f"sim-time drift:  max={dt.max()*1000:.0f}ms mean={dt.mean()*1000:.0f}ms   "
          f"first >20ms @ step {st}")

    # 3) gyro (the drone's actual response -- clean in VQ2)
    sg = None
    if a["gyro"] is not None and b["gyro"] is not None:
        gd = np.linalg.norm(np.nan_to_num(a["gyro"][:n] - b["gyro"][:n]), axis=1)
        sg = _first_exceed(gd, 0.5)
        print(f"gyro L2 diff:    max={gd.max():.2f} rad/s   first >0.5 @ step {sg}")

    # 4) gate pose (noisy: best-gate can jump; only compare same-gate frames)
    sp = None
    if a["gate_pose"] is not None and b["gate_pose"] is not None:
        op, rp = a["gate_pose"][:n], b["gate_pose"][:n]
        same_gate = (a["gate_index"][:n] == b["gate_index"][:n]
                     if a["gate_index"] is not None and b["gate_index"] is not None
                     else np.ones(n, bool))
        valid = same_gate & ~(np.isnan(op).any(1) | np.isnan(rp).any(1))
        pd = np.full(n, np.nan)
        if valid.any():
            pd[valid] = np.linalg.norm(op[valid] - rp[valid], axis=1)
        sp = _first_exceed(pd, 0.5)
        pmax = float(np.nanmax(pd)) if valid.any() else float("nan")
        print(f"gate-pose diff:  max={pmax:.2f} m   first >0.5m @ step {sp}   "
              f"(same-gate frames only: {int(valid.sum())}/{n})")

    # 5) gate-pass timing (sim-authoritative checkpoints)
    if a["gate_index"] is not None and b["gate_index"] is not None:
        print("\ngate passes  (gate: step, sim-time s):")
        print(f"  A: {_passes(a['gate_index'], a['sim_us'])}")
        print(f"  B: {_passes(b['gate_index'], b['sim_us'])}")

    # earliest divergence across signals -> where to start debugging + why
    cands = {"command": sc, "sim-time": st, "gyro": sg, "gate-pose": sp}
    cands = {k: v for k, v in cands.items() if v is not None}
    print()
    if cands:
        first = min(cands, key=cands.get)
        step = cands[first]
        why = {
            "command": "commands themselves differ -> clipping/normalization bug",
            "sim-time": "commands land at different sim-times -> scheduling/timing jitter",
            "gyro": "same commands+timing, drone responds differently -> start-state "
                    "misalignment OR the sim is non-deterministic",
            "gate-pose": "trajectory splits here (commands+timing+gyro matched earlier)",
        }[first]
        print(f"=> FIRST divergence: '{first}' at step {step} "
              f"(sim-time {o_off[step]:.2f}s).\n   likely cause: {why}")
    else:
        print("=> no divergence above thresholds -- the two traces match.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="trace name; alone -> compares <a>.npz vs <a>_replay.npz")
    ap.add_argument("b", nargs="?", default=None,
                    help="second trace name -> compares <a> vs <b> (e.g. replay vs replay)")
    args = ap.parse_args()
    diff(args.a, args.b)
