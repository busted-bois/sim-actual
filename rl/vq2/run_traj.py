"""Replay a saved trajectory's ACTIONS open-loop in the real simulator.

This is the milestone-10 verification: it takes a trajectory saved by
`rl2-log-demos` and re-sends its recorded action sequence through the EXACT same
path the RL policy/servo uses -- `controller.action_to_attitude` ->
`SimInterface.send_attitude_quat_deg` -- at the recorded timing. If the drone
flies the same course as the original demonstration, the save + action-channel
are faithful (and the policy trained on these actions will reproduce the flight).

No neural net, no pilot -- just the recorded commands.

    # start (or restart) the race in the sim, then:
    make rl2-run ARGS="demo_1737764812"      # a save name from rl/data/vq2/saves/
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from simulator.gp_vision import _yolo_pose_estimate
from rl.vq2 import controller
from rl.vq2.reset import RaceGo
from rl.sim_interface import SimInterface

DATA = os.path.join("rl", "data", "vq2")
SAVES = os.path.join(DATA, "saves")


def _resolve(name: str) -> str:
    """Accept a bare name, a name.npz, or a full path."""
    for cand in (name, name + ".npz", os.path.join(SAVES, name),
                 os.path.join(SAVES, name + ".npz")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"no trajectory '{name}' (looked in {SAVES}). List saves: make rl2-list-demos")


def _dts(t: np.ndarray, n: int) -> np.ndarray:
    """Per-step sleep from recorded WALL-clock timestamps (clamped) -- fallback for
    old saves without a sim clock. Falls back to 50 Hz if timestamps are missing."""
    if t is not None and len(t) == n and n > 1:
        d = np.diff(t)
        d = np.clip(d, 0.0, 0.1)
        return np.append(d, d[-1] if len(d) else 0.02)
    return np.full(n, 1.0 / 50.0)


def _cur_sim_us(sim: SimInterface):
    """Current sim clock (HIGHRES_IMU time_us), or None if no fresh IMU."""
    imu = sim.data.get("imu")
    ts = (imu or {}).get("time_us") or (imu or {}).get("time_usec")
    return int(ts) if ts else None


def _gyro(sim: SimInterface):
    imu = sim.data.get("imu") or {}
    return (float(imu.get("gx", float("nan"))), float(imu.get("gy", float("nan"))),
            float(imu.get("gz", float("nan"))))


def _pose_xyz(sim: SimInterface):
    p = _yolo_pose_estimate(sim.data)
    if p is None:
        return (float("nan"), float("nan"), float("nan"))
    return (float(p["body_x_m"]), float(p["body_y_m"]), float(p["body_z_m"]))


def replay(name: str, wait_race: bool = True, tag: str = "", no_yolo: bool = False):
    path = _resolve(name)
    d = np.load(path)
    act = d["act"].astype(np.float32)
    t = d["t"] if "t" in d else None
    sim_us = d["sim_us"] if "sim_us" in d else None
    rec_gate = d["gate_index"] if "gate_index" in d else None
    n = len(act)
    dts = _dts(t, n)
    # Schedule on the SIM clock when we have it (deterministic-correct): send each
    # command when the sim advances to its recorded offset-from-GO, so wall-clock
    # jitter / sim time-scaling can't shift when commands land.
    use_sim_time = (sim_us is not None and len(sim_us) == n and n > 1
                    and bool(np.all(np.asarray(sim_us) > 0)))
    base_us = int(sim_us[0]) if use_sim_time else 0
    mode = "SIM-clock" if use_sim_time else "wall-clock (no sim_us in save)"
    dur = float((sim_us[-1] - sim_us[0]) / 1e6) if use_sim_time else float(np.sum(dts))
    print(f"[rl2-run] loaded {path}: {n} steps (~{dur:.1f}s recorded)  timing={mode}",
          flush=True)

    sim = SimInterface()
    sim.data["_quiet_vision"] = True
    if no_yolo:
        # Kill the YOLO-pose thread so it can't contend for the GPU with the sim's
        # rendering (tests whether vision load perturbs sim timing run-to-run).
        # gate_pose stops updating -> gate_pose logged as NaN, which is fine.
        try:
            sim.vision_rx.gate_pose.is_running = False
            sim.vision_rx.is_running = False        # also stop camera-frame RX
            print("[rl2-run] YOLO + camera RX DISABLED for this replay.", flush=True)
        except Exception as exc:
            print(f"[rl2-run] could not disable vision: {exc}", flush=True)
    print("[rl2-run] connected. Start/Restart the race in the sim...", flush=True)
    # Record the replay's OWN observed trace so it can be diffed against the
    # original: when the drone turns differently, the diff shows the first step
    # where they split and which signal moved -> why.
    r_sim, r_cmd, r_gyro, r_pose, r_gate, r_wall = [], [], [], [], [], []
    try:
        # Arm, then wait for the REAL GO -- the GPPilot WAIT_FOR_START gate: a
        # fresh countdown that has ELAPSED with the physics clock live. Waiting on
        # `race_started` alone starts during the countdown and tips the drone over.
        last_arm = 0.0
        if wait_race:
            print("[rl2-run] armed -- WAIT_FOR_START (start/restart the race)...",
                  flush=True)
            go = RaceGo(debug=True)
            while not go(sim.data):
                now = time.time()
                if now - last_arm > 1.0:
                    sim.arm()
                    last_arm = now
                time.sleep(1 / 90.0)
        sim.arm()
        print("[rl2-run] Countdown complete! Replaying recorded actions...", flush=True)

        t0 = time.time()
        go_sim = None                       # sim clock captured at the first command
        guard_fires = 0                     # commands sent WITHOUT reaching their sim-time
        max_wait = 0.0
        DEAD_SIM_S = 3.0                    # only escape a TRULY frozen sim clock
        for i in range(n):
            if not sim.data.get("armed", False):
                sim.arm()

            # ---- send when the SIM clock reaches this command's recorded offset ----
            # STRICT: wait until the sim clock actually crosses the target, so the
            # command lands at the SAME sim-time every run (the old 0.5 s give-up
            # fired early at inconsistent sim-times -> the 466 ms run-to-run drift).
            # Only a multi-second frozen clock breaks the wait, and that's counted.
            if use_sim_time:
                target = int(sim_us[i]) - base_us          # us since first command
                w0 = time.time()
                while True:
                    cs = _cur_sim_us(sim)
                    if cs is not None:
                        if go_sim is None:
                            go_sim = cs                    # t=0 of the replay, in sim-time
                        if cs - go_sim >= target:
                            break
                    if time.time() - w0 > DEAD_SIM_S:      # sim clock truly frozen
                        guard_fires += 1
                        break
                    time.sleep(0.001)                      # yields GIL so RX keeps updating IMU
                max_wait = max(max_wait, time.time() - w0)

            roll_deg, pitch_deg, yaw_deg, thrust = controller.action_to_attitude(act[i])
            sim.send_attitude_quat_deg(roll_deg, pitch_deg, yaw_deg, thrust)

            # capture what actually happened this step (for the diff).
            r_sim.append(_cur_sim_us(sim) or 0)
            r_cmd.append((roll_deg, pitch_deg, yaw_deg, thrust))
            r_gyro.append(_gyro(sim))
            r_pose.append(_pose_xyz(sim))
            r_gate.append(int(sim.data.get("active_gate_index", -1) or -1))
            r_wall.append(time.time() - t0)

            if i % 25 == 0:
                live_gate = sim.data.get("active_gate_index")
                rg = int(rec_gate[i]) if rec_gate is not None else -1
                soff = (int(sim_us[i]) - base_us) / 1e6 if use_sim_time else 0.0
                print(f"  step {i:4d}/{n}  wall={time.time()-t0:5.1f}s "
                      f"sim_off={soff:5.1f}s  recorded_gate={rg}  live_gate={live_gate}  "
                      f"cmd=(r{roll_deg:+.1f} p{pitch_deg:+.1f} y{yaw_deg:+.1f} thr{thrust:.2f})",
                      flush=True)

            if not use_sim_time:
                time.sleep(float(dts[i]))
        live_gate = sim.data.get("active_gate_index")
        print(f"[rl2-run] replay done. final live gate index = {live_gate}", flush=True)
        if use_sim_time:
            print(f"[rl2-run] scheduler: {guard_fires}/{n} commands hit the "
                  f"{DEAD_SIM_S:.0f}s frozen-sim guard, max single wait {max_wait:.2f}s. "
                  f"0 guard-fires => the sim-time schedule is deterministic (any "
                  f"remaining run-to-run diff is the sim itself, not us).", flush=True)
    except KeyboardInterrupt:
        print("\n[rl2-run] stopped.", flush=True)
    finally:
        _save_replay(path, r_sim, r_cmd, r_gyro, r_pose, r_gate, r_wall, tag)
        sim.close()


def _save_replay(orig_path, r_sim, r_cmd, r_gyro, r_pose, r_gate, r_wall, tag=""):
    """Persist the replay's observed trace as <name>_replay[_<tag>].npz. Use --tag
    to keep multiple replays of the SAME save so you can diff replay-vs-replay
    (measures the sim's own non-determinism)."""
    if not r_cmd:
        print("[rl2-run] no steps replayed -- nothing to save.", flush=True)
        return
    base = os.path.splitext(os.path.basename(orig_path))[0]
    suffix = f"_replay_{tag}" if tag else "_replay"
    out = os.path.join(SAVES, f"{base}{suffix}.npz")
    np.savez(
        out,
        sim_us=np.asarray(r_sim, np.int64),
        cmd=np.asarray(r_cmd, np.float32),          # (roll,pitch,yaw,thrust) actually sent
        gyro=np.asarray(r_gyro, np.float32),        # drone response (clean in VQ2)
        gate_pose=np.asarray(r_pose, np.float32),   # YOLO body-frame gate xyz
        gate_index=np.asarray(r_gate, np.int32),
        wall=np.asarray(r_wall, np.float64),
    )
    print(f"[rl2-run] replay trace saved -> {out}  ({len(r_cmd)} steps)", flush=True)
    print(f"[rl2-run] DIFF vs original:  make rl2-diff ARGS=\"{base}\"", flush=True)
    if tag:
        print(f"[rl2-run] DIFF replay-vs-replay: make rl2-diff "
              f"ARGS=\"{base}{suffix} {base}_replay_<other>\"", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="trajectory save name (from rl/data/vq2/saves/)")
    ap.add_argument("--no-wait", action="store_true",
                    help="don't wait for race_started; replay immediately")
    ap.add_argument("--tag", default="",
                    help="label this replay (saves <name>_replay_<tag>.npz) to keep "
                         "multiple replays of the same save for replay-vs-replay diff")
    ap.add_argument("--no-yolo", action="store_true",
                    help="disable YOLO + camera RX during replay (removes GPU/vision "
                         "load as a variable in the timing-jitter test)")
    args = ap.parse_args()
    replay(args.name, wait_race=not args.no_wait, tag=args.tag, no_yolo=args.no_yolo)
