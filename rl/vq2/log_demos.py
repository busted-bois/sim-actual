"""Collect behaviour-cloning demos by running the PROVEN GP pilot in the real
VQ2 sim and taping its command.

The GP pilot (`AUTO_PILOT=gp`, the classical vision-servo flyer) already clears
gates on VQ2 and sends `controller.set_attitude_quat_deg(roll,pitch,yaw,thrust)`.
We tap that command, rebuild the SAME observation the RL policy will see
(`rl.vq2.observation` from the pilot's GPEstimation + YOLO/PnP vision), map the
command into our normalized action space (inverse of `rl.vq2.controller`), and
log (obs, action) pairs. A BC warm-start on these (`rl.vq2.train_bc`) puts PPO in
the right region so real-time training converges in hours, not a week.

Cloning the tuned pilot (right signs, flies the course) avoids a hand-written
expert. Run it during a normal race; each run APPENDS to rl/data/vq2/demos.npz.

    # start the VQ2 sim, then:
    uv run -m rl.vq2.log_demos            # arms, flies GP pilot, logs until course-done/Ctrl-C
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

DATA = os.path.join("rl", "data", "vq2")
DEMOS = os.path.join(DATA, "demos.npz")
SAVES = os.path.join(DATA, "saves")   # per-run replayable trajectories


def main(name: str | None = None):
    os.environ.setdefault("AUTO_PILOT", "gp")
    os.environ.setdefault("GP_DISPLAY", "0")

    from simulator.setup import setup_components
    from simulator.gp_vision import _yolo_pose_estimate, best_pose_gate
    from rl.vq2.controller import gp_cmd_to_action
    from rl.vq2.observation import ObsStacker, build_frame

    data = {"_quiet_vision": True}
    boot_ms = int(time.time() * 1000)
    try:
        comps = setup_components(data, boot_ms, "127.0.0.1", 14550)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", flush=True)
        sys.exit(1)
    controller = comps["controller"]

    # Tap the GP pilot's outgoing attitude-quat command.
    last = {"cmd": None}
    _orig = controller.set_attitude_quat_deg

    def _tapped(roll_deg, pitch_deg, yaw_deg, thrust):
        last["cmd"] = (roll_deg, pitch_deg, yaw_deg, thrust)
        return _orig(roll_deg, pitch_deg, yaw_deg, thrust)

    controller.set_attitude_quat_deg = _tapped

    stacker = ObsStacker()
    obs_log, act_log, t_log, gate_log = [], [], [], []
    prev_action = np.zeros(4, np.float32)
    started = False
    if name is None:
        name = f"demo_{int(time.time())}"

    print("Arming GP pilot; start the race. Logging demos after GO (Ctrl+C to stop).",
          flush=True)
    controller.arm()
    last_tb = 0.0
    try:
        while True:
            try:
                controller.update()
            except Exception:
                now = time.monotonic()
                if now - last_tb >= 5.0:
                    traceback.print_exc()
                    last_tb = now
                time.sleep(1 / 90.0)
                continue

            if not data.get("race_started"):
                time.sleep(1 / 90.0)
                continue

            pilot = getattr(controller, "pilot", None)
            est = getattr(pilot, "est", None)
            cmd = last["cmd"]
            if est is None or cmd is None:
                time.sleep(1 / 90.0)
                continue

            ego = est.snapshot()
            # cmd = (roll_deg, pitch_deg, yaw_deg, thrust) -- absolute attitude on
            # the quat wire, the exact form action_to_attitude sends at deploy.
            action = gp_cmd_to_action(*cmd)

            pose_est = _yolo_pose_estimate(data)
            g = best_pose_gate(data)
            conf = float(g["conf"]) if g else 0.0
            frame = build_frame(pose_est, conf, ego, last_action=prev_action)
            obs = stacker.reset(frame) if not started else stacker.push(frame)
            started = True

            obs_log.append(obs)
            act_log.append(action)
            t_log.append(time.time())
            gate_log.append(int(data.get("active_gate_index", -1) or -1))
            prev_action = action

            # stop when the course completes (all gates cleared).
            if data.get("race_finish_time_ns", -1) is not None and \
                    int(data.get("race_finish_time_ns", -1) or -1) >= 0:
                print("[log_demos] race finished.", flush=True)
                break
            time.sleep(1 / 90.0)
    except KeyboardInterrupt:
        print("\n[log_demos] stopped.", flush=True)
    finally:
        controller.set_attitude_quat_deg = _orig
        _save_run(name, obs_log, act_log, t_log, gate_log)   # replayable, per-run
        _save(obs_log, act_log)                              # append to BC bootstrap
        for k in ("ts_loop", "mavlink_rx", "vision_rx"):
            try:
                comps[k].get_thread_for_join().join(timeout=2.0)
            except Exception:
                pass


def _save_run(name, obs_log, act_log, t_log, gate_log):
    """Save ONE run as a standalone, replayable trajectory (`rl2-run <name>`)."""
    if not act_log:
        return None
    os.makedirs(SAVES, exist_ok=True)
    path = os.path.join(SAVES, f"{name}.npz")
    np.savez(
        path,
        obs=np.asarray(obs_log, np.float32),
        act=np.asarray(act_log, np.float32),
        t=np.asarray(t_log, np.float64),           # per-step wall-clock (for replay timing)
        gate_index=np.asarray(gate_log, np.int32),  # active gate at each step
    )
    print(f"\n[log_demos] trajectory saved -> {path}  ({len(act_log)} steps)", flush=True)
    print(f"[log_demos] REPLAY IT WITH:  make rl2-run ARGS=\"{name}\"", flush=True)
    return path


def _save(obs_log, act_log):
    if not obs_log:
        print("[log_demos] no samples logged (did the race start?).", flush=True)
        return
    os.makedirs(DATA, exist_ok=True)
    new_obs = np.asarray(obs_log, dtype=np.float32)
    new_act = np.asarray(act_log, dtype=np.float32)
    if os.path.exists(DEMOS):
        old = np.load(DEMOS)
        new_obs = np.concatenate([old["obs"], new_obs])
        new_act = np.concatenate([old["act"], new_act])
    np.savez(DEMOS, obs=new_obs, act=new_act)
    print(f"[log_demos] saved {len(new_act)} total samples -> {DEMOS} "
          f"(obs {new_obs.shape}, act {new_act.shape})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="save name for the replayable trajectory (default: demo_<ts>)")
    args = ap.parse_args()
    main(args.name)
