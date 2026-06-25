"""Module 8 (deploy) — run the trained policy on the live simulator.

Closes the loop end-to-end on the real sim:
    IMU + odometry --EKF--> filtered state --Module6--> 24-D obs
    --policy.pt--> action --scale_action--> attitude-rate + thrust --> MAVLink

Gate progression mirrors the training env (signed-distance plane crossing),
so the obs the policy sees in deployment matches what it saw in training.

State estimation: the EKF predicts on IMU and updates on the sim's odometry
position + attitude (loosely-coupled). If a trained gate_pose.pt is present, the
YOLO-pose -> PnP vision pipeline (Modules 3-4) builds a world gate tracker and
the policy flies the vision-refined gate map instead of the raw broadcast;
odometry still anchors the drone position.

    uv run -m rl.deploy                 # fly the policy on the live sim
    uv run -m rl.deploy --selftest      # closed-loop on internal env (no sim)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rl import gatepose, spec
from rl.ekf import ESKF
from rl.env import GateRacingEnv
from rl.gate_tracker import GateTracker
from rl.observation import build_observation
from rl.perception import GatePerception
from rl.train_ppo import POLICY_PT, StandalonePolicy


def load_policy(path: str = POLICY_PT, device: str = "cpu"):
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev)
    net = StandalonePolicy(
        obs_dim=ckpt.get("obs_dim", spec.OBS_DIM),
        act_dim=ckpt.get("act_dim", spec.ACTION_DIM),
        arch=ckpt.get("arch", [64, 64, 64]),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.to(dev)
    net.eval()

    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, np.float32)[None]).to(dev)
        return np.clip(net(x)[0].cpu().numpy(), -1.0, 1.0)

    return act


def _gate_normal(g):
    return spec.gate_normal_world(g["quat"])


def _passed(p, prev_signed, g):
    """Return (passed, new_signed)."""
    gc = np.asarray(g["pos"], float)
    R = spec.gate_rotation(g["quat"])  # convention-corrected gate frame
    n = R[:, 0]  # gate normal (through-axis)
    signed = float(n @ (p - gc))
    if prev_signed < 0.0 <= signed:
        # Square opening: offsets along the gate's local right/down axes vs its
        # reported w/h.
        rel = p - gc
        dy = abs(float(R[:, 1] @ rel))  # width axis (right)
        dz = abs(float(R[:, 2] @ rel))  # height axis (down)
        hw = 0.5 * float(g.get("w", spec.GATE_SIZE_M))
        hh = 0.5 * float(g.get("h", spec.GATE_SIZE_M))
        return (dy < hw and dz < hh), signed
    return False, signed


class PolicyRunner:
    def __init__(self, policy_path: str = POLICY_PT):
        self.act = load_policy(policy_path)
        self.ekf = ESKF()
        self.last_action = np.zeros(spec.ACTION_DIM)
        self.gate_idx = 0
        self._prev_signed = None
        self._last_imu_t = None

        # Vision: YOLO-pose -> PnP -> world gate tracker. Auto-on when a trained
        # gate_pose.pt exists; otherwise fall back to the broadcast gate map.
        self.tracker = GateTracker()
        self.vision = None
        self._last_vision_fid = -1
        if os.path.exists(gatepose.WEIGHTS_PATH):
            try:
                self.vision = GatePerception()
                print(f"[deploy] vision ON — {gatepose.WEIGHTS_PATH}", flush=True)
            except Exception as e:  # missing ultralytics / bad weights
                print(f"[deploy] vision OFF — {e}", flush=True)
        else:
            print("[deploy] vision OFF — no gate_pose.pt (broadcast map)", flush=True)

    def run(self):
        from rl.sim_interface import GATE_MAP_PATH, SimInterface, load_gate_map

        sim = SimInterface()
        if not sim.wait_for_telemetry():
            print("[deploy] no telemetry — is the simulator running?")
            return
        # Gate map is broadcast once at race start; if we joined mid-race and
        # missed it, fall back to the saved map (static course/origin). Vision
        # refines gate POSITIONS live, so a slightly stale saved map is fine.
        gate_map = sim.capture_gate_map(timeout_s=5.0)
        if not gate_map and os.path.exists(GATE_MAP_PATH):
            gate_map = load_gate_map(GATE_MAP_PATH)
            print(f"[deploy] using saved gate map ({len(gate_map)} gates)", flush=True)
        if not gate_map:
            print("[deploy] no gate map (no broadcast, no saved file); aborting")
            return

        # Seed EKF from first odometry.
        snap = sim.snapshot()
        if snap.has_pose():
            self.ekf.p = np.array(snap.pos_ned)
            self.ekf.v = np.array(snap.vel_ned)
            self.ekf.q = np.array(snap.quat)
        self._prev_signed = float(
            _gate_normal(gate_map[0]) @ (self.ekf.p - np.array(gate_map[0]["pos"]))
        )
        sim.arm()
        print("[deploy] armed; flying policy...", flush=True)

        while True:
            snap = sim.snapshot()

            # EKF predict on IMU — gate on the sensor timestamp, not the loop
            # clock: the control loop (100 Hz) outruns the IMU rate, so keying
            # off wall time would re-integrate the same sample and double the
            # predicted drift between corrections.
            if snap.imu is not None:
                imu_t = snap.imu["time_us"]
                if self._last_imu_t is not None and imu_t != self._last_imu_t:
                    dt = (imu_t - self._last_imu_t) * 1e-6
                    accel = np.array([snap.imu["ax"], snap.imu["ay"], snap.imu["az"]])
                    gyro = np.array([snap.imu["gx"], snap.imu["gy"], snap.imu["gz"]])
                    self.ekf.predict(accel, gyro, dt)
                self._last_imu_t = imu_t
            # EKF update on odometry position + attitude.
            if snap.pos_ned is not None:
                self.ekf.update_position(np.array(snap.pos_ned))
            if snap.quat is not None:
                self.ekf.update_attitude(np.array(snap.quat))

            st = self.ekf.state()
            ang_vel = np.array(snap.ang_vel) if snap.ang_vel else np.zeros(3)

            # --- Vision: detect gates this frame -> world tracks -> map fix ----
            # The policy flies vision-refined gate positions, keeping the
            # broadcast map's order/orientation/size. Drift cancels because the
            # track and the observation use the same drone pose (see gate_tracker).
            effective_map = gate_map
            if self.vision is not None:
                frame = sim.data.get("frame")
                if frame is not None and frame["frame_id"] != self._last_vision_fid:
                    self._last_vision_fid = frame["frame_id"]
                    seen = self.vision.process(frame["img"], st["p"], st["q"])
                    self.tracker.update(
                        [g["gate_pos_world"] for g in seen], snap.t_mono
                    )
                effective_map = self.tracker.corrected_map(gate_map)

            # Gate progression (on the vision-corrected map).
            if self.gate_idx < len(effective_map):
                passed, self._prev_signed = _passed(
                    st["p"], self._prev_signed, effective_map[self.gate_idx]
                )
                if passed:
                    self.gate_idx += 1
                    print(
                        f"[deploy] gate {self.gate_idx}/{len(effective_map)} passed",
                        flush=True,
                    )
                    if self.gate_idx >= len(effective_map):
                        print("[deploy] COURSE COMPLETE", flush=True)
                        sim.send_attitude_rates(0, 0, 0, spec.HOVER_THRUST)
                        return
                    g = effective_map[self.gate_idx]
                    self._prev_signed = float(
                        _gate_normal(g) @ (st["p"] - np.array(g["pos"]))
                    )

            obs = build_observation(
                st["p"],
                st["v"],
                st["q"],
                ang_vel,
                effective_map,
                self.gate_idx,
                self.last_action[:3],
            )
            action = self.act(obs)
            self.last_action = action
            roll, pitch, yaw, thrust = spec.scale_action(action)
            sim.send_attitude_rates(roll, pitch, yaw, thrust)
            time.sleep(1.0 / 100.0)


def _selftest():
    if not os.path.exists(POLICY_PT):
        print("[selftest] no policy.pt — run `uv run -m rl.train_ppo --quick` first")
        return
    act = load_policy(POLICY_PT)
    # Closed-loop on the internal env (deterministic policy) — verifies the full
    # obs->policy->action loop runs and the exported weights drive the sim model.
    for stage in (0, 2):
        env = GateRacingEnv(stage=stage, seed=7)
        o, _ = env.reset()
        assert o.shape == (spec.OBS_DIM,)
        tot, steps = 0.0, 0
        term = trunc = False
        while not (term or trunc):
            a = act(o)
            assert a.shape == (spec.ACTION_DIM,) and np.all(np.isfinite(a))
            o, r, term, trunc, info = env.step(a)
            tot += r
            steps += 1
        print(
            f"[selftest] stage {stage}: policy ran {steps} steps, "
            f"reward={tot:.1f}, gate_idx={env.gate_idx}, info={info}"
        )
    print("[selftest] OK — policy.pt loads, infers, and drives the loop")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        PolicyRunner().run()
