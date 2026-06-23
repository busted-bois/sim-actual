"""Module 8 (deploy) — run the trained policy on the live simulator.

Closes the loop end-to-end on the real sim:
    IMU + odometry --EKF--> filtered state --Module6--> 24-D obs
    --policy.pt--> outer-loop action --scale_velocity_action-->
    geo_control.velocity_to_action--> attitude-rate + thrust --> MAVLink

Gate progression mirrors the training env (signed-distance plane crossing),
so the obs the policy sees in deployment matches what it saw in training.

    uv run -m rl.deploy                 # fly the policy on the live sim
    uv run -m rl.deploy --selftest      # closed-loop on internal env (no sim)
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from rl import spec
from rl.ekf import ESKF
from rl.env import GateRacingEnv
from rl.geo_control import GeoGains, velocity_to_action
from rl.observation import build_observation
from rl.policy import POLICY_PT, StandalonePolicy

# Live-sim inner loop (measured hover thrust 0.27).
DEPLOY_GAINS = GeoGains()


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


def _quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _gate_normal(g):
    return spec.quat_to_R(np.asarray(g["quat"], float)) @ np.array([1.0, 0, 0])


def _passed(p, prev_signed, g):
    """Return (passed, new_signed)."""
    gc = np.asarray(g["pos"], float)
    R = spec.quat_to_R(np.asarray(g["quat"], float))
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

    def run(self):
        from rl.sim_interface import SimInterface

        sim = SimInterface()
        if not sim.wait_for_telemetry():
            print("[deploy] no telemetry — is the simulator running?")
            return
        gate_map = sim.capture_gate_map()
        if not gate_map:
            print("[deploy] no gate map; aborting")
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

            # Gate progression.
            if self.gate_idx < len(gate_map):
                passed, self._prev_signed = _passed(
                    st["p"], self._prev_signed, gate_map[self.gate_idx]
                )
                if passed:
                    self.gate_idx += 1
                    print(
                        f"[deploy] gate {self.gate_idx}/{len(gate_map)} passed",
                        flush=True,
                    )
                    if self.gate_idx >= len(gate_map):
                        print("[deploy] COURSE COMPLETE", flush=True)
                        sim.send_attitude_rates(0, 0, 0, DEPLOY_GAINS.hover_thrust)
                        return
                    g = gate_map[self.gate_idx]
                    self._prev_signed = float(
                        _gate_normal(g) @ (st["p"] - np.array(g["pos"]))
                    )

            obs = build_observation(
                st["p"],
                st["v"],
                st["q"],
                ang_vel,
                gate_map,
                self.gate_idx,
                self.last_action[:3],
            )
            action = self.act(obs)
            self.last_action = action
            v_des, yaw_rate_des = spec.scale_velocity_action(action)
            roll, pitch, yaw = _quat_to_rpy(st["q"])
            rates_thrust = velocity_to_action(
                v_des,
                yaw_rate_des,
                st["v"],
                roll,
                pitch,
                yaw,
                DEPLOY_GAINS,
            )
            sim.send_attitude_rates(*rates_thrust)
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
