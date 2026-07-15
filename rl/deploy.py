"""Module 8 (deploy) — run the trained policy on the live simulator.

Closes the loop end-to-end on the real sim:
    IMU + odometry --EKF--> filtered state --Module6--> 24-D obs
    --policy.pt--> action --scale_action--> attitude-rate + thrust --> MAVLink

Gate progression mirrors the training env (signed-distance plane crossing),
so the obs the policy sees in deployment matches what it saw in training.

State estimation: the EKF predicts on IMU and updates on the sim's odometry
position + attitude (loosely-coupled). If a trained gatenet.pt is present, the
GateNet->PnP vision pose can be fused too (Modules 3-4); odometry is the
robust default since the sim provides it directly.

    uv run -m rl.deploy                 # fly the policy on the live sim
    uv run -m rl.deploy --selftest      # closed-loop on internal env (no sim)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rl import spec
from rl.ekf import ESKF
from rl.env import GateRacingEnv
from rl.fly2_course import HOVER_T as LIVE_HOVER_THRUST
from rl.observation import build_observation
from rl.train_ppo import POLICY_PT, StandalonePolicy

# Training plant (rl/env) uses HOVER≈0.5 and rate caps of 4 rad/s. The live
# FlightSim plant hovers at ~0.27 and becomes unstable well below those rates
# (fly2 clips at 0.30). Remap scaled policy actions before MAVLink send.
TRAIN_HOVER_THRUST = 0.5
LIVE_RATE_CLIP = 0.60  # rad/s — above fly2's 0.30 so policy can bank, below 4
LIVE_THRUST_MIN = 0.12
LIVE_THRUST_MAX = 0.55
FLIP_RECOVERY_S = 2.0
TILT_FLIP_RAD = np.radians(70.0)


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


def live_scale_action(a: np.ndarray) -> np.ndarray:
    """Map policy [-1,1]^4 onto live-sim rate/thrust limits.

    Keeps the policy's signed deltas around hover, but recenters thrust on the
    measured live hover (~0.27) and hard-clips rates so a train-env policy
    cannot dump ±4 rad/s into FlightSim (the failure mode that flips the
    airframe after arm / reset).
    """
    roll, pitch, yaw, thrust_train = spec.scale_action(a)
    roll = float(np.clip(roll, -LIVE_RATE_CLIP, LIVE_RATE_CLIP))
    pitch = float(np.clip(pitch, -LIVE_RATE_CLIP, LIVE_RATE_CLIP))
    yaw = float(np.clip(yaw, -LIVE_RATE_CLIP, LIVE_RATE_CLIP))
    # thrust_train≈0.5 → LIVE_HOVER; keep proportional residuals so climb/dive
    # intents survive, but never near the train-env "double hover" of 0.5+.
    thrust = LIVE_HOVER_THRUST + (float(thrust_train) - TRAIN_HOVER_THRUST) * (
        LIVE_HOVER_THRUST / TRAIN_HOVER_THRUST
    )
    thrust = float(np.clip(thrust, LIVE_THRUST_MIN, LIVE_THRUST_MAX))
    return np.array([roll, pitch, yaw, thrust], dtype=np.float64)


def _attitude_tilt_rad(q: np.ndarray) -> float:
    """Angle between body-down and world-down (0 = upright)."""
    gb = spec.quat_to_R(np.asarray(q, float)).T @ np.array([0.0, 0.0, 1.0])
    return float(np.arccos(np.clip(gb[2], -1.0, 1.0)))


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
        self._recover_until = 0.0
        self._armed_prev = False
        self._last_race_start = None

    def _seed_ekf(self, snap, gate_map: list) -> None:
        if snap.has_pose():
            self.ekf.p = np.array(snap.pos_ned)
            self.ekf.v = np.array(snap.vel_ned)
            self.ekf.q = np.array(snap.quat)
        self.gate_idx = 0
        self.last_action[:] = 0.0
        self._last_imu_t = None
        self._prev_signed = float(
            _gate_normal(gate_map[0]) @ (self.ekf.p - np.array(gate_map[0]["pos"]))
        )

    def run(self):
        from rl.fly2_course import resolve_gate_map
        from rl.sim_interface import GATE_MAP_PATH, SimInterface

        sim = SimInterface()
        # Stop per-frame vision spam so Race/arm logs stay visible.
        sim.data["_quiet_vision"] = True
        if not sim.wait_for_telemetry():
            print("[deploy] no telemetry — is the simulator running?")
            return

        # Live burst (same as make capture-gates) with JSON fallback — VQ2 and
        # missed race-start bursts previously aborted here before arming.
        gate_map = sim.capture_gate_map(timeout_s=90.0)
        if not gate_map:
            gate_map = resolve_gate_map(sim.data)
        if not gate_map:
            print(
                f"[deploy] no gate map; aborting. Run `make capture-gates`, "
                f"click Race while it listens, then `make fly-policy` again "
                f"(expects {GATE_MAP_PATH}).",
                flush=True,
            )
            return
        print(f"[deploy] using {len(gate_map)} gates", flush=True)
        print(
            f"[deploy] live remap: hover={LIVE_HOVER_THRUST:.2f} "
            f"rate_clip=±{LIVE_RATE_CLIP:.2f} rad/s "
            "(policy trained at hover=0.5 / ±4 rad/s)",
            flush=True,
        )

        snap = sim.snapshot()
        if not snap.has_pose():
            print(
                "[deploy] WARNING: no pose yet — EKF starts at origin; "
                "prefer TRAINING with odometry or wait for estimator",
                flush=True,
            )
        self._seed_ekf(snap, gate_map)
        sim.arm()
        print("[deploy] armed; flying policy...", flush=True)

        while True:
            snap = sim.snapshot()
            race = sim.data.get("race_status") or {}
            race_start = race.get("race_start_boot_time_ms", -1)
            # Sim auto-reset / restart: re-seed EKF + hold hover so the policy
            # does not keep commanding the post-crash tumble rates.
            if (
                self._last_race_start is not None
                and race_start is not None
                and race_start >= 0
                and race_start != self._last_race_start
            ):
                print(
                    "[deploy] race restart detected — reseeding + hover hold",
                    flush=True,
                )
                self._seed_ekf(snap, gate_map)
                self._recover_until = time.monotonic() + FLIP_RECOVERY_S
            if race_start is not None and race_start >= 0:
                self._last_race_start = race_start
            if snap.armed and not self._armed_prev:
                self._seed_ekf(snap, gate_map)
                self._recover_until = time.monotonic() + 0.5
            self._armed_prev = bool(snap.armed)

            # EKF predict on IMU — gate on the sensor timestamp, not the loop
            # clock: the control loop (100 Hz) outruns the IMU rate, so keying
            # off wall time would re-integrate the same sample and double the
            # predicted drift between corrections.
            if snap.imu is not None:
                imu_t = snap.imu["time_us"]
                if self._last_imu_t is not None and imu_t != self._last_imu_t:
                    dt = (imu_t - self._last_imu_t) * 1e-6
                    if 0.0 < dt < 0.1:
                        accel = np.array(
                            [snap.imu["ax"], snap.imu["ay"], snap.imu["az"]]
                        )
                        gyro = np.array(
                            [snap.imu["gx"], snap.imu["gy"], snap.imu["gz"]]
                        )
                        self.ekf.predict(accel, gyro, dt)
                self._last_imu_t = imu_t
            # EKF update on odometry position + attitude.
            if snap.pos_ned is not None:
                self.ekf.update_position(np.array(snap.pos_ned))
            if snap.quat is not None:
                self.ekf.update_attitude(np.array(snap.quat))

            st = self.ekf.state()
            ang_vel = np.array(snap.ang_vel) if snap.ang_vel else np.zeros(3)

            if _attitude_tilt_rad(st["q"]) > TILT_FLIP_RAD:
                if time.monotonic() >= self._recover_until:
                    print(
                        "[deploy] flip detected — leveling (zero rates + hover)",
                        flush=True,
                    )
                self._recover_until = time.monotonic() + FLIP_RECOVERY_S

            if time.monotonic() < self._recover_until:
                sim.send_attitude_rates(0.0, 0.0, 0.0, LIVE_HOVER_THRUST)
                time.sleep(1.0 / 100.0)
                continue

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
                        sim.send_attitude_rates(0, 0, 0, LIVE_HOVER_THRUST)
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
            roll, pitch, yaw, thrust = live_scale_action(action)
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
