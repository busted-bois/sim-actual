"""Module 8 (deploy) — run the trained policy on the live simulator.

Closes the loop end-to-end on the real sim:
    IMU + commanded thrust --ESKF--> filtered state --Module6--> 24-D obs
    --policy.pt--> action --scale_action--> attitude-rate + thrust --> MAVLink

State estimation: the ESKF predicts on commanded thrust + gyro (NOT raw
accelerometer). Updates from odometry + native PnP via step_pnp_fusion.

Fallback uses camera-native expert (GateEstimateSmoother + VisionVelocityTracker
+ compute_guidance). Recovery requires accepted PnP fusion hysteresis.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time

import numpy as np
import torch

from rl.core import spec
from rl.core.config import load_config, resolve_device
from rl.estimation.ekf import ESKF
from rl.environment.env import GateRacingEnv
from rl.experts.fly2_course import HOVER_T as LIVE_HOVER_THRUST, detect_climb_course
from rl.core.observation import RATE_SCALE, build_observation
from rl.experts.vision_fallback import (
    LIVE_THRUST_MAX,
    LIVE_THRUST_MIN,
    FallbackBrain,
)
from rl.perception.vision_fusion import step_pnp_fusion
from rl.training.train_ppo import POLICY_PT, StandalonePolicy
from simulator.gp_vision import best_pose_gate
from simulator.vq2_pose import gate_z_ned

LEGACY_TRAIN_HOVER = 0.5
LEGACY_ACTION_SCALE = (4.0, 4.0, 3.0)
LIVE_RATE_CLIP = 0.60
FLIP_RECOVERY_S = 2.0
TILT_FLIP_RAD = np.radians(70.0)

MAX_COAST_FRAMES = 30
MAX_COVARIANCE_TRACE = 80.0
STALL_GATE_TICKS = 300
RECOVERY_PNP_FRAMES = 3  # accepted fresh PnP frames before policy resume
PNP_GATE_FLOOR = 3.0
CAM_HZ = 30.0

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def normalize_gate_map(gate_map: list) -> list:
    """Deep-copy gate map with NED-normalized positions using detect_climb_course."""
    if not gate_map:
        return []
    gm = copy.deepcopy(gate_map)
    flipz = detect_climb_course(gm)
    for g in gm:
        p = g["pos"]
        g["pos"] = [p[0], p[1], gate_z_ned(p, flipz)]
    return gm


def load_policy(path: str = POLICY_PT, device: str = "cpu"):
    if not os.path.exists(path):
        raise SystemExit(
            f"[deploy] no policy at {path} — run `make train-ppo` "
            "(or `uv run -m rl.training.train_ppo --quick` for a smoke checkpoint)"
        )
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    net = StandalonePolicy(
        obs_dim=ckpt.get("obs_dim", spec.OBS_DIM),
        act_dim=ckpt.get("act_dim", spec.ACTION_DIM),
        arch=ckpt.get("arch", [64, 64, 64]),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.to(dev)
    net.eval()

    if "train_hover_thrust" not in ckpt or "action_scale" not in ckpt:
        print(
            "[deploy] WARNING: legacy checkpoint without training metadata — "
            f"assuming hover={LEGACY_TRAIN_HOVER} rates ±{LEGACY_ACTION_SCALE}"
            " rad/s (pre-calibration plant). Retrain to embed the real plant.",
            flush=True,
        )
    meta = {
        "train_hover": float(ckpt.get("train_hover_thrust", LEGACY_TRAIN_HOVER)),
        "action_scale": tuple(
            float(s) for s in ckpt.get("action_scale", LEGACY_ACTION_SCALE)
        ),
    }
    if abs(meta["action_scale"][1] - RATE_SCALE) > 1e-9:
        print(
            "[deploy] WARNING: checkpoint pitch-rate scale "
            f"{meta['action_scale'][1]:.2f} != current obs RATE_SCALE "
            f"{RATE_SCALE:.2f} — the policy's observation normalization no "
            "longer matches this build; retrain before trusting it.",
            flush=True,
        )

    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, np.float32)[None]).to(dev)
        return np.clip(net(x)[0].cpu().numpy(), -1.0, 1.0)

    return act, meta


def live_scale_action(a: np.ndarray, meta: dict) -> np.ndarray:
    a = np.clip(np.asarray(a, np.float64), -1.0, 1.0)
    scale = meta["action_scale"]
    roll = float(np.clip(a[0] * scale[0], -LIVE_RATE_CLIP, LIVE_RATE_CLIP))
    pitch = float(np.clip(a[1] * scale[1], -LIVE_RATE_CLIP, LIVE_RATE_CLIP))
    yaw = float(np.clip(a[2] * scale[2], -LIVE_RATE_CLIP, LIVE_RATE_CLIP))
    thrust_train = (float(a[3]) + 1.0) * 0.5
    train_hover = meta["train_hover"]
    thrust = LIVE_HOVER_THRUST + (thrust_train - train_hover) * (
        LIVE_HOVER_THRUST / train_hover
    )
    thrust = float(np.clip(thrust, LIVE_THRUST_MIN, LIVE_THRUST_MAX))
    return np.array([roll, pitch, yaw, thrust], dtype=np.float64)


def _attitude_tilt_rad(q: np.ndarray) -> float:
    gb = spec.quat_to_R(np.asarray(q, float)).T @ np.array([0.0, 0.0, 1.0])
    return float(np.arccos(np.clip(gb[2], -1.0, 1.0)))


def _gate_normal(g):
    return spec.quat_to_R(np.asarray(g["quat"], float)) @ np.array([1.0, 0, 0])


def _passed(p, prev_signed, g):
    gc = np.asarray(g["pos"], float)
    R = spec.quat_to_R(np.asarray(g["quat"], float))
    n = R[:, 0]
    signed = float(n @ (p - gc))
    if prev_signed < 0.0 <= signed:
        rel = p - gc
        dy = abs(float(R[:, 1] @ rel))
        dz = abs(float(R[:, 2] @ rel))
        hw = 0.5 * float(g.get("w", spec.GATE_SIZE_M))
        hh = 0.5 * float(g.get("h", spec.GATE_SIZE_M))
        return (dy < hw and dz < hh), signed
    return False, signed


def _safe_extract_vec(d: dict, keys: list[str], shape: int) -> np.ndarray | None:
    try:
        v = np.array([float(d[k]) for k in keys], dtype=np.float64)
        if v.shape == (shape,) and np.isfinite(v).all():
            return v
    except (TypeError, ValueError, KeyError):
        pass
    return None


class DeployBrain:
    """Socket-free deploy decision logic: ESKF, fallback, gate progression."""

    def __init__(self, act, meta: dict, now_fn=None):
        self.act = act
        self.meta = meta
        self.ekf = ESKF()
        self.last_action = np.zeros(spec.ACTION_DIM)
        self.gate_idx = 0
        self._prev_signed = None
        self._last_imu_t = None
        self._recover_until = 0.0
        self._armed_prev = False
        self._last_race_start = None
        self._last_applied_thrust = 0.0
        self._fallback = FallbackBrain()
        self._in_fallback = False
        self._accepted_pnp_frames = 0
        self._last_gate_idx = 0
        self._last_pnp_frame_id = None
        self._last_gate_progress_t = None
        self._now = now_fn if now_fn is not None else time.monotonic

    def _seed_ekf(self, snap, gate_map: list) -> None:
        if snap.has_pose():
            p0 = np.array(snap.pos_ned)
            v0 = np.array(snap.vel_ned)
            q0 = np.array(snap.quat)
        else:
            p0 = self.ekf.p.copy()
            v0 = self.ekf.v.copy()
            q0 = self.ekf.q.copy()
        self.ekf.reset(p0=p0, v0=v0, q0=q0)
        self.gate_idx = 0
        self.last_action[:] = 0.0
        self._last_imu_t = None
        self._last_applied_thrust = 0.0
        self._prev_signed = float(
            _gate_normal(gate_map[0]) @ (self.ekf.p - np.array(gate_map[0]["pos"]))
        )
        self._in_fallback = False
        self._accepted_pnp_frames = 0
        self._last_gate_idx = 0
        self._last_gate_progress_t = None
        self._last_pnp_frame_id = None
        self._fallback.reset()

    def _check_ekf_healthy(self) -> bool:
        if not self.ekf.healthy():
            return False
        st = self.ekf.state()
        if st["P_trace"] > MAX_COVARIANCE_TRACE:
            return False
        if self.ekf.coast_count >= MAX_COAST_FRAMES:
            return False
        return True

    def _check_gate_stall(self, now: float, gate_map: list) -> bool:
        if self.gate_idx >= len(gate_map):
            return False
        if self.gate_idx != self._last_gate_idx:
            self._last_gate_idx = self.gate_idx
            self._last_gate_progress_t = now
            return False
        if self._last_gate_progress_t is None:
            self._last_gate_progress_t = now
        if now - self._last_gate_progress_t > STALL_GATE_TICKS / 100.0:
            return True
        return False

    def _should_engage_fallback(self, now: float, gate_map: list) -> bool:
        if self._in_fallback:
            return True
        if not self._check_ekf_healthy():
            return True
        if self._check_gate_stall(now, gate_map):
            return True
        return False

    def _run_pnp_fusion(self, data: dict, gate_map: list) -> str | None:
        """Run PnP fusion for one frame. Returns step_pnp_fusion result or None."""
        pnp_pkt = data.get("pose") or {}
        if not isinstance(pnp_pkt, dict):
            return None
        pnp_frame_id = pnp_pkt.get("frame_id")
        if pnp_frame_id is None or pnp_frame_id == self._last_pnp_frame_id:
            return None
        self._last_pnp_frame_id = pnp_frame_id
        try:
            active = int(data.get("active_gate_index", 0) or 0)
        except (TypeError, ValueError):
            return None
        if not gate_map or not (0 <= active < len(gate_map)):
            return None
        try:
            gate_world = np.asarray(gate_map[active]["pos"], dtype=float).reshape(3)
            if not np.isfinite(gate_world).all():
                return None
            det = best_pose_gate(data)
        except (KeyError, TypeError, ValueError):
            return None
        cam_dt = 1.0 / CAM_HZ
        return step_pnp_fusion(
            self.ekf,
            det,
            gate_world,
            cam_dt,
            max_coast=MAX_COAST_FRAMES,
            gate_floor=PNP_GATE_FLOOR,
        )

    def tick(
        self,
        snap,
        gate_map: list,
        race: dict | None = None,
        data: dict | None = None,
    ) -> np.ndarray:
        now = self._now()
        data = data or {}

        race_start = (race or {}).get("race_start_boot_time_ms", -1)
        if (
            self._last_race_start is not None
            and race_start is not None
            and race_start >= 0
            and race_start != self._last_race_start
        ):
            print("[deploy] race restart detected — reseeding + hover hold", flush=True)
            self._seed_ekf(snap, gate_map)
            self._recover_until = now + FLIP_RECOVERY_S
        if race_start is not None and race_start >= 0:
            self._last_race_start = race_start
        if snap.armed and not self._armed_prev:
            self._seed_ekf(snap, gate_map)
            self._recover_until = now + 0.5
        self._armed_prev = bool(snap.armed)

        if snap.imu is not None:
            try:
                imu_t = int(snap.imu["time_us"])
            except (KeyError, TypeError, ValueError):
                imu_t = None
            if (
                imu_t is not None
                and self._last_imu_t is not None
                and imu_t != self._last_imu_t
            ):
                dt = (imu_t - self._last_imu_t) * 1e-6
                if 0.0 < dt < 0.1:
                    gyro = _safe_extract_vec(snap.imu, ["gx", "gy", "gz"], 3)
                    if gyro is not None:
                        self.ekf.predict_commanded(self._last_applied_thrust, gyro, dt)
            if imu_t is not None:
                self._last_imu_t = imu_t

        if snap.pos_ned is not None:
            self.ekf.update_position(np.array(snap.pos_ned))
        if snap.quat is not None:
            self.ekf.update_attitude(np.array(snap.quat))

        pnp_result = self._run_pnp_fusion(data, gate_map)

        st = self.ekf.state()
        try:
            ang_vel = np.asarray(snap.ang_vel, dtype=float).reshape(3)
            if not np.isfinite(ang_vel).all():
                ang_vel = np.zeros(3)
        except (TypeError, ValueError):
            ang_vel = np.zeros(3)

        if _attitude_tilt_rad(st["q"]) > TILT_FLIP_RAD:
            if now >= self._recover_until:
                print(
                    "[deploy] flip detected — leveling (zero rates + hover)", flush=True
                )
            self._recover_until = now + FLIP_RECOVERY_S
        if now < self._recover_until:
            self._last_applied_thrust = LIVE_HOVER_THRUST
            return np.array([0.0, 0.0, 0.0, LIVE_HOVER_THRUST], dtype=np.float64)

        if self._should_engage_fallback(now, gate_map):
            if not self._in_fallback:
                print("EXPERT FALLBACK ENGAGED", flush=True)
                self._in_fallback = True
                self._accepted_pnp_frames = 0
                self._fallback.reset()

            # Track accepted PnP fusion for recovery hysteresis
            if pnp_result in ("update", "reset") and self._check_ekf_healthy():
                self._accepted_pnp_frames += 1
            elif pnp_result is not None or not self._check_ekf_healthy():
                self._accepted_pnp_frames = 0

            if self._accepted_pnp_frames >= RECOVERY_PNP_FRAMES:
                print("POLICY RESUMED", flush=True)
                self._in_fallback = False
                self._accepted_pnp_frames = 0
                self._last_gate_progress_t = now
            else:
                cmd = self._fallback.update(data, st["q"], 1.0 / 100.0)
                self._last_applied_thrust = cmd[3]
                return np.array(cmd, dtype=np.float64)

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
                    self._last_applied_thrust = LIVE_HOVER_THRUST
                    return np.array([0, 0, 0, LIVE_HOVER_THRUST], dtype=np.float64)
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
        roll, pitch, yaw, thrust = live_scale_action(action, self.meta)
        self._last_applied_thrust = thrust
        return np.array([roll, pitch, yaw, thrust], dtype=np.float64)


class PolicyRunner:
    def __init__(self, policy_path: str = POLICY_PT, device: str = "cpu"):
        self.act, self.meta = load_policy(policy_path, device=device)
        self.brain = DeployBrain(self.act, self.meta)

    def run(self):
        from rl.experts.fly2_course import resolve_gate_map
        from rl.environment.sim_interface import GATE_MAP_PATH, SimInterface

        sim = SimInterface()
        sim.data["_quiet_vision"] = True
        if not sim.wait_for_telemetry():
            print("[deploy] no telemetry — is the simulator running?")
            return

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
        gate_map = normalize_gate_map(gate_map)
        print(f"[deploy] using {len(gate_map)} gates", flush=True)
        print(
            f"[deploy] live remap: hover={LIVE_HOVER_THRUST:.2f} "
            f"rate_clip=±{LIVE_RATE_CLIP:.2f} rad/s "
            f"(ckpt trained at hover={self.meta['train_hover']:.2f} / "
            f"±{max(self.meta['action_scale']):.1f} rad/s)",
            flush=True,
        )

        snap = sim.snapshot()
        if not snap.has_pose():
            print(
                "[deploy] WARNING: no pose yet — EKF starts at origin; "
                "prefer TRAINING with odometry or wait for estimator",
                flush=True,
            )
        self.brain._seed_ekf(snap, gate_map)
        sim.arm()
        print("[deploy] armed; flying policy...", flush=True)

        while True:
            snap = sim.snapshot()
            race = sim.data.get("race_status") or {}
            cmd = self.brain.tick(snap, gate_map, race, data=sim.data)
            sim.send_attitude_rates(
                float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])
            )
            time.sleep(1.0 / 100.0)


def _selftest():
    if not os.path.exists(POLICY_PT):
        print(
            "[selftest] no policy.pt — run `uv run -m rl.training.train_ppo --quick` first"
        )
        return
    act, meta = load_policy(POLICY_PT)
    hover_a = np.array([0.0, 0.0, 0.0, 2.0 * meta["train_hover"] - 1.0])
    cmd = live_scale_action(hover_a, meta)
    assert abs(cmd[3] - LIVE_HOVER_THRUST) < 1e-6, cmd
    sat = live_scale_action(np.ones(spec.ACTION_DIM), meta)
    assert np.all(np.abs(sat[:3]) <= LIVE_RATE_CLIP + 1e-9), sat
    assert sat[3] <= LIVE_THRUST_MAX + 1e-9, sat
    print(
        f"[selftest] remap OK: ckpt hover={meta['train_hover']:.2f} -> live "
        f"{LIVE_HOVER_THRUST:.2f}, rates clipped at ±{LIVE_RATE_CLIP:.2f}"
    )

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

    _selftest_fallback(act, meta)
    _selftest_missing_gatenet()

    print("[selftest] OK — policy.pt loads, infers, and drives the loop")


def _selftest_fallback(act, meta):
    import types

    print("[selftest] testing forced-divergence fallback...", flush=True)

    gate_map = normalize_gate_map(
        [{"pos": [-10.0, 0.0, -5.0], "quat": [1, 0, 0, 0], "w": 2.72, "h": 2.72}]
    )

    t = [0.0]

    def now_fn():
        t[0] += 0.01
        return t[0]

    brain = DeployBrain(act, meta, now_fn=now_fn)

    snap = types.SimpleNamespace(
        t_mono=0.0,
        armed=True,
        pos_ned=None,
        vel_ned=None,
        quat=None,
        ang_vel=None,
        imu=None,
        frame=None,
    )
    snap.has_pose = lambda: False
    brain.ekf = ESKF(p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0.0, 0.0, 0.0]))
    brain.gate_idx = 0
    brain._last_imu_t = None
    brain._last_applied_thrust = 0.0
    brain._prev_signed = -1.0
    brain._armed_prev = True
    brain._in_fallback = False
    brain._accepted_pnp_frames = 0
    brain._last_gate_idx = 0
    brain._last_pnp_frame_id = None
    brain._last_gate_progress_t = None

    brain.ekf.P[:] = 1e6

    brain.tick(snap, gate_map, data={})
    assert brain._in_fallback, "fallback should have engaged on high covariance"
    print("[selftest] fallback engaged on divergence OK", flush=True)

    for _ in range(10):
        brain.tick(snap, gate_map, data={})
    assert brain._in_fallback, "should stay in fallback"

    # Reset EKF to healthy but NO fresh PnP — must NOT resume
    brain.ekf.reset(p0=np.zeros(3), v0=np.zeros(3))
    for _ in range(100):
        brain.tick(snap, gate_map, data={})
        if not brain._in_fallback:
            break
    assert brain._in_fallback, "should NOT resume without fresh PnP"

    # Feed RECOVERY_PNP_FRAMES valid new PnP frames with healthy EKF
    brain.ekf.reset(p0=np.array([-11.0, 0.0, -5.0]), v0=np.zeros(3))
    for i in range(RECOVERY_PNP_FRAMES + 1):
        pnp_data = {
            "pose": {
                "frame_id": 1000 + i,
                "gates": [
                    {
                        "conf": 0.9,
                        "pose": {
                            "gate_pos_body": [1.0, 0.0, 0.0],
                            "range_m": 1.0,
                            "reproj_px": 0.5,
                        },
                    }
                ],
            },
            "active_gate_index": 0,
            "frame": {"frame_id": 1000 + i},
        }
        brain.tick(snap, gate_map, data=pnp_data)
        if not brain._in_fallback:
            break
    assert not brain._in_fallback, "should have resumed after accepted PnP frames"
    print("[selftest] fallback recovery hysteresis OK", flush=True)


def _selftest_missing_gatenet():
    print("[selftest] testing missing gatenet.pt behavior...", flush=True)
    fb = FallbackBrain()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    roll, pitch, yaw, thrust = fb.update({}, q, 1.0 / 100.0)
    assert (
        math.isfinite(roll)
        and math.isfinite(pitch)
        and math.isfinite(yaw)
        and math.isfinite(thrust)
    )
    assert thrust >= LIVE_THRUST_MIN
    print("[selftest] missing gatenet.pt graceful degradation OK", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--policy", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
    else:
        cfg = load_config(args.config)
        policy_path = args.policy or cfg.policy_path
        device_str = args.device or cfg.device
        device = resolve_device(device_str)
        PolicyRunner(policy_path=policy_path, device=str(device)).run()


if __name__ == "__main__":
    main()
