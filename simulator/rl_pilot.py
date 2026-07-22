"""VQ2 RL pilot — fly the trained policy vision-only (no gate map, no odometry).

Opt-in via AUTO_PILOT=rl (make rl-flight). This is the VQ2 counterpart of the
classical GP pilot: it reuses the exact same live harness (Controller + the
shared `data` dict filled by the MAVLink/vision RX threads) and the GP pilot's
estimation + YOLO-pose vision front-end, but replaces the hand-written
guidance law with the learned policy (rl/data/policy.pt).

Observation (must match training's rl.observation exactly):
  * gate pose (to_gate_body, dist, gate_normal_body) from YOLO+PnP via
    GateEstimateSmoother — body FRD, NO gate map;
  * vel_body / ang_vel / gravity_body from GPEstimation (GyroAHRS + IMU);
  * next-gate slots ZEROED (VQ2 sees one gate at a time — trained that way);
  * frame-stacked to the policy's OBS_STACK depth.
Action: policy [-1,1]^4 -> attitude-rate + thrust, live-scaled + sign-corrected,
sent via controller.set_attitude_rates.

CALIBRATION (live-tunable — the sim's rate sign/gain can only be confirmed in
flight). Env vars, all optional:
  RL_SIGN_ROLL / RL_SIGN_PITCH / RL_SIGN_YAW  (default +1/+1/+1 — the live RATE
                  path is NOT inverted, measured 2026-07-21)
  RL_RATE_SCALE   extra gain on commanded rates (default 0.4 — the live plant
                  amplifies rates ~3x; start gentle, raise if sluggish)
  RL_THRUST_TRIM  additive thrust trim (default 0.0)
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from enum import Enum, auto

import numpy as np

from rl.deploy import (
    LIVE_HOVER_THRUST,
    LIVE_RATE_CLIP,
    LIVE_THRUST_MAX,
    LIVE_THRUST_MIN,
    live_scale_action,
    load_policy,
)
from rl.observation import build_observation_body
from simulator.gp_estimation import GPEstimation
from simulator.gp_vision import GateEstimateSmoother, VisionVelocityTracker

RL_CONTROL_HZ = 50  # matches the policy's trained DECISION_HZ (< 100 Hz cap)
VEL_CORRECT_K = 0.15  # vision-velocity anchor gain in the complementary filter
VEL_CLIP_MPS = 8.0  # hard bound on the fused body velocity (safety)
# Cap on the rotational-flow correction (omega x gate_body). At a far gate + high
# spin it can blow up to 100+ m/s and CORRUPT the velocity it's meant to clean
# (live: vrot hit 163). A real turn near a gate is only a few m/s; clamp it.
VROT_CAP_MPS = 6.0
# When no drift-free vision velocity is available, the raw IMU vel_body dead-
# reckons to +-50 m/s within seconds. Don't free-run it: hold the last fused
# value a few ticks, then decay toward zero so a lost-lock window can't pin the
# velocity obs at the clip (which is itself wildly out-of-distribution).
BLIND_VEL_HOLD_TICKS = 3
VEL_DECAY = 0.9
# After this many control ticks with no fresh gate, stop repeating the last
# command and coast level (zero rates, hover thrust). ~1.6 ticks is normal
# between 30 Hz vision frames; a long gap = near-gate blind window / lost lock,
# where holding an open-loop bank would drive into the gate edge.
BLIND_HOLD_TICKS = 3

ARM_RETRY_S = 1.0
CLOCK_RESET_SLACK_MS = 500
IMU_FROZEN_S = 2.0
DISARM_PERSIST_S = 1.0
DEBUG_EVERY_N = 10  # ~5 Hz at 50 Hz control — dense enough to diagnose a crash


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Phase(Enum):
    WAIT_FOR_DATA = auto()
    WAIT_FOR_START = auto()
    FLYING = auto()


class RLPilot:
    """Learned-policy pilot for the VQ2 vision-only course."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        # Silence the per-frame "[vision] GATE cx=..." HSV spam so the [rl]
        # debug lines are readable (keeps the acquired/lost one-liners).
        data["_quiet_vision"] = True
        self.act, self.meta = load_policy()
        self._obs_stack = int(self.meta.get("obs_stack", 1))
        self.est = GPEstimation(data)
        self.gate_smoother = GateEstimateSmoother()
        self.vel_tracker = VisionVelocityTracker()

        # Rate-command signs. Live-measured 2026-07-21: the RATE path is NOT
        # inverted (sent -0.24 roll rate -> roll went negative; policy's
        # +bank-right must send +rate), so all +1. The -1/+1/-1 that flew the
        # drone into a left roll-runaway was the GP pilot's ATTITUDE-ANGLE
        # convention (KR=KY=-1) wrongly applied to rates.
        self._signs = np.array(
            [
                _env_float("RL_SIGN_ROLL", 1.0),
                _env_float("RL_SIGN_PITCH", 1.0),
                _env_float("RL_SIGN_YAW", 1.0),
            ]
        )
        # 0.3 ~= 1/3.2: cancels the live plant's measured ~3x rate gain (roll
        # ran 0.77 rad/s from a 0.24 cmd) so the drone rotates at the rate the
        # policy INTENDS. At 0.4 every maneuver overshot ~1.2x (yaw drifted to
        # -44deg on approach, roll wobbled +-15deg) and it missed gate 1 by a hair.
        self._rate_scale = _env_float("RL_RATE_SCALE", 0.3)
        self._thrust_trim = _env_float("RL_THRUST_TRIM", 0.0)

        self._frames: deque = deque(maxlen=self._obs_stack)
        self._last_vis_fid = None  # advance the stack only on a NEW vision frame
        self._cmd: tuple | None = None  # last command, repeated between frames
        self._stale_ticks = 0  # control ticks since the last fresh gate
        self._vel_fused = np.zeros(3)  # complementary-filtered body velocity
        self._v_imu_prev: np.ndarray | None = None
        self._vel_blind_ticks = 0
        self._vis_rot_mag = 0.0  # last rotational-flow correction magnitude (debug)
        self.last_action = np.zeros(4)
        self.n_passed = 0

        self.phase = Phase.WAIT_FOR_DATA
        self._tick = 0
        self._est_started = False
        self._last_arm_attempt = 0.0
        self._wait_start_sim_ms: int | None = None
        self._go_start_ms: int | None = None
        self._finish_noted = False
        self._imu_ts_seen = None
        self._imu_ts_wall = 0.0
        self._frozen_noted = False
        self._disarm_since = None
        self._debug = os.environ.get("GP_DEBUG", "").strip() in ("1", "true", "yes")

        controller.control_hz = RL_CONTROL_HZ
        controller.set_control_mode("attitude")  # body-rate + thrust path
        controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
        print(
            f"[rl] RL policy pilot ready (make rl-flight): obs_stack={self._obs_stack} "
            f"signs={tuple(self._signs)} rate_scale={self._rate_scale} "
            f"train_hover={self.meta['train_hover']:.3f}",
            flush=True,
        )

    # ---- harness API ----------------------------------------------------
    @property
    def gates_passed(self) -> int:
        return self.n_passed

    def on_attempt_start(self) -> None:
        self._reset_state()

    def reset_for_attempt(self) -> None:
        self._reset_state()
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)

    def shutdown(self) -> None:
        self.est.stop()

    def _reset_state(self) -> None:
        self.n_passed = 0
        self.phase = Phase.WAIT_FOR_DATA
        self._frames.clear()
        self._last_vis_fid = None
        self._cmd = None
        self._stale_ticks = 0
        self._vel_fused = np.zeros(3)
        self._v_imu_prev = None
        self._vel_blind_ticks = 0
        self.last_action[:] = 0.0
        self.est.reset()
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._tick = 0
        self._last_arm_attempt = 0.0
        self._wait_start_sim_ms = None
        self._go_start_ms = None
        self._finish_noted = False
        self._imu_ts_seen = None
        self._imu_ts_wall = 0.0
        self._frozen_noted = False
        self._disarm_since = None

    # ---- physics-live guard (mirrors GPPilot) ---------------------------
    def _physics_live(self, imu) -> bool:
        now = time.time()
        if imu is not None:
            ts = imu.get("time_us") or imu.get("time_usec")
            if ts != self._imu_ts_seen:
                self._imu_ts_seen = ts
                self._imu_ts_wall = now
                self._frozen_noted = False
        live = self._imu_ts_wall > 0.0 and now - self._imu_ts_wall <= IMU_FROZEN_S
        if not live and self._imu_ts_wall > 0.0 and not self._frozen_noted:
            print(
                "[rl] Sim physics IDLE (IMU clock frozen) — click Restart Race.",
                flush=True,
            )
            self._frozen_noted = True
        return live

    # ---- observation from vision + estimation ---------------------------
    def _gravity_body(self, quat) -> np.ndarray:
        w, x, y, z = quat
        return np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )

    @staticmethod
    def _plausible(v: dict) -> bool:
        """Reject physically-implausible gate estimates. Live YOLO+PnP
        intermittently emits garbage (gate 70 m away, 57 m to the side) after a
        lost/reacquire; one such frame saturates the policy. The real next gate
        is a few metres ahead and roughly centred."""
        bx = v.get("body_x_m")
        by = v.get("body_y_m")
        bz = v.get("body_z_m")
        if bx is None or by is None or bz is None:
            return False
        if not all(math.isfinite(x) for x in (bx, by, bz)):
            return False
        return 0.2 < bx < 20.0 and abs(by) < 12.0 and abs(bz) < 12.0

    def _update_velocity(self, snap: dict, gate_body: np.ndarray | None) -> None:
        """Complementary filter for body velocity. Applies the IMU velocity's
        per-tick CHANGE (accurate short-term even as the absolute drifts), then
        anchors to the drift-free vision velocity so the estimate stays bounded
        and grounded instead of diverging like raw IMU dead-reckoning.

        The vision velocity is ROTATION-COMPENSATED. VisionVelocityTracker
        computes -Delta(gate_body)/dt, but gate_body rotates with the drone even
        with zero translation: v_body = -Delta(gate_body)/dt - omega x gate_body.
        The tracker omits the omega x gate_body term, so during a turn (gate 2's
        ~40deg yaw) it reports several m/s of PHANTOM lateral velocity at gate
        range — the out-of-distribution velocity that stalls the policy exactly
        when it turns. Subtract it here to recover the true translational vel."""
        v_imu = np.asarray(snap["vel_body"], float)
        vv = self.vel_tracker.last_velocity
        if vv is not None:
            # Drift-free vision velocity available: bridge with the IMU's
            # short-term CHANGE, then anchor to the (rotation-corrected) vision.
            v_vis = np.array(
                [vv["vx_body_mps"], vv["vy_body_mps"], vv["vz_body_mps"]], float
            )
            if gate_body is not None:
                omega = np.radians(np.asarray(snap["rates_body_dps"], float))
                v_rot = np.cross(omega, gate_body)  # rotational optical flow
                m = float(np.linalg.norm(v_rot))
                self._vis_rot_mag = m
                if m > VROT_CAP_MPS:  # clamp so a fast spin can't explode it
                    v_rot = v_rot * (VROT_CAP_MPS / m)
                v_vis = v_vis - v_rot
            if self._v_imu_prev is not None:
                self._vel_fused = self._vel_fused + (v_imu - self._v_imu_prev)
            self._vel_fused += VEL_CORRECT_K * (v_vis - self._vel_fused)
            self._vel_blind_ticks = 0
        else:
            # No drift-free anchor: the IMU alone diverges, so hold briefly then
            # decay toward zero rather than free-running it.
            self._vel_blind_ticks += 1
            if self._vel_blind_ticks > BLIND_VEL_HOLD_TICKS:
                self._vel_fused = self._vel_fused * VEL_DECAY
        self._v_imu_prev = v_imu
        self._vel_fused = np.clip(self._vel_fused, -VEL_CLIP_MPS, VEL_CLIP_MPS)

    def _build_frame(self, vision: dict, snap: dict) -> np.ndarray:
        """One 24-D observation frame from a body-relative gate estimate + IMU
        state. Called only on a fresh (non-None) vision estimate."""
        gate_body = np.array(
            [vision["body_x_m"], vision["body_y_m"], vision["body_z_m"]], float
        )
        dist = float(np.linalg.norm(gate_body))
        nb = vision.get("normal_body")
        # Trust the vision normal ONLY when it's a real full-PnP solve. The
        # "edge-pair" fallback emits a fixed camera-tilt placeholder normal
        # (~20deg up), and anduril/HSV give no normal at all — both would feed
        # the policy an out-of-distribution gate_normal (training only ever saw
        # a UNIT travel-axis). In those cases aim straight through the gate.
        measured = (
            nb is not None
            and np.all(np.isfinite(nb))
            and vision.get("method") not in (None, "edge-pair")
        )
        # Vision normal is BACK-facing (points at the drone); training's is the
        # TRAVEL axis — negate. build_observation_body stores gate_normal
        # verbatim (no auto-unit), so normalize here to match training's unit.
        normal_travel = -np.asarray(nb, float) if measured else gate_body
        n = float(np.linalg.norm(normal_travel))
        normal_travel = normal_travel / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        frame = build_observation_body(
            to_gate_body=gate_body,
            dist_to_gate=dist,
            gate_normal_body=normal_travel,
            # Complementary-filtered body velocity: raw IMU dead-reckoning
            # diverges to 10s of m/s; this anchors it to the drift-free vision
            # velocity so the policy gets a bounded, grounded estimate.
            vel_body=self._vel_fused,
            ang_vel=np.radians(np.asarray(snap["rates_body_dps"], float)),
            gravity_body=self._gravity_body(snap["quat"]),
            to_next_gate_body=np.zeros(3),  # VQ2: one gate at a time (trained)
            dist_to_next_gate=0.0,
            last_action=self.last_action[:3],
        )
        return frame

    def _stacked(self, frame: np.ndarray) -> np.ndarray:
        if not self._frames:
            self._frames.extend([frame] * self._obs_stack)
        else:
            self._frames.append(frame)
        return np.concatenate(self._frames, dtype=np.float32)

    def _policy_cmd(self, obs: np.ndarray) -> tuple:
        """Policy obs -> live attitude-rate + thrust command (signed, clipped)."""
        action = self.act(obs)
        self.last_action = action
        roll, pitch, yaw, thrust = live_scale_action(action, self.meta)
        rates = self._signs * self._rate_scale * np.array([roll, pitch, yaw])
        rates = np.clip(rates, -LIVE_RATE_CLIP, LIVE_RATE_CLIP)
        thrust = float(
            np.clip(thrust + self._thrust_trim, LIVE_THRUST_MIN, LIVE_THRUST_MAX)
        )
        return float(rates[0]), float(rates[1]), float(rates[2]), thrust

    # ---- main tick ------------------------------------------------------
    def tick(self) -> None:
        self._tick += 1
        armed = bool(self.data.get("armed", False))
        imu = self.data.get("imu")
        race = self.data.get("race_status")
        physics_live = self._physics_live(imu)

        if self.phase == Phase.WAIT_FOR_DATA:
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
            if not armed:
                now = time.time()
                if now - self._last_arm_attempt >= ARM_RETRY_S:
                    print("[rl] arming...", flush=True)
                    self.controller.arm()
                    self._last_arm_attempt = now
            elif imu is not None:
                if not self._est_started:
                    self.est.start()
                    self._est_started = True
                print("[rl] armed + IMU ready -> WAIT_FOR_START", flush=True)
                self.phase = Phase.WAIT_FOR_START
            return

        if self.phase == Phase.WAIT_FOR_START:
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
            if not armed:
                self.phase = Phase.WAIT_FOR_DATA
                return
            if race is not None:
                sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
                start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
                finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                if sim_ms < self._wait_start_sim_ms - CLOCK_RESET_SLACK_MS:
                    self._reset_state()
                    return
                race_fresh = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_fresh and sim_ms >= start_ms and finish_ns < 0
                if countdown_done and physics_live:
                    self._enter_flying(start_ms)
                elif start_ms > 0 and finish_ns >= 0 and not self._finish_noted:
                    print("[rl] Race FINISHED — click Restart Race.", flush=True)
                    self._finish_noted = True
            return

        # FLYING
        if not armed:
            now = time.time()
            if self._disarm_since is None:
                self._disarm_since = now
            elif now - self._disarm_since >= DISARM_PERSIST_S:
                print("[rl] disarmed — re-arming.", flush=True)
                self._reset_state()
                return
        else:
            self._disarm_since = None

        if race is not None and self._should_abort(race):
            return

        if not self._est_started and imu is not None:
            self.est.start()
            self._est_started = True

        snap = self.est.snapshot()

        # Gate counter (for reporting only). active_gate_index survives VQ2 and
        # is monotonic. Race END is handled by _should_abort on
        # race_finish_time_ns — no hardcoded gate count (the VQ2 course length
        # isn't known here, and a wrong constant would either stop early or
        # never trigger).
        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self.n_passed:
            self.n_passed = active

        vision = self.gate_smoother.update(self.data)
        self.vel_tracker.update(vision)
        # Current gate vector (body FRD) for the rotation-flow correction; only
        # when the estimate is physically plausible (garbage range would inflate
        # the omega x gate_body term).
        gate_body = None
        if vision is not None and self._plausible(vision):
            gate_body = np.array(
                [vision["body_x_m"], vision["body_y_m"], vision["body_z_m"]], float
            )
        self._update_velocity(snap, gate_body)  # keep the fused body velocity current
        # Decide only on a fresh, RELIABLE vision frame (YOLO ~30 Hz). The
        # smoother returns the identical estimate between camera frames (dedups
        # on frame_id), so re-stacking every 50 Hz tick would fill the stack
        # with duplicates and zero out the gate-relative derivatives. HSV
        # fallbacks (reliable=False) carry noisy pinhole depth + no real
        # normal — don't servo on them, hold instead.
        fid = vision.get("frame_id") if vision is not None else None
        fresh = (
            vision is not None
            and bool(vision.get("reliable"))
            and self._plausible(vision)
            and fid != self._last_vis_fid
        )
        if fresh:
            self._stale_ticks = 0
            self._last_vis_fid = fid
            obs = self._stacked(self._build_frame(vision, snap))
            self._cmd = self._policy_cmd(obs)
        else:
            # No fresh gate this tick. Repeat the last command for a couple of
            # ticks (normal between 30 Hz frames), but NEVER hold an open-loop
            # bank through a long blind window (near-gate suppression / lost
            # lock ~0.3 s) — after BLIND_HOLD_TICKS coast level: zero rates,
            # hover thrust, so it crosses the gate plane without banking into
            # the edge or dropping.
            self._stale_ticks += 1
            if self._cmd is None or self._stale_ticks > BLIND_HOLD_TICKS:
                self._cmd = (0.0, 0.0, 0.0, LIVE_HOVER_THRUST)
        self.controller.set_attitude_rates(*self._cmd)

        if self._debug and self._tick % DEBUG_EVERY_N == 0:
            v = vision or {}
            vb = np.asarray(snap["vel_body"], float)
            vf = self._vel_fused
            print(
                f"[rl] passed={self.n_passed} src={v.get('source', 'blind')} "
                f"gate(bx,by,bz)=({v.get('body_x_m', float('nan')):+.1f},"
                f"{v.get('body_y_m', float('nan')):+.1f},"
                f"{v.get('body_z_m', float('nan')):+.1f}) "
                f"VELfused(f,r,d)=({vf[0]:+.1f},{vf[1]:+.1f},{vf[2]:+.1f}) "
                f"VELimu_f={vb[0]:+.1f} vrot={self._vis_rot_mag:.1f} "
                f"act=[{self.last_action[0]:+.2f},{self.last_action[1]:+.2f},"
                f"{self.last_action[2]:+.2f},{self.last_action[3]:+.2f}]",
                flush=True,
            )

    # ---- flight entry / abort (mirror GPPilot) --------------------------
    def _enter_flying(self, start_ms: int) -> None:
        print("[rl] Countdown complete — flying policy!", flush=True)
        self._go_start_ms = start_ms
        self._finish_noted = False
        self.phase = Phase.FLYING
        self.est.reset()
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._frames.clear()
        self._last_vis_fid = None
        self._cmd = None
        self._stale_ticks = 0
        self._vel_fused = np.zeros(3)
        self._v_imu_prev = None
        self._vel_blind_ticks = 0
        self.n_passed = 0  # fresh lap — every GO is a new attempt
        self.last_action[:] = 0.0

    def _abort_to_wait(self, reason: str) -> None:
        print(f"[rl] {reason} — holding for countdown", flush=True)
        self.phase = Phase.WAIT_FOR_START
        self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
        self.est.reset()
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._frames.clear()
        self._last_vis_fid = None
        self._cmd = None
        self._vel_fused = np.zeros(3)
        self._v_imu_prev = None
        self._vel_blind_ticks = 0
        self._go_start_ms = None
        self._finish_noted = False
        race = self.data.get("race_status")
        self._wait_start_sim_ms = (
            int(race.get("sim_boot_time_ms", 0) or 0) if race is not None else None
        )

    def _should_abort(self, race: dict) -> bool:
        sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
        start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
        finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
        if (
            self._wait_start_sim_ms is not None
            and sim_ms < self._wait_start_sim_ms - CLOCK_RESET_SLACK_MS
        ):
            self._abort_to_wait("Sim clock reset mid-flight")
            return True
        if start_ms > 0 and start_ms > sim_ms:
            self._abort_to_wait("New race countdown")
            return True
        if (
            start_ms > 0
            and self._go_start_ms is not None
            and start_ms > self._go_start_ms
            and finish_ns < 0
        ):
            self._abort_to_wait("Restart Race")
            return True
        if finish_ns >= 0:
            self._abort_to_wait("Race finished")
            return True
        return False
