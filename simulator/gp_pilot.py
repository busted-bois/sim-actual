"""AndurilGP guidance pilot — bearing bank + elev thrust + GyroAHRS est.

Opt-in via AUTO_PILOT=gp. Same rate+thrust action interface as IBVS.
Action space matches rl/spec.py for later expert / RL merge.
"""

from __future__ import annotations

import math
import os
import time
from enum import Enum, auto

import numpy as np

from simulator.gp_estimation import GPEstimation
from simulator.gp_vision import (
    VisionVelocityTracker,
    gate_tilt_deg_from_normal,
    vision_gate_estimate,
)

HOVER_THRUST = 0.264
DESIRED_PITCH_DEG = -3.0
K_BEARING = 4.5
K_LAT_D = 9.0
MAX_BANK_DEG = 25.0
PERP_BLEND_DIST = 6.0
TILT_EMA_ALPHA = 0.25
K_P_THRUST = 0.014
K_D_THRUST = 0.0175
BEARING_RATE_CLAMP_DEG_S = 60.0
ELEV_RATE_CLAMP_M_S = 5.0
MIN_BX_FOR_ELEV = 3.0
VIS_VEL_EMA_ALPHA = 0.35
OF_ALPHA = 0.6
KP, KR, KY = 1.0, -1.0, -1.0
DEBUG_EVERY_N = 45  # ~2 Hz at 90 Hz
ARM_RETRY_S = 1.0


class Phase(Enum):
    WAIT_FOR_DATA = auto()
    WAIT_FOR_START = auto()
    FLYING = auto()


def compute_guidance(
    *,
    roll_deg: float,
    pitch_deg: float,
    quat: np.ndarray,
    vY: float,
    vD: float,
    vision: dict | None,
    vision_vel: dict | None,
    state: dict,
) -> tuple[float, float, float, float, dict]:
    """Anduril FLYING guidance. Mutates `state`. Returns rates (rad/s) + thrust."""
    vision_valid = False
    bx = by = bz = float("nan")
    vis_frame_id = None
    if vision is not None:
        bx = float(vision.get("body_x_m", float("nan")))
        by = float(vision.get("body_y_m", float("nan")))
        bz = float(vision.get("body_z_m", float("nan")))
        vis_frame_id = vision.get("frame_id")
        if not any(math.isnan(v) for v in (bx, by, bz)) and bx > 0.1:
            vision_valid = True

    elev_rate = 0.0
    if vision_valid and bx > MIN_BX_FOR_ELEV and vis_frame_id is not None:
        qw, qx, qy, qz = quat
        gate_pD = (
            2 * (qx * qz - qw * qy) * bx
            + 2 * (qy * qz + qw * qx) * by
            + (1 - 2 * (qx * qx + qy * qy)) * bz
        )
        prev_fid = state.get("prev_elev_frame_id")
        if prev_fid is not None and 0 < vis_frame_id - prev_fid <= 3:
            dt_e = (vis_frame_id - prev_fid) / 30.0
            elev_rate = (gate_pD - state["prev_gate_pD"]) / dt_e
        state["prev_gate_pD"] = gate_pD
        state["prev_elev_frame_id"] = vis_frame_id
        state["last_elev_err"] = gate_pD
    elif not vision_valid:
        state["prev_gate_pD"] = None
        state["prev_elev_frame_id"] = None

    # Vision-IMU velocity fusion (lateral + body-down only; vX stays IMU).
    if (
        vision_vel is not None
        and vis_frame_id is not None
        and vis_frame_id != state.get("last_fused_frame_id")
    ):
        vy_raw = float(vision_vel["vy_body_mps"])
        vz_raw = float(vision_vel["vz_body_mps"])
        state["vision_vy_ema"] = (
            VIS_VEL_EMA_ALPHA * vy_raw
            + (1.0 - VIS_VEL_EMA_ALPHA) * state["vision_vy_ema"]
        )
        state["vision_vz_ema"] = (
            VIS_VEL_EMA_ALPHA * vz_raw
            + (1.0 - VIS_VEL_EMA_ALPHA) * state["vision_vz_ema"]
        )
        vY = OF_ALPHA * vY + (1.0 - OF_ALPHA) * state["vision_vy_ema"]
        state["last_fused_frame_id"] = vis_frame_id

    bearing_body = 0.0
    blend = 0.0
    yaw_err = 0.0
    bearing_rate = 0.0
    gate_tilt_deg = float("nan")

    if vision_valid:
        bearing_body = float(np.clip(math.degrees(math.atan2(by, bx)), -25.0, 25.0))
        blend = float(np.clip(bx / PERP_BLEND_DIST, 0.0, 1.0))

        prev_bf = state.get("prev_bearing_frame_id")
        if (
            vis_frame_id is not None
            and prev_bf is not None
            and 0 < vis_frame_id - prev_bf <= 3
        ):
            dt_b = (vis_frame_id - prev_bf) / 30.0
            bearing_rate = (bearing_body - state["prev_bearing_body"]) / dt_b
        if vis_frame_id is not None:
            state["prev_bearing_body"] = bearing_body
            state["prev_bearing_frame_id"] = vis_frame_id

        nb = vision.get("normal_body")
        if nb is not None:
            tilt_raw = gate_tilt_deg_from_normal(nb)
            ema = state.get("gate_tilt_ema")
            state["gate_tilt_ema"] = (
                tilt_raw
                if ema is None
                else TILT_EMA_ALPHA * tilt_raw + (1.0 - TILT_EMA_ALPHA) * ema
            )
            gate_tilt_deg = state["gate_tilt_ema"]
        else:
            state["gate_tilt_ema"] = None

        yaw_bearing = float(np.clip(bearing_body, -12.0, 12.0))
        if not math.isnan(gate_tilt_deg):
            yaw_err = blend * yaw_bearing + (1.0 - blend) * gate_tilt_deg
        else:
            yaw_err = yaw_bearing
    else:
        state["gate_tilt_ema"] = None
        state["prev_bearing_body"] = None
        state["prev_bearing_frame_id"] = None
        bearing_rate = 0.0

    is_new_d = (
        vision_valid
        and vis_frame_id is not None
        and vis_frame_id != state.get("last_d_frame_id")
    )
    if is_new_d:
        br_c = float(
            np.clip(bearing_rate, -BEARING_RATE_CLAMP_DEG_S, BEARING_RATE_CLAMP_DEG_S)
        )
        d_lat = -math.radians(br_c) * max(bx, 0.5)
        er_c = float(np.clip(elev_rate, -ELEV_RATE_CLAMP_M_S, ELEV_RATE_CLAMP_M_S))
        d_vert = -er_c
        state["vY_at_vision"] = vY
        state["vD_at_vision"] = vD
        state["last_d_frame_id"] = vis_frame_id
    elif not vision_valid:
        d_lat = 0.0
        d_vert = 0.0
        state["vY_at_vision"] = vY
        state["vD_at_vision"] = vD
        state["last_d_frame_id"] = None
    else:
        d_lat = vY - state["vY_at_vision"]
        d_vert = vD - state["vD_at_vision"]

    pitch_cmd_deg = (DESIRED_PITCH_DEG - pitch_deg) * KP
    p_lat = K_BEARING * bearing_body * blend
    d_lat_term = K_LAT_D * d_lat * blend
    desired_roll = float(np.clip(p_lat - d_lat_term, -MAX_BANK_DEG, MAX_BANK_DEG))
    roll_cmd_deg = (desired_roll - roll_deg) * KR
    yaw_cmd_deg = yaw_err * KY

    tilt = max(
        0.01,
        math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg)),
    )
    elev_err = float(state.get("last_elev_err", 0.0))
    thrust = (HOVER_THRUST - elev_err * K_P_THRUST + d_vert * K_D_THRUST) / tilt
    thrust = float(np.clip(thrust, 0.0, 1.0))

    # Anduril encoded deg commands via quat→sim quirk; we send true rad/s.
    roll_rate = math.radians(roll_cmd_deg)
    pitch_rate = math.radians(pitch_cmd_deg)
    yaw_rate = math.radians(yaw_cmd_deg)

    dbg = {
        "bearing_deg": bearing_body,
        "blend": blend,
        "elev_err": elev_err,
        "desired_roll": desired_roll,
        "d_lat": d_lat,
        "d_vert": d_vert,
        "vision_valid": vision_valid,
        "bx": bx,
        "by": by,
        "bz": bz,
    }
    return roll_rate, pitch_rate, yaw_rate, thrust, dbg


def _fresh_hold_state() -> dict:
    return {
        "last_elev_err": 0.0,
        "last_fused_frame_id": None,
        "gate_tilt_ema": None,
        "last_d_frame_id": None,
        "vY_at_vision": 0.0,
        "vD_at_vision": 0.0,
        "vision_vy_ema": 0.0,
        "vision_vz_ema": 0.0,
        "prev_bearing_body": None,
        "prev_bearing_frame_id": None,
        "prev_gate_pD": None,
        "prev_elev_frame_id": None,
    }


class GPPilot:
    """AndurilGP estimation + guidance (WAIT_FOR_* phases like their Controller)."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self.n_passed = 0
        self.phase = Phase.WAIT_FOR_DATA
        self.est = GPEstimation(data)
        self.vel_tracker = VisionVelocityTracker()
        self._hold = _fresh_hold_state()
        self._tick = 0
        self._est_started = False
        self._last_arm_attempt = 0.0
        self._wait_start_sim_ms = None
        self._debug = os.environ.get("GP_DEBUG", "").strip() in ("1", "true", "yes")
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
        print("[gp] AndurilGP controls pilot ready (make auto-gp)", flush=True)

    @property
    def gates_passed(self) -> int:
        return self.n_passed

    def on_attempt_start(self) -> None:
        self._reset_state()

    def reset_for_attempt(self) -> None:
        self._reset_state()
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)

    def _reset_state(self) -> None:
        self.n_passed = 0
        self.phase = Phase.WAIT_FOR_DATA
        self._hold = _fresh_hold_state()
        self.vel_tracker.reset()
        self.est.reset()
        self._tick = 0
        self._last_arm_attempt = 0.0
        self._wait_start_sim_ms = None

    def tick(self) -> None:
        self._tick += 1
        armed = bool(self.data.get("armed", False))
        imu = self.data.get("imu")
        race = self.data.get("race_status")

        if self.phase == Phase.WAIT_FOR_DATA:
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
            if not armed:
                now = time.time()
                if now - self._last_arm_attempt >= ARM_RETRY_S:
                    print("Sending arm command...", flush=True)
                    self.controller.arm()
                    self._last_arm_attempt = now
            elif imu is not None:
                if not self._est_started:
                    self.est.start()
                    self._est_started = True
                print("Armed and IMU ready. Moving to WAIT_FOR_START.", flush=True)
                self.phase = Phase.WAIT_FOR_START
            return

        if self.phase == Phase.WAIT_FOR_START:
            # Hold on pad — zero thrust until race GO (AndurilGP).
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
            if race is not None:
                sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
                start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                    print(f"[WAIT] Anchor set: sim_ms={sim_ms}", flush=True)
                race_fresh = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                go = race_fresh and sim_ms >= start_ms
                if self._debug and self._tick % DEBUG_EVERY_N == 0:
                    print(
                        f"[WAIT] sim_ms={sim_ms} race_start={start_ms} "
                        f"fresh={race_fresh} go={go}",
                        flush=True,
                    )
                if go:
                    print("Countdown complete! Flying!", flush=True)
                    self.phase = Phase.FLYING
            elif self._debug and self._tick % DEBUG_EVERY_N == 0:
                print("[WAIT] No race_status yet — holding...", flush=True)
            return

        # FLYING — Anduril guidance
        if not self._est_started and imu is not None:
            self.est.start()
            self._est_started = True

        snap = self.est.snapshot()
        roll_deg, pitch_deg, _yaw_deg = snap["att_deg"]
        vY = float(snap["vel_body"][1])
        vD = float(snap["vel_ned"][2])

        vision = vision_gate_estimate(self.data)
        vision_vel = self.vel_tracker.update(vision)

        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self.n_passed:
            self.n_passed = active

        roll_r, pitch_r, yaw_r, thrust, dbg = compute_guidance(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            quat=snap["quat"],
            vY=vY,
            vD=vD,
            vision=vision,
            vision_vel=vision_vel,
            state=self._hold,
        )
        self.controller.set_attitude_rates(roll_r, pitch_r, yaw_r, thrust)

        if self._debug and self._tick % DEBUG_EVERY_N == 0:
            print(
                f"[gp] att=({roll_deg:+.1f}r {pitch_deg:+.1f}p) "
                f"gate=({dbg['bx']:+.1f},{dbg['by']:+.1f},{dbg['bz']:+.1f}) "
                f"blend={dbg['blend']:.2f} elev={dbg['elev_err']:+.2f} "
                f"T={thrust:.3f}",
                flush=True,
            )

    def shutdown(self) -> None:
        self.est.stop()
