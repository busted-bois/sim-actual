"""AndurilGP guidance pilot — bearing bank + elev thrust + GyroAHRS est.

Opt-in via AUTO_PILOT=gp (make control-flight). Same rate+thrust action
interface as IBVS. Action space matches rl/spec.py for later expert / RL merge.

Smooth-flight additions on top of the original AndurilGP port:
  * closed-loop forward speed (~8 km/h cruise) via speed-PD -> pitch lean,
  * vision smoothing (gate-position EMA, PnP-sticky source selection) and
    attitude-command slew limiting against YOLO frame flicker,
  * collision backoff: reverse a few meters after a hit, then re-acquire.
"""

from __future__ import annotations

import csv
import math
import os
import time
from enum import Enum, auto

import numpy as np

from simulator.gp_estimation import GPEstimation
from simulator.gp_vision import (
    GateEstimateSmoother,
    VisionVelocityTracker,
    gate_tilt_deg_from_normal,
)

# Anduril's own measured trim, byte-faithful to the original that flew the
# course (their controller.py hardcoded 0.264). The RL stack keeps using
# spec.HOVER_THRUST — rl.gp_expert passes its own hover_thrust in.
HOVER_THRUST = 0.264
# Original AndurilGP command rate (2:1 with the 30 Hz camera; spec cap 100).
GP_CONTROL_HZ = 60
DESIRED_PITCH_DEG = -2.0
K_BEARING = 2.5
K_LAT_D = 5.0
MAX_BANK_DEG = 14.0
PERP_BLEND_DIST = 6.0
# Near-gate floor on lateral blend. blend=bx/PERP_BLEND_DIST shrinks to 0 as
# we close, which starved bank authority exactly when off-center (CSV seg11:
# by grew to -1.5 m while blend cut P/D). Keep enough bank to finish centering.
NEAR_LAT_BLEND_FLOOR = 0.75
BEARING_NEED_BANK_DEG = 3.0  # |bearing| above this → apply the floor
# Aim between current and next gate: p_la = p_cur + λ (p_next - p_cur).
# Δ is expressed in the *current gate* frame (map quat), not AHRS — GyroAHRS
# yaw is unreferenced and was rotating NED Δ into random body axes (CSV:
# bx collapsed by ~λ|Δx| → wild bank). Vision-aligned: gate frame ≈ body.
# Default OFF: live logs showed map gate axes put ~24 m along-track onto
# "right", yanking body-y by ~λ·24. Code path kept for λ>0 experiments.
LOOKAHEAD_LAMBDA = 0.0
LOOKAHEAD_OFFSET_MAX_M = 2.0  # clamp λ·lateral / λ·vert when λ>0
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
CLOCK_RESET_SLACK_MS = 500
IMU_FROZEN_S = 2.0  # sensor clock stuck this long => sim physics is idle
DISARM_PERSIST_S = 1.0  # ignore 1 Hz heartbeat armed-flag blips mid-flight

# Closed-loop forward speed. Hard cap 10 km/h — always-on IMU regulation
# (not gated on vision) so lost-lock cannot open-loop dive to 20–30 km/h.
MAX_SPEED_MPS = 10.0 / 3.6  # ≈2.78 m/s
CRUISE_SPEED_MPS = 2.2
THRU_SPEED_MPS = 1.2  # near-gate / weak-detection crawl
BLIND_CRAWL_MPS = 1.0  # no gate in view
SLOWDOWN_START_M = 5.0
K_SPEED_P = 2.5  # deg of pitch lean per m/s of speed error
K_SPEED_D = 0.6  # deg per m/s^2 damping on forward speed
PITCH_DES_MIN_DEG = -2.5
PITCH_DES_MAX_DEG = 6.0
PITCH_WIRE_MAX_DEG = 18.0  # clamp on attitude-quat pitch command

# Attitude-command slew limits (vision flicker → bank twitch).
CMD_SLEW_DEG_S = 90.0
THRUST_SLEW_PER_S = 1.0

# Collision backoff: reverse only as far as needed to re-acquire the gate.
#
# The reverse lean is TIME-SCHEDULED (push → coast → brake), not purely
# regulated on vX: the IMU strapdown is dead-reckoning only, so vX is
# untrustworthy right after an impact. The schedule bounds reverse speed even
# when vX is stuck at zero and the AHRS is biased — the failure mode that used
# to reverse at 20–30 km/h.
BACKOFF_MAX_SPEED_MPS = 0.8  # ~2.9 km/h regulated target
BACKOFF_PITCH_DEG = 4.0  # push-off nose-up lean
BACKOFF_BRAKE_PITCH_DEG = -4.0  # nose-down, arrest reverse before FLYING
BACKOFF_PITCH_WIRE_MAX_DEG = 5.0  # hard clamp on the attitude-quat command
BACKOFF_PITCH_AUTH_DEG = 1.0  # how far the AHRS may trim the scheduled lean
BACKOFF_PUSH_S = 1.2  # full lean window
BACKOFF_COAST_S = 1.6  # lean decays linearly to zero
BACKOFF_BRAKE_S = 1.2  # brake window
BACKOFF_MAX_S = 4.0  # == PUSH + COAST + BRAKE (load-bearing invariant)
BACKOFF_DIST_M = 2.0  # fallback exit; vision re-acquire normally exits first
BACKOFF_MIN_S = 0.6
BACKOFF_EXIT_REV_MPS = 0.3  # reverse considered arrested
BACKOFF_COOLDOWN_S = 1.5  # re-entry lockout after a backoff ends
BACKOFF_COLLISION_GRACE_S = 0.3  # ignore repeats of the originating impact
# Gate re-acquisition during the reverse: stop backing off as soon as the gate
# is usefully in view again, rather than always running the full distance.
BACKOFF_REACQ_MIN_BX_M = 4.0  # > MIN_BX_FOR_ELEV, leaves room to re-accelerate
BACKOFF_REACQ_HOLD_S = 0.2  # ~6 camera frames; rejects single-frame flicker
BACKOFF_VIS_VEL_HOLD_S = 0.25  # 30 Hz camera vs 60 Hz control
BACKOFF_YAW_MAX_DEG = 6.0  # half of FLYING's +/-12 bearing clip
BACKOFF_YAW_DECAY_S = 1.5  # stale bearing decays out rather than spinning us
WEAK_BLEND_SCALE = 0.35  # reduce lateral authority on unreliable detections

# Post-GO speed safety: IMU vX is often ~0 right after reset, which otherwise
# saturates PITCH_DES_MIN and open-loop dives to 20–30 km/h.
LEAN_RAMP_S = 2.5  # after GO: no dive past DESIRED_PITCH_DEG
UNTRUSTED_VX_MPS = 0.5  # |vX| below this → no dive (immediate)


class Phase(Enum):
    WAIT_FOR_DATA = auto()
    WAIT_FOR_START = auto()
    FLYING = auto()
    BACKOFF = auto()


def gate_segment_delta_ned(
    gate_map: list | None, active: int, *, flipz: bool = False
) -> np.ndarray | None:
    """Δ = p_next − p_current in NED (optional climb-course Z flip)."""
    if not gate_map or active < 0 or active + 1 >= len(gate_map):
        return None
    try:
        p0 = np.asarray(gate_map[active]["pos"], dtype=float).copy()
        p1 = np.asarray(gate_map[active + 1]["pos"], dtype=float).copy()
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if p0.shape != (3,) or p1.shape != (3,):
        return None
    if flipz:
        p0[2] = -p0[2]
        p1[2] = -p1[2]
    return p1 - p0


def apply_lookahead_body(
    bx: float,
    by: float,
    bz: float,
    gate_quat: np.ndarray | None,
    delta_ned: np.ndarray | None,
    lam: float,
) -> tuple[float, float, float, float]:
    """Bias aim toward next gate: gate-frame lateral + vertical only.

    Map quats often put the long along-track Δ on gate-"right" (live log:
    delta_gate≈[-2.1, +23.6, -5.1]). Adding that to body-y yanked ~8 m
    sideways. Use the *smaller* horizontal gate-frame component as lateral,
    clamp offsets, leave bx unchanged.
    """
    if lam <= 0.0 or delta_ned is None or gate_quat is None:
        return bx, by, bz, 0.0
    from rl.spec import quat_to_R

    R_wg = quat_to_R(np.asarray(gate_quat, dtype=float))  # gate → world
    dg = R_wg.T @ np.asarray(delta_ned, dtype=float)
    # Horizontal gate axes: thru=dg[0], right=dg[1]. Along-track is the large one.
    if abs(float(dg[1])) >= abs(float(dg[0])):
        lateral = float(dg[0])
    else:
        lateral = float(dg[1])
    vert = float(dg[2])
    lat_off = float(np.clip(lam * lateral, -LOOKAHEAD_OFFSET_MAX_M, LOOKAHEAD_OFFSET_MAX_M))
    vert_off = float(np.clip(lam * vert, -LOOKAHEAD_OFFSET_MAX_M, LOOKAHEAD_OFFSET_MAX_M))
    ax = bx
    ay = by + lat_off
    az = bz + vert_off
    if ax <= 0.1:
        return bx, by, bz, 0.0
    return ax, ay, az, float(lam)


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
    hover_thrust: float = HOVER_THRUST,
    vX: float = float("nan"),
    dt: float = 1.0 / GP_CONTROL_HZ,
    flying_t: float = float("nan"),
    delta_ned: np.ndarray | None = None,
    lookahead_lambda: float = LOOKAHEAD_LAMBDA,
    gate_quat: np.ndarray | None = None,
) -> tuple[float, float, float, float, dict]:
    """Anduril FLYING guidance. Mutates `state`.

    Returns DEGREE-valued commands (roll, pitch, yaw) + thrust, exactly like
    the original — the live pilot ships them on the attitude-quaternion wire
    (Controller "attitude_quat" mode); rl.gp_expert converts to rad/s for the
    internal rate-plant env.

    When a forward-speed estimate `vX` (body m/s) is supplied, a speed-PD
    pitch loop always runs (gate or not), targeting at most MAX_SPEED_MPS
    (10 km/h). Callers that omit vX (RL expert, sign-audit harness) keep the
    original fixed DESIRED_PITCH_DEG behavior.

    `flying_t` is seconds since GO (lean ramp / untrusted-vX guards).
    `delta_ned` / `gate_quat` / `lookahead_lambda` bias aim toward next gate
    in the current-gate frame (vision-aligned ≈ body).
    """
    vision_valid = False
    reliable = False
    bx = by = bz = float("nan")
    vis_frame_id = None
    lam_used = 0.0
    if vision is not None:
        bx = float(vision.get("body_x_m", float("nan")))
        by = float(vision.get("body_y_m", float("nan")))
        bz = float(vision.get("body_z_m", float("nan")))
        vis_frame_id = vision.get("frame_id")
        if not any(math.isnan(v) for v in (bx, by, bz)) and bx > 0.1:
            vision_valid = True
            # Anduril reliable tier; YOLO/legacy estimates default True.
            reliable = bool(vision.get("reliable", True))
            bx, by, bz, lam_used = apply_lookahead_body(
                bx, by, bz, gate_quat, delta_ned, float(lookahead_lambda)
            )
            if bx <= 0.1:
                vision_valid = False
                lam_used = 0.0

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

    # Vision-IMU velocity fusion (lateral + body-down). Forward speed for the
    # cap stays IMU-only — OF understates closing rate and caused dive saturation.
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
        if not reliable:
            # Weak Anduril hint: servo descend/yaw only — don't bank hard.
            blend *= WEAK_BLEND_SCALE

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

    if math.isnan(vX):
        # RL expert / offline harnesses: original fixed lean.
        pitch_des_deg = DESIRED_PITCH_DEG
        v_target = float("nan")
        state["prev_vx"] = None
    else:
        if vision_valid and reliable:
            ease = float(np.clip(bx / SLOWDOWN_START_M, 0.0, 1.0))
            v_target = THRU_SPEED_MPS + (CRUISE_SPEED_MPS - THRU_SPEED_MPS) * ease
        elif vision_valid:
            v_target = THRU_SPEED_MPS  # weak detection: crawl
        else:
            v_target = BLIND_CRAWL_MPS  # lost lock: crawl, never free-dive
        v_target = min(v_target, MAX_SPEED_MPS)
        prev_vx = state.get("prev_vx")
        a_fwd = 0.0 if prev_vx is None else (vX - prev_vx) / max(dt, 1e-3)
        state["prev_vx"] = vX
        state["ax_fwd_ema"] = 0.3 * a_fwd + 0.7 * state["ax_fwd_ema"]
        pitch_des_deg = float(
            np.clip(
                DESIRED_PITCH_DEG
                - K_SPEED_P * (v_target - vX)
                + K_SPEED_D * state["ax_fwd_ema"],
                PITCH_DES_MIN_DEG,
                PITCH_DES_MAX_DEG,
            )
        )
        # Hard speed cap: nose-up whenever over MAX, stronger past 15% over.
        if vX > MAX_SPEED_MPS * 1.15:
            pitch_des_deg = PITCH_DES_MAX_DEG
        elif vX > MAX_SPEED_MPS:
            pitch_des_deg = max(pitch_des_deg, 0.5 * PITCH_DES_MAX_DEG)
        # Post-GO lean ramp: don't max-dive while IMU vX is still settling.
        if not math.isnan(flying_t) and flying_t < LEAN_RAMP_S:
            pitch_des_deg = max(pitch_des_deg, DESIRED_PITCH_DEG)
        # Untrusted-vX: dead-reckon near zero → no dive (immediate). Waiting
        # used to open-loop to 30–40 km/h after Restart Race.
        if abs(vX) < UNTRUSTED_VX_MPS:
            pitch_des_deg = max(pitch_des_deg, DESIRED_PITCH_DEG)
        state["min_dive_s"] = 0.0
    pitch_cmd_deg = float(
        np.clip(
            (pitch_des_deg - pitch_deg) * KP,
            -PITCH_WIRE_MAX_DEG,
            PITCH_WIRE_MAX_DEG,
        )
    )
    p_lat = K_BEARING * bearing_body * blend
    d_lat_term = K_LAT_D * d_lat * blend
    # When still off-center near the gate, don't let blend starve the bank.
    if vision_valid and abs(bearing_body) >= BEARING_NEED_BANK_DEG:
        lat_blend = max(blend, NEAR_LAT_BLEND_FLOOR)
        p_lat = K_BEARING * bearing_body * lat_blend
        d_lat_term = K_LAT_D * d_lat * lat_blend
    desired_roll = float(np.clip(p_lat - d_lat_term, -MAX_BANK_DEG, MAX_BANK_DEG))
    roll_cmd_deg = (desired_roll - roll_deg) * KR
    yaw_cmd_deg = yaw_err * KY

    tilt = max(
        0.01,
        math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg)),
    )
    elev_err = float(state.get("last_elev_err", 0.0))
    thrust = (hover_thrust - elev_err * K_P_THRUST + d_vert * K_D_THRUST) / tilt
    thrust = float(np.clip(thrust, 0.0, 1.0))

    dbg = {
        "bearing_deg": bearing_body,
        "blend": blend,
        "elev_err": elev_err,
        "desired_roll": desired_roll,
        "d_lat": d_lat,
        "d_vert": d_vert,
        "vision_valid": vision_valid,
        "reliable": reliable,
        "bx": bx,
        "by": by,
        "bz": bz,
        "v_target": v_target,
        "pitch_des_deg": pitch_des_deg,
        "source": (vision or {}).get("source", ""),
        "lookahead": lam_used,
    }
    return roll_cmd_deg, pitch_cmd_deg, yaw_cmd_deg, thrust, dbg


def _backoff_pitch_target(elapsed: float, rev: float, braking: bool) -> float:
    """Time-scheduled reverse lean (degrees, +nose-up). Mutates nothing.

    push → coast → brake, driven by `elapsed`. `rev` (measured reverse speed,
    m/s) may only REDUCE authority, never extend it: with rev stuck at 0.0 —
    the exact post-impact failure mode — the lean still decays to zero and then
    goes negative purely on elapsed time, so a broken speed estimate can no
    longer hold the nose up all the way to the timeout.
    """
    if braking or elapsed >= BACKOFF_PUSH_S + BACKOFF_COAST_S:
        return BACKOFF_BRAKE_PITCH_DEG
    if rev > BACKOFF_MAX_SPEED_MPS:
        return BACKOFF_BRAKE_PITCH_DEG  # overspeed: brake, not merely level
    if elapsed >= BACKOFF_PUSH_S:
        frac = 1.0 - (elapsed - BACKOFF_PUSH_S) / BACKOFF_COAST_S
        decayed = BACKOFF_PITCH_DEG * max(0.0, frac)
        if rev > 0.8 * BACKOFF_MAX_SPEED_MPS:
            return min(decayed, 0.3 * BACKOFF_PITCH_DEG)
        return decayed
    if rev > 0.8 * BACKOFF_MAX_SPEED_MPS:
        return 0.3 * BACKOFF_PITCH_DEG
    return BACKOFF_PITCH_DEG


def _fresh_hold_state() -> dict:
    return {
        "last_elev_err": 0.0,
        "last_fused_frame_id": None,
        "gate_tilt_ema": None,
        "last_d_frame_id": None,
        "vY_at_vision": 0.0,
        "vD_at_vision": 0.0,
        "vision_vx_ema": 0.0,
        "vision_vy_ema": 0.0,
        "vision_vz_ema": 0.0,
        "prev_bearing_body": None,
        "prev_bearing_frame_id": None,
        "prev_gate_pD": None,
        "prev_elev_frame_id": None,
        "prev_vx": None,
        "ax_fwd_ema": 0.0,
        "min_dive_s": 0.0,
    }


class CommandSlew:
    """Rate-limit the outgoing attitude/thrust commands.

    Per-frame YOLO jitter otherwise lands on the wire as step changes in the
    attitude target — visible as bank/pitch twitch. Limiting the slew keeps
    the response smooth without touching the guidance gains.
    """

    def __init__(self, hz: float = GP_CONTROL_HZ):
        self._max_step_deg = CMD_SLEW_DEG_S / hz
        self._max_step_thrust = THRUST_SLEW_PER_S / hz
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[float, float, float, float] | None = None

    def apply(
        self, roll: float, pitch: float, yaw: float, thrust: float
    ) -> tuple[float, float, float, float]:
        if self._prev is None:
            self._prev = (roll, pitch, yaw, thrust)
            return self._prev
        pr, pp, py, pt = self._prev
        m = self._max_step_deg
        out = (
            pr + float(np.clip(roll - pr, -m, m)),
            pp + float(np.clip(pitch - pp, -m, m)),
            py + float(np.clip(yaw - py, -m, m)),
            pt
            + float(
                np.clip(thrust - pt, -self._max_step_thrust, self._max_step_thrust)
            ),
        )
        self._prev = out
        return out


class GPPilot:
    """AndurilGP estimation + guidance (WAIT_FOR_* phases like their Controller)."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self.n_passed = 0
        self.phase = Phase.WAIT_FOR_DATA
        self.est = GPEstimation(data)
        self.vel_tracker = VisionVelocityTracker()
        self.gate_smoother = GateEstimateSmoother()
        self._cmd_slew = CommandSlew()
        self._hold = _fresh_hold_state()
        self._backoff_start = 0.0
        self._backoff_dist = 0.0
        self._backoff_last_t = 0.0
        self._backoff_brake_since: float | None = None
        self._backoff_reacq_since: float | None = None
        self._backoff_rev_vis = 0.0
        self._backoff_rev_vis_t = 0.0
        self._backoff_exit_reacq = False
        self._last_backoff_end = 0.0
        self._last_gate_bearing_deg: float | None = None
        self._flying_since: float | None = None
        self._go_start_ms: int | None = None
        self._tick = 0
        self._est_started = False
        self._last_arm_attempt = 0.0
        self._wait_start_sim_ms = None
        self._finish_noted = False
        self._imu_ts_seen = None
        self._imu_ts_wall = 0.0
        self._frozen_noted = False
        self._disarm_since = None
        self._log = None
        self._log_wr = None
        self._log_last_flush = 0.0
        self._debug = os.environ.get("GP_DEBUG", "").strip() in ("1", "true", "yes")
        self.gate_map: list = []
        self._gate_flipz = False
        self._refresh_gate_map()
        # Original AndurilGP wire behavior: degree commands on the attitude
        # quaternion at 60 Hz (the encoding that flew the course).
        controller.control_hz = GP_CONTROL_HZ
        controller.set_control_mode("attitude_quat")
        controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
        print(
            f"[gp] AndurilGP controls pilot ready (lookahead λ={LOOKAHEAD_LAMBDA}"
            f", flipz={self._gate_flipz}, gates={len(self.gate_map)})",
            flush=True,
        )

    def _refresh_gate_map(self) -> None:
        """Load / refresh gate list (live track burst, else gate_map.json)."""
        from rl.fly2_course import detect_climb_course, resolve_gate_map

        gm = resolve_gate_map(self.data)
        if gm:
            self.gate_map = gm
            self._gate_flipz = detect_climb_course(gm)

    @property
    def gates_passed(self) -> int:
        return self.n_passed

    def on_attempt_start(self) -> None:
        self._reset_state()

    def reset_for_attempt(self) -> None:
        self._reset_state()
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)

    def _reset_state(self) -> None:
        self.n_passed = 0
        self.phase = Phase.WAIT_FOR_DATA
        self._hold = _fresh_hold_state()
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._cmd_slew.reset()
        self.est.reset()
        self.data.pop("collision", None)
        self._tick = 0
        self._last_arm_attempt = 0.0
        self._wait_start_sim_ms = None
        self._finish_noted = False
        self._imu_ts_seen = None
        self._imu_ts_wall = 0.0
        self._frozen_noted = False
        self._disarm_since = None
        self._backoff_start = 0.0
        self._backoff_dist = 0.0
        self._backoff_last_t = 0.0
        self._backoff_brake_since = None
        self._backoff_reacq_since = None
        self._backoff_rev_vis = 0.0
        self._backoff_rev_vis_t = 0.0
        self._backoff_exit_reacq = False
        self._last_backoff_end = 0.0
        self._last_gate_bearing_deg = None
        self._flying_since = None
        self._go_start_ms = None
        self._close_log()

    def _open_log(self) -> None:
        self._close_log()
        try:
            os.makedirs(os.path.join("rl", "data"), exist_ok=True)
            path = os.path.join("rl", "data", time.strftime("gp_log_%Y%m%d_%H%M%S.csv"))
            self._log = open(path, "w", newline="")
            self._log_wr = csv.writer(self._log)
            self._log_wr.writerow(
                "t roll pitch yaw cmd_roll_deg cmd_pitch_deg cmd_yaw_deg "
                "thrust bx by bz blend d_lat d_vert vY vD vX v_target "
                "pitch_des source lookahead".split()
            )
            print(f"[gp] flight log -> {path}", flush=True)
        except OSError as e:  # telemetry must never ground the pilot
            print(f"[gp] flight log unavailable: {e}", flush=True)
            self._log, self._log_wr = None, None

    def _close_log(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
        self._log, self._log_wr = None, None

    def _log_tick(self, att, cmds, thrust, vY, vD, dbg, vX=float("nan")) -> None:
        if self._log_wr is None:
            return
        try:
            now = time.time()
            vt = dbg.get("v_target", float("nan"))
            pd = dbg.get("pitch_des_deg", float("nan"))
            src = str(dbg.get("source", "") or "")
            la = float(dbg.get("lookahead", 0.0) or 0.0)
            self._log_wr.writerow(
                [f"{now:.3f}"]
                + [f"{v:.3f}" for v in att]
                + [f"{v:.3f}" for v in cmds]
                + [f"{thrust:.4f}"]
                + [f"{dbg[k]:.3f}" for k in ("bx", "by", "bz")]
                + [f"{dbg['blend']:.3f}", f"{dbg['d_lat']:.4f}", f"{dbg['d_vert']:.4f}"]
                + [
                    f"{vY:.3f}",
                    f"{vD:.3f}",
                    f"{vX:.3f}",
                    f"{vt:.3f}",
                    f"{pd:.3f}",
                    src,
                    f"{la:.3f}",
                ]
            )
            if now - self._log_last_flush >= 1.0:
                self._log.flush()
                self._log_last_flush = now
        except (OSError, ValueError, KeyError):
            self._close_log()

    def _physics_live(self, imu) -> bool:
        """Track the IMU sensor clock; frozen clock = sim idled the physics.

        Observed on stale races and countdown holds: telemetry keeps
        streaming and arm is acknowledged, but the HIGHRES_IMU timestamp
        stops advancing and attitude setpoints do nothing. Flying commands
        into that state and being released mid-command is what tips the
        drone over at launch — so FLYING waits for a live clock.
        """
        now = time.time()
        if imu is not None:
            ts = imu.get("time_us") or imu.get("time_usec")
            if ts != self._imu_ts_seen:
                self._imu_ts_seen = ts
                self._imu_ts_wall = now
                if self._frozen_noted:
                    print("[gp] IMU clock moving again — physics live.", flush=True)
                    self._frozen_noted = False
        live = self._imu_ts_wall > 0.0 and now - self._imu_ts_wall <= IMU_FROZEN_S
        if not live and self._imu_ts_wall > 0.0 and not self._frozen_noted:
            print(
                "[gp] Sim physics is IDLE (IMU clock frozen) — the drone "
                "will not respond. Click Restart Race in FlightSim; this "
                "client re-arms and flies the new countdown automatically.",
                flush=True,
            )
            self._frozen_noted = True
        return live

    def tick(self) -> None:
        self._tick += 1
        armed = bool(self.data.get("armed", False))
        imu = self.data.get("imu")
        race = self.data.get("race_status")
        physics_live = self._physics_live(imu)

        if self.phase == Phase.WAIT_FOR_DATA:
            self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
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
            self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
            if not armed:
                # A sim reset disarms the drone; go back and re-arm before
                # the new countdown finishes, or GO would fly a dead stick.
                self.phase = Phase.WAIT_FOR_DATA
                return
            if race is not None:
                sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
                start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                    print(f"[WAIT] Anchor set: sim_ms={sim_ms}", flush=True)
                if sim_ms < self._wait_start_sim_ms - CLOCK_RESET_SLACK_MS:
                    print(
                        f"[WAIT] Sim clock reset (sim_ms={sim_ms} < anchor="
                        f"{self._wait_start_sim_ms}) — re-arming for the new race.",
                        flush=True,
                    )
                    self._reset_state()
                    return
                finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
                # Vendor-faithful GO: only a *fresh* countdown that has elapsed.
                # No "already running → fly now" (that skipped the 3s hold after
                # manual Restart Race).
                race_fresh = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_fresh and sim_ms >= start_ms and finish_ns < 0
                if self._debug and self._tick % DEBUG_EVERY_N == 0:
                    print(
                        f"[WAIT] sim_ms={sim_ms} race_start={start_ms} "
                        f"finish_ns={finish_ns} fresh={race_fresh} "
                        f"go={countdown_done and physics_live}",
                        flush=True,
                    )
                if countdown_done and physics_live:
                    self._enter_flying(start_ms)
                elif start_ms > 0 and finish_ns >= 0 and not self._finish_noted:
                    print(
                        "[WAIT] Race is FINISHED — click Restart Race in "
                        "FlightSim; this client will re-arm and fly the new "
                        "countdown automatically.",
                        flush=True,
                    )
                    self._finish_noted = True
            elif self._debug and self._tick % DEBUG_EVERY_N == 0:
                print("[WAIT] No race_status yet — holding...", flush=True)
            return

        # FLYING / BACKOFF — Anduril guidance (+ collision reverse)
        if not armed:
            now = time.time()
            if self._disarm_since is None:
                self._disarm_since = now
            elif now - self._disarm_since >= DISARM_PERSIST_S:
                print("[gp] Disarmed (sim reset?) — re-arming.", flush=True)
                self._reset_state()
                return
        else:
            self._disarm_since = None

        # Abort into WAIT if Restart Race / new countdown / finish while airborne.
        if self.phase in (Phase.FLYING, Phase.BACKOFF) and race is not None:
            if self._should_abort_flying(race):
                return

        if not self._est_started and imu is not None:
            self.est.start()
            self._est_started = True

        if self._log is None:
            self._open_log()

        snap = self.est.snapshot()
        roll_deg, pitch_deg, yaw_deg = snap["att_deg"]
        vX = float(snap["vel_body"][0])
        vY = float(snap["vel_body"][1])
        vD = float(snap["vel_ned"][2])
        dt = 1.0 / GP_CONTROL_HZ
        flying_t = (
            time.time() - self._flying_since
            if self._flying_since is not None
            else float("nan")
        )

        if self.phase == Phase.BACKOFF:
            self._tick_backoff(roll_deg, pitch_deg, yaw_deg, vX, vY, vD, dt)
            return

        # Enter backoff on a fresh MAVLink COLLISION (mavlink_rx writes the key).
        if self.data.get("collision") is not None:
            # Pop unconditionally: suppressing entry without popping leaves a
            # stale key that re-fires every tick, turning the cooldown into a
            # permanent lockout.
            self.data.pop("collision", None)
            if time.time() - self._last_backoff_end >= BACKOFF_COOLDOWN_S:
                self._enter_backoff()
                self._tick_backoff(roll_deg, pitch_deg, yaw_deg, vX, vY, vD, dt)
                return

        vision = self.gate_smoother.update(self.data)
        vision_vel = self.vel_tracker.update(vision)

        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self.n_passed:
            self.n_passed = active

        if not self.gate_map:
            self._refresh_gate_map()
        delta_ned = gate_segment_delta_ned(
            self.gate_map, active, flipz=self._gate_flipz
        )
        gate_quat = None
        if self.gate_map and 0 <= active < len(self.gate_map):
            try:
                gate_quat = np.asarray(self.gate_map[active]["quat"], dtype=float)
            except (KeyError, TypeError, ValueError):
                gate_quat = None

        roll_cmd, pitch_cmd, yaw_cmd, thrust, dbg = compute_guidance(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            quat=snap["quat"],
            vY=vY,
            vD=vD,
            vision=vision,
            vision_vel=vision_vel,
            state=self._hold,
            vX=vX,
            dt=dt,
            flying_t=flying_t,
            delta_ned=delta_ned,
            lookahead_lambda=LOOKAHEAD_LAMBDA,
            gate_quat=gate_quat,
        )
        # Latch the live bearing so a backoff can steer back toward the gate
        # after it leaves the camera FOV. Cheap, and _enter_backoff can't
        # recover it afterwards.
        if dbg.get("vision_valid"):
            self._last_gate_bearing_deg = float(dbg.get("bearing_deg", 0.0))
        roll_cmd, pitch_cmd, yaw_cmd, thrust = self._cmd_slew.apply(
            roll_cmd, pitch_cmd, yaw_cmd, thrust
        )
        # Degree commands on the attitude-quaternion wire (original encoding).
        self.controller.set_attitude_quat_deg(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        self._log_tick(
            (roll_deg, pitch_deg, yaw_deg),
            (roll_cmd, pitch_cmd, yaw_cmd),
            thrust,
            vY,
            vD,
            dbg,
            vX=vX,
        )

        if self._debug and self._tick % DEBUG_EVERY_N == 0:
            vt = dbg.get("v_target", float("nan"))
            src = dbg.get("source", "")
            print(
                f"[gp] att=({roll_deg:+.1f}r {pitch_deg:+.1f}p) "
                f"gate=({dbg['bx']:+.1f},{dbg['by']:+.1f},{dbg['bz']:+.1f}) "
                f"vX={vX:+.2f}/{vt:.2f} src={src} blend={dbg['blend']:.2f} "
                f"elev={dbg['elev_err']:+.2f} T={thrust:.3f}",
                flush=True,
            )

    def _resume_flying(
        self, *, reseed_attitude: bool, preserve_vision: bool = False
    ) -> None:
        """Shared FLYING entry: re-arm lean ramp + clear dead-reckoned speed.

        `reseed_attitude` re-seeds the AHRS to the launch-ramp pitch. That is
        correct on the pad and WRONG mid-air: GyroAHRS is pure gyro
        integration with no accel correction, so a bad seed is a permanent
        bias for the rest of the flight — it never washes out. A mid-air
        resume clears velocity only.

        `preserve_vision` keeps the gate lock when the caller already has one,
        so a re-acquire exit doesn't drop straight back to BLIND_CRAWL_MPS.
        """
        self.phase = Phase.FLYING
        self._flying_since = time.time()
        if reseed_attitude:
            self.est.reset()
        else:
            self.est.zero_velocity()
        self._hold = _fresh_hold_state()
        self.vel_tracker.reset()
        if not preserve_vision:
            self.gate_smoother.reset()
        self._cmd_slew.reset()
        self.data.pop("collision", None)

    def _enter_flying(self, start_ms: int) -> None:
        """Countdown complete → FLYING with clean speed state."""
        print("Countdown complete! Flying!", flush=True)
        self._go_start_ms = start_ms
        self._finish_noted = False
        # On the pad: the drone really is sitting at LAUNCH_PITCH_DEG.
        self._resume_flying(reseed_attitude=True)

    def _abort_to_wait(self, reason: str) -> None:
        """Drop out of FLYING/BACKOFF and hold zero thrust for a fresh countdown."""
        print(f"[gp] {reason} — holding for countdown", flush=True)
        self.phase = Phase.WAIT_FOR_START
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
        self.est.reset()
        self._hold = _fresh_hold_state()
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._cmd_slew.reset()
        self._flying_since = None
        self._go_start_ms = None
        self._finish_noted = False
        race = self.data.get("race_status")
        if race is not None:
            self._wait_start_sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
            print(f"[WAIT] Anchor set: sim_ms={self._wait_start_sim_ms}", flush=True)
        else:
            self._wait_start_sim_ms = None

    def _should_abort_flying(self, race: dict) -> bool:
        """True if Restart Race / finish requires returning to WAIT (mutates state)."""
        sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
        start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
        finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
        if (
            self._wait_start_sim_ms is not None
            and sim_ms < self._wait_start_sim_ms - CLOCK_RESET_SLACK_MS
        ):
            self._abort_to_wait("Sim clock reset mid-flight")
            return True
        # New countdown: race_start is in the future (3s hold not elapsed yet).
        if start_ms > 0 and start_ms > sim_ms:
            self._abort_to_wait("New race countdown")
            return True
        # New race_start after the one we launched on (Restart Race finished GO).
        if (
            start_ms > 0
            and self._go_start_ms is not None
            and start_ms > self._go_start_ms
            and finish_ns < 0
        ):
            self._abort_to_wait("Restart Race — new race_start")
            return True
        if finish_ns >= 0:
            self._abort_to_wait("Race finished")
            return True
        return False

    def _enter_backoff(self) -> None:
        now = time.time()
        self.phase = Phase.BACKOFF
        self._backoff_start = now
        self._backoff_dist = 0.0
        self._backoff_last_t = now
        self._backoff_brake_since = None
        self._backoff_reacq_since = None
        self._backoff_rev_vis = 0.0
        self._backoff_rev_vis_t = 0.0
        self._backoff_exit_reacq = False
        self.data.pop("collision", None)
        # Drop vision D-terms so re-acquire after the reverse isn't polluted.
        # NOTE: gate_smoother is deliberately NOT reset — _tick_backoff tracks
        # the gate through the reverse so it can stop as soon as it's back in
        # view, and _last_gate_bearing_deg survives to steer the reverse.
        self._hold = _fresh_hold_state()
        self.vel_tracker.reset()
        self._cmd_slew.reset()
        # Impact leaves the strapdown holding pre-collision FORWARD speed. Left
        # stale, rev = max(0, -vX) reads 0 and the regulator commands maximum
        # nose-up all the way to the timeout. Clear velocity but NOT attitude
        # (zero_velocity, not reset — reset reseeds the launch-ramp pitch).
        self.est.zero_velocity()
        print(
            f"[gp] COLLISION — backing off up to {BACKOFF_DIST_M:.0f} m",
            flush=True,
        )

    def _backoff_rev(self, vX: float) -> float:
        """Reverse speed (m/s, >=0), IMU fused with vision range-rate.

        `max`, not a weighted blend: this is a one-sided safety limiter and
        both sources fail TOWARD zero (IMU when stale after impact, vision
        when the lock drops). Taking the max means either source seeing speed
        is enough to cut authority; a blend would let a zero-reading source
        mask a live one, which is the failure being fixed here.
        """
        rev_imu = max(0.0, -vX)
        if time.time() - self._backoff_rev_vis_t <= BACKOFF_VIS_VEL_HOLD_S:
            return max(rev_imu, self._backoff_rev_vis)
        return rev_imu

    def _tick_backoff(
        self,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        vX: float,
        vY: float,
        vD: float,
        dt: float,
    ) -> None:
        now = time.time()
        elapsed = now - self._backoff_start
        # Integrate reverse travel from body-forward speed (negative = reverse).
        step_dt = max(dt, now - self._backoff_last_t)
        self._backoff_last_t = now
        self._backoff_dist += max(0.0, -vX) * step_dt

        # Keep watching for the gate through the reverse — the whole point is
        # to back off only as far as it takes to see it again.
        vision = self.gate_smoother.update(self.data)
        vision_vel = self.vel_tracker.update(vision)
        if vision_vel is not None:
            # Reversing grows the gate range, so vx_body_mps (approach-positive)
            # goes negative. Same convention as rev_imu, directly comparable.
            self._backoff_rev_vis = max(0.0, -float(vision_vel["vx_body_mps"]))
            self._backoff_rev_vis_t = now
        bx = float(vision["body_x_m"]) if vision is not None else float("nan")

        reacquired = (
            vision is not None
            and bool(vision.get("reliable"))
            and not math.isnan(bx)
            and bx >= BACKOFF_REACQ_MIN_BX_M
        )
        if reacquired:
            if self._backoff_reacq_since is None:
                self._backoff_reacq_since = now
        else:
            self._backoff_reacq_since = None
        if (
            self._backoff_reacq_since is not None
            and now - self._backoff_reacq_since >= BACKOFF_REACQ_HOLD_S
            and elapsed >= BACKOFF_MIN_S
            and self._backoff_brake_since is None
        ):
            # Latch the brake rather than exiting outright — every exit hands
            # FLYING a drone whose reverse has been arrested.
            self._backoff_brake_since = now
            self._backoff_exit_reacq = True
            print(f"[gp] gate re-acquired at {bx:.1f} m — braking out", flush=True)

        # We hit something BEHIND us — reversing harder is exactly wrong. The
        # grace window ignores repeat messages from the originating impact
        # (ground contact can fire hundreds of times per second).
        if (
            self.data.get("collision") is not None
            and elapsed >= BACKOFF_COLLISION_GRACE_S
        ):
            self.data.pop("collision", None)
            if self._backoff_brake_since is None:
                self._backoff_brake_since = now
                print("[gp] COLLISION during backoff — braking out", flush=True)

        # Time-scheduled lean; rev can only cut authority, never extend it.
        rev = self._backoff_rev(vX)
        braking = self._backoff_brake_since is not None
        pitch_target = _backoff_pitch_target(elapsed, rev, braking)
        # These commands ARE the attitude setpoint on the quaternion wire, so
        # the P-on-error form needs an explicit magnitude bound — FLYING has
        # one (PITCH_WIRE_MAX_DEG), this path used to have none. With a biased
        # AHRS, (4.0 - (-17.8)) shipped 21.8 deg of lean: ~28 km/h in reverse.
        #
        # The magnitude bound alone is not enough. Because the command is
        # (target - measured), a biased `pitch_deg` inverts the brake: at
        # target=-4 with a -17.8 bias the raw command is +8.8, i.e. MORE
        # nose-up, and it just pins to the clamp. So bound nose-up relative to
        # what the schedule actually asked for — the estimate may trim within
        # BACKOFF_PITCH_AUTH_DEG of the target, never override its sign. The
        # schedule outranks the estimate, which is the only one of the two
        # that is observable after an impact.
        raw_pitch = (pitch_target - pitch_deg) * KP
        pitch_cmd = float(
            np.clip(
                min(raw_pitch, pitch_target + BACKOFF_PITCH_AUTH_DEG),
                -BACKOFF_PITCH_WIRE_MAX_DEG,
                BACKOFF_PITCH_WIRE_MAX_DEG,
            )
        )
        roll_cmd = float(np.clip((0.0 - roll_deg) * KR, -MAX_BANK_DEG, MAX_BANK_DEG))
        # Steer the reverse back toward the gate: a straight-line retreat only
        # helps if the gate was dead ahead, and after a glancing strike it
        # rarely is. Kept small — yaw rotates the body-X axis that rev and
        # _backoff_dist are both measured in.
        if vision is not None and not math.isnan(float(vision["body_y_m"])):
            bearing = math.degrees(math.atan2(float(vision["body_y_m"]), max(bx, 0.5)))
        elif self._last_gate_bearing_deg is not None:
            # Stale bearing decays out over BACKOFF_YAW_DECAY_S rather than
            # steering on an ever-older measurement.
            decay = max(0.0, 1.0 - elapsed / BACKOFF_YAW_DECAY_S)
            bearing = self._last_gate_bearing_deg * decay
        else:
            bearing = 0.0
        yaw_cmd = (
            float(np.clip(bearing, -BACKOFF_YAW_MAX_DEG, BACKOFF_YAW_MAX_DEG)) * KY
        )
        # A bare HOVER_THRUST constant sinks through the whole reverse: no
        # vertical damping at all. Reuse FLYING's form minus the proportional
        # term — there's no trustworthy gate elevation during backoff. vD is
        # NED-down (positive = descending), so +K_D*vD arrests a sink.
        tilt = max(
            0.01,
            math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg)),
        )
        thrust = float(np.clip((HOVER_THRUST + K_D_THRUST * vD) / tilt, 0.0, 1.0))
        roll_cmd, pitch_cmd, yaw_cmd, thrust = self._cmd_slew.apply(
            roll_cmd, pitch_cmd, yaw_cmd, thrust
        )
        self.controller.set_attitude_quat_deg(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        self._log_tick(
            (roll_deg, pitch_deg, yaw_deg),
            (roll_cmd, pitch_cmd, yaw_cmd),
            thrust,
            vY,
            vD,
            {
                "bx": bx,
                "by": float(vision["body_y_m"]) if vision is not None else float("nan"),
                "bz": float(vision["body_z_m"]) if vision is not None else float("nan"),
                "blend": 0.0,
                "d_lat": 0.0,
                "d_vert": 0.0,
                "source": "backoff-brake" if braking else "backoff",
            },
            vX=vX,
        )

        # Distance no longer exits directly — it latches the brake, and the
        # brake exits. Every exit path therefore hands FLYING a drone that has
        # had its reverse velocity arrested, instead of one at peak speed.
        if self._backoff_brake_since is None:
            # The schedule is already braking past PUSH+COAST, so latch there
            # unconditionally — otherwise a vX too broken to ever reach
            # BACKOFF_DIST_M would leave the brake unlatched and strand the
            # pilot in BACKOFF with no exit check running at all.
            if elapsed >= BACKOFF_PUSH_S + BACKOFF_COAST_S or (
                self._backoff_dist >= BACKOFF_DIST_M and elapsed >= BACKOFF_MIN_S
            ):
                self._backoff_brake_since = now
        if self._backoff_brake_since is not None:
            brake_done = now - self._backoff_brake_since >= BACKOFF_BRAKE_S
            if (brake_done and rev <= BACKOFF_EXIT_REV_MPS) or elapsed >= BACKOFF_MAX_S:
                why = "reacquired" if self._backoff_exit_reacq else "distance/time"
                print(
                    f"[gp] backoff done ({why}) dist={self._backoff_dist:.1f}m "
                    f"t={elapsed:.1f}s rev={rev:.2f}m/s — resuming chase",
                    flush=True,
                )
                self._last_backoff_end = now
                self._resume_flying(
                    reseed_attitude=False,
                    preserve_vision=self._backoff_exit_reacq,
                )

    def shutdown(self) -> None:
        self.est.stop()
        self._close_log()
