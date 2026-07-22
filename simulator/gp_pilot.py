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
from simulator.track_line import detect_track

# --- Blue-track-line fallback -------------------------------------------------
# The VQ2 course is marked by a glowing cyan floor ribbon that is visible to the
# camera almost continuously, unlike the gates (sparse, lost after each pass).
# When no usable GATE is in view, synthesize a virtual forward target ON the
# ribbon so guidance keeps steering along the course instead of coasting blind
# through the starvation windows that stall every pilot after a gate or two.
# Defaults are env-overridable for live tuning (GP_TRACK_*). Shorter lookahead +
# higher lateral gain = tighter tracing of the ribbon (corrects offset sooner);
# too tight oscillates. 4 m / 3.5 is a moderate-tight starting point.
TRACK_LOOKAHEAD_M = 4.0  # virtual target forward distance (body x)
TRACK_LAT_GAIN = 3.5  # m of body-y per unit image offset (normalized -1..1)
TRACK_ANGLE_GAIN = 0.8  # weight on the ribbon-heading lookahead term (curves)
TRACK_ANGLE_CLAMP = 0.6  # rad; ignore near-horizontal (low-strength) headings
TRACK_MIN_STRENGTH = 0.33  # require >= ~1/3 of bands (matches the detector floor)
TRACK_ANGLE_MIN_STRENGTH = 0.5  # trust the ribbon HEADING only above this (>=6 bands)

# --- Post-pass turn search ----------------------------------------------------
# After a gate pass the drone often goes blind before the (off-axis) next gate
# enters the narrow forward camera. Blind guidance only banks (translates) and
# never re-orients, so on a real turn it coasts straight past. When blind for a
# beat after a pass, ARC toward where the course is trending (the ribbon
# offset / last gate bearing) via a strong side virtual target, sweeping the
# next gate into view instead of flying off straight.
SEARCH_START_TICKS = 18  # ~0.3 s blind after a pass before arcing
SEARCH_LOOKAHEAD_M = 4.0
SEARCH_LAT_M = 4.0  # strong side offset -> max turn command toward the course
SEARCH_CUE_ALPHA = 0.15  # EMA on the lateral course-direction cue
# During search, YAW to reorient toward the next gate but scale the BANK down so
# the drone turns to FACE it rather than slamming sideways and overshooting the
# opening (then having to swing back). Once facing it, the normal approach takes
# it mostly straight in — matches the observed "dip left then go right" miss.
SEARCH_ROLL_SCALE = 0.35


# A real next gate is near and roughly ahead. After a pass YOLO intermittently
# locks a FAR background gate (live: bx=73.8 m, by=-16.4 m) and the pilot banks
# hard at the phantom, kicking off the drift that crashes it. Drop such estimates
# so the blue-line fallback (or the blind sideslip null) takes over instead.
GATE_MAX_RANGE_M = 25.0
GATE_MAX_LAT_M = 12.0


def _plausible_gate(vision) -> bool:
    """Geometrically believable as the next gate (near, ahead, finite)."""
    if vision is None:
        return False
    bx = vision.get("body_x_m")
    by = vision.get("body_y_m")
    bz = vision.get("body_z_m")
    if bx is None or by is None or bz is None:
        return False
    if not all(math.isfinite(v) for v in (bx, by, bz)):
        return False
    return 0.1 < bx < GATE_MAX_RANGE_M and abs(by) < GATE_MAX_LAT_M


def _gate_usable(vision) -> bool:
    """A real gate estimate good enough to steer to (reliable, near, ahead)."""
    if vision is None or not vision.get("reliable", False):
        return False
    return _plausible_gate(vision)


class TrackVirtualGate:
    """Turn the blue-ribbon detection into a virtual gate for compute_guidance.

    A point TRACK_LOOKAHEAD_M ahead, offset in body-y by where the ribbon sits
    (near offset + a heading-projected lookahead), held level (body_z=0). Feeding
    it as `vision` makes the proven guidance bank to CENTER the ribbon and cruise
    forward — no new control law, no velocity estimate. detect_track runs only on
    a NEW camera frame (~30 Hz, ~2 ms) and the result is cached between ticks."""

    def __init__(self):
        self._last_fid = None
        self._last = None

        def _f(name, default):
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        self.lookahead = _f("GP_TRACK_LOOKAHEAD", TRACK_LOOKAHEAD_M)
        self.lat_gain = _f("GP_TRACK_LATGAIN", TRACK_LAT_GAIN)
        self.angle_gain = _f("GP_TRACK_ANGGAIN", TRACK_ANGLE_GAIN)

    def reset(self) -> None:
        self._last_fid = None
        self._last = None

    def synth(self, data) -> dict | None:
        frame = data.get("frame")
        if not frame or frame.get("img") is None:
            return None
        fid = frame.get("frame_id")
        if fid != self._last_fid:
            self._last_fid = fid
            try:
                self._last = detect_track(frame["img"])
            except Exception:
                self._last = None
        t = self._last
        if t is None or t.get("strength", 0.0) < TRACK_MIN_STRENGTH:
            return None
        # Offset (near-ribbon lateral position) is the reliable signal. The
        # heading (angle) is only trustworthy with enough bands: the up-tilted
        # camera sees little of the floor ribbon, so most detections are 2-4
        # bands where `angle` saturates near +-pi/2 and points the WRONG way
        # (live: a=+1.09 at s=0.33 banked +14 deg RIGHT into a LEFT curve). Use it
        # only above TRACK_ANGLE_MIN_STRENGTH.
        by = t["offset"] * self.lat_gain
        if float(t["strength"]) >= TRACK_ANGLE_MIN_STRENGTH:
            ang = max(-TRACK_ANGLE_CLAMP, min(TRACK_ANGLE_CLAMP, float(t["angle"])))
            by += math.tan(ang) * self.lookahead * self.angle_gain
        return {
            "body_x_m": self.lookahead,
            "body_y_m": float(by),
            "body_z_m": 0.0,
            "frame_id": fid,
            "reliable": True,
            "source": "track",
            "normal_body": None,
            "method": None,
            "strength": float(t["strength"]),
        }


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
TILT_EMA_ALPHA = 0.25
# Elevation PD retuned 2026-07-20 (bottom-bar strike forensics): at the old
# 0.014/0.0175 the loop's bandwidth was ~0.72 rad/s, zeta 0.45 — a ~3 m
# inter-gate descent could not converge in the ~3.4 s between re-acquire and
# the bx<=MIN_BX_FOR_ELEV cutoff at cruise, so every descending approach
# reached the cutoff still ~0.8 m off with ~0.5 m/s of sink. 0.03/0.045 puts
# it at ~1.06 rad/s, zeta ~0.78 (settle ~4 s).
K_P_THRUST = 0.03
K_D_THRUST = 0.045
ELEV_ERR_CLAMP_M = 2.0  # bound the P authority against far-range PnP swings
# Slow integral trim on the elevation loop. P-only left a steady-state offset
# equal to the hover-trim mismatch (0.264 const vs ~0.27 measured): logs show
# the drone riding ~0.5 m BELOW gate centre on every approach. Integrates only
# while a gate is actively ranged (bx > MIN_BX_FOR_ELEV), held elsewhere.
K_I_THRUST = 0.006  # thrust per m of elev error per second
ELEV_I_CLAMP = 0.03  # ~11% of hover — enough for trim, can't run away
# Seed at the measured hover deficit (~0.270 actual vs the byte-faithful
# 0.264 const): un-seeded, every blind window flew slightly BELOW true hover
# (log-verified mean 0.2639 while sinking at 0.7 m/s into gate 2's bottom
# bar) until the slow I-term charged, which it never did on early gates.
ELEV_I_SEED = 0.006
# Anti-windup: integrate only in the small-error trim regime. The gate-1
# climb-out holds a multi-meter elev error for seconds, which wound the
# integrator to the clamp and ballooned the drone over the gate.
ELEV_I_ERR_GATE_M = 1.0
BEARING_RATE_CLAMP_DEG_S = 60.0
# 2.0 (was 5.0): no sane approach needs >2 m/s of commanded vertical rate at
# <=10 km/h, but post-target-switch PnP refinement produced ±5 m/s phantom
# elev rates that cut thrust to 0.23 for ~0.5 s while 1.8 m LOW (log rows
# 217-221 of gp_log_20260720_171348). With K_D tripled the clamp must shrink.
ELEV_RATE_CLAMP_M_S = 2.0
# Vertical-speed null when no fresh elevation measurement exists (blind OR
# inside MIN_BX_FOR_ELEV): d_vert = clip(vD) so the K_D term arrests residual
# sink/climb instead of integrating it open-loop through the gate. Exact
# vertical analog of the K_BLIND_VY_DEG sideslip null; clamped because vD is
# IMU dead-reckoning (bound the damage, like BLIND_BANK_DEG does laterally).
BLIND_VD_CLAMP_MPS = 1.5
# Descent-rate cap. Overshoot into gate 2's bottom bar came from building more
# sink than the (slow) elevation loop could arrest before the near-gate blind
# window. Once descending faster than this, thrust is not allowed below the
# hover trim (no further cut), so gravity alone can't push the sink much past
# the cap and the loop gets to converge on centre instead of diving through it.
MAX_DESCENT_RATE_MPS = 0.8
# Floor safety net for the flat arena floor. The drone starts on the pad and
# the descending course keeps every gate above that floor, so GO-time NED z is
# a valid ground reference. Below this clearance above it, blend climb thrust
# in; suppressed while a fresh gate still sits below us (that descent is
# intended). Inert without position telemetry (alt blocked / offline harness).
FLOOR_CLEARANCE_M = 0.8
FLOOR_CLIMB_THRUST = 0.06
# 2.5 (was 3.0): take the frozen elevation sample as late as the 20°-tilted
# camera geometry allows, so the through-gate thrust servo runs on fresher data.
MIN_BX_FOR_ELEV = 2.5
# Blind-phase handling (pass-through suppression / lost lock). Verified fail
# mode: with vision invalid, blend=0 zeroed all lateral authority, so residual
# sideslip integrated unopposed for the ~1-1.5 s blind window and drifted the
# drone into the gate edge even after a perfectly centred approach.
K_BLIND_VY_DEG = 8.0  # deg of bank per m/s residual sideslip while blind
BLIND_BANK_DEG = 10.0  # cap (was 6): vY is vision-fused now, needs real authority
# Predictive lateral aim. The raw P-bank chases the gate's INSTANTANEOUS bearing,
# so a small offset (0.3 m right) keeps banking toward the gate even while the
# drone rushes rightward past it — it overshoots into the right edge in the blind
# zone (the recurring gate-3 strike). Instead, aim at where the gate sits
# relative to the drone AT THE CROSSING: by_pred = by - vY * t_lead, t_lead =
# time-to-gate (bx/vX), capped. Nulling by_pred REVERSES sideslip before the edge
# instead of feeding it. Undisturbed when vY~0 (gates entered centred + slow).
LAT_LEAD_S_MAX = 0.6  # cap on the sideslip look-ahead time (s)
ELEV_BLIND_DECAY = 0.97  # per 60 Hz tick (~0.55 s tau) on the frozen elev err
# Vision-derivative frame-gap tolerance: YOLO (primary source) skips camera
# frames when inference lags; dt scales by the actual gap, so up to 6 frames
# (200 ms) still yields a usable rate instead of silently dropping damping.
VIS_DERIV_MAX_GAP_FR = 6
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
# Don't cruise into a gate that isn't vertically converged: scale the cruise
# margin down as |elev_err| grows so the (slow) elevation loop gets more time
# per metre. Full cruise at <=0.25 m of error, pure THRU crawl at >=1.25 m.
VERT_SETTLED_ERR_M = 0.25
VERT_SLOW_ERR_M = 1.25
# Lateral analog: slow down when the gate needs a big turn so the swing-over has
# time to finish before the crossing, instead of barreling through the opening
# sideways and clipping an edge (gate-4 turn: 4 m offset at 2.5 m/s -> 2.3 m/s
# overshoot -> edge). Full cruise within LAT_SETTLED bearing, crawl beyond SLOW.
LAT_SETTLED_BEARING_DEG = 6.0
LAT_SLOW_BEARING_DEG = 18.0
K_SPEED_P = 2.5  # deg of pitch lean per m/s of speed error
K_SPEED_D = 0.6  # deg per m/s^2 damping on forward speed
PITCH_DES_MIN_DEG = -2.5
PITCH_DES_MAX_DEG = 6.0
PITCH_WIRE_MAX_DEG = 18.0  # clamp on attitude-quat pitch command

# Attitude-command slew limits (vision flicker → bank twitch).
CMD_SLEW_DEG_S = 90.0
THRUST_SLEW_PER_S = 1.0

# Collision backoff: reverse a few meters then re-acquire.
BACKOFF_DIST_M = 3.0
BACKOFF_PITCH_DEG = 4.0  # mild nose-up while reversing
BACKOFF_MAX_SPEED_MPS = 1.5  # ~5.4 km/h reverse cap
BACKOFF_MIN_S = 0.6
BACKOFF_MAX_S = 4.0
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
    floor_clearance_m: float = float("nan"),
    yaw_rate_rps: float = 0.0,
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
    """
    vision_valid = False
    reliable = False
    bx = by = bz = float("nan")
    vis_frame_id = None
    if vision is not None:
        bx = float(vision.get("body_x_m", float("nan")))
        by = float(vision.get("body_y_m", float("nan")))
        bz = float(vision.get("body_z_m", float("nan")))
        vis_frame_id = vision.get("frame_id")
        if not any(math.isnan(v) for v in (bx, by, bz)) and bx > 0.1:
            vision_valid = True
            # Anduril reliable tier; YOLO/legacy estimates default True.
            reliable = bool(vision.get("reliable", True))

    if vision_valid and vision.get("track_break"):
        # The smoother switched gates: previous frames describe a DIFFERENT
        # target, so any derivative across the switch is a phantom rate (a
        # 2 m by-jump in one frame reads as ~570 deg/s bearing rate → pins
        # the roll at the clamp in the wrong direction).
        state["prev_gate_pD"] = None
        state["prev_elev_frame_id"] = None
        state["prev_bearing_body"] = None
        state["prev_bearing_frame_id"] = None
        state["last_d_frame_id"] = None
        state["gate_tilt_ema"] = None

    elev_rate = 0.0
    elev_fresh = vision_valid and bx > MIN_BX_FOR_ELEV and vis_frame_id is not None
    if elev_fresh:
        qw, qx, qy, qz = quat
        gate_pD = (
            2 * (qx * qz - qw * qy) * bx
            + 2 * (qy * qz + qw * qx) * by
            + (1 - 2 * (qx * qx + qy * qy)) * bz
        )
        prev_fid = state.get("prev_elev_frame_id")
        if prev_fid is not None and 0 < vis_frame_id - prev_fid <= VIS_DERIV_MAX_GAP_FR:
            dt_e = (vis_frame_id - prev_fid) / 30.0
            elev_rate = (gate_pD - state["prev_gate_pD"]) / dt_e
        state["prev_gate_pD"] = gate_pD
        state["prev_elev_frame_id"] = vis_frame_id
        state["last_elev_err"] = gate_pD
        # Gate above => gate_pD < 0 => trim thrust up (and vice versa).
        if abs(gate_pD) < ELEV_I_ERR_GATE_M:
            state["elev_i"] = float(
                np.clip(
                    state.get("elev_i", 0.0) - gate_pD * K_I_THRUST * dt,
                    -ELEV_I_CLAMP,
                    ELEV_I_CLAMP,
                )
            )
    else:
        # No fresh elevation measurement: fully blind OR vision valid but
        # inside MIN_BX_FOR_ELEV (the valid-but-close case used to HOLD the
        # frozen error un-decayed and keep a stale below-hover command alive
        # into the gate — log-verified on the gate-2 bottom-bar strike).
        state["prev_gate_pD"] = None
        state["prev_elev_frame_id"] = None
        # Decay (don't hold) the frozen elev error: if the last sample came
        # from a blended/wrong target, holding it locks a vertical impulse in
        # open-loop all the way through the gate.
        state["last_elev_err"] = float(state.get("last_elev_err", 0.0)) * (
            ELEV_BLIND_DECAY
        )

    # Vision-IMU velocity fusion (lateral + body-down). Forward speed for the
    # cap stays IMU-only — OF understates closing rate and caused dive saturation.
    if (
        vision_vel is not None
        and vis_frame_id is not None
        and vis_frame_id != state.get("last_fused_frame_id")
    ):
        # De-rotate: the vision velocity is -d(gate_body)/dt, which during a YAW
        # picks up a phantom lateral term (the gate sweeping across the image is
        # rotation, not translation): v_true = v_vision - omega_z * bx. Untreated,
        # a search/turn yaw spiked vY to +-2.6 m/s and slammed the bank the wrong
        # way (the hit-or-miss turns). omega_z ~= 0 on a straight approach, so
        # those are unchanged.
        # SIGN-SAFE: the gyro sign convention here is uncertain (estimator negates
        # gz; sim is inverted). Only accept the de-rotation when it SHRINKS |vY|
        # (correct sign removing a real phantom). If it would GROW |vY| (wrong
        # sign), keep the raw value — so this can never be worse than baseline.
        vy_meas = float(vision_vel["vy_body_mps"])
        vy_derot = vy_meas - yaw_rate_rps * bx
        vy_raw = vy_derot if abs(vy_derot) <= abs(vy_meas) else vy_meas
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
            and 0 < vis_frame_id - prev_bf <= VIS_DERIV_MAX_GAP_FR
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

    if not elev_fresh and not math.isnan(vD):
        # Vertical analog of the blind sideslip null: with no fresh elevation
        # measurement the vision D-term reads ~0 exactly when it matters (log:
        # d_vert 0.01 vs vD 0.70 inside bx<2.5, 0.000 while blind) and 0.5 m/s
        # of residual sink drops ~0.8-1.0 m across the blind span — 60%+ of
        # the half-gate. Null the measured vertical speed instead; the
        # existing +d_vert*K_D_THRUST term turns it into arresting thrust.
        d_vert = float(np.clip(vD, -BLIND_VD_CLAMP_MPS, BLIND_VD_CLAMP_MPS))

    if math.isnan(vX):
        # RL expert / offline harnesses: original fixed lean.
        pitch_des_deg = DESIRED_PITCH_DEG
        v_target = float("nan")
        state["prev_vx"] = None
    else:
        if vision_valid and reliable:
            ease = float(np.clip(bx / SLOWDOWN_START_M, 0.0, 1.0))
            vert_ok = float(
                np.clip(
                    (VERT_SLOW_ERR_M - abs(float(state.get("last_elev_err", 0.0))))
                    / (VERT_SLOW_ERR_M - VERT_SETTLED_ERR_M),
                    0.0,
                    1.0,
                )
            )
            # Slow for a big turn: scale cruise down as the gate bearing grows so
            # the lateral swing finishes before the crossing (edge-clip fix).
            lat_ok = float(
                np.clip(
                    (LAT_SLOW_BEARING_DEG - abs(bearing_body))
                    / (LAT_SLOW_BEARING_DEG - LAT_SETTLED_BEARING_DEG),
                    0.0,
                    1.0,
                )
            )
            v_target = (
                THRU_SPEED_MPS
                + (CRUISE_SPEED_MPS - THRU_SPEED_MPS) * ease * vert_ok * lat_ok
            )
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
    if vision_valid:
        # Predictive lateral aim: null where the gate will be at the crossing
        # given current sideslip, not its instantaneous bearing. t_lead grows at
        # close range (bx/vX) so drift is reversed BEFORE the blind zone; capped
        # because vY, though vision-fused, is not exact.
        vx_eff = vX if not math.isnan(vX) and vX > 0.5 else 2.0
        t_lead = min(bx / vx_eff, LAT_LEAD_S_MAX)
        by_pred = by - vY * t_lead
        bearing_ctrl = float(
            np.clip(math.degrees(math.atan2(by_pred, bx)), -25.0, 25.0)
        )
        p_lat = K_BEARING * bearing_ctrl * blend
        d_lat_term = K_LAT_D * d_lat * blend
        desired_roll = float(np.clip(p_lat - d_lat_term, -MAX_BANK_DEG, MAX_BANK_DEG))
    else:
        p_lat = K_BEARING * bearing_body * blend
        d_lat_term = K_LAT_D * d_lat * blend
        # Blind (threading / suppressed): don't just level the wings — null
        # the residual sideslip so we cross the gate plane without drifting
        # into the edge. Inert whenever vision is valid.
        desired_roll = float(
            np.clip(-K_BLIND_VY_DEG * vY, -BLIND_BANK_DEG, BLIND_BANK_DEG)
        )
    roll_cmd_deg = (desired_roll - roll_deg) * KR
    yaw_cmd_deg = yaw_err * KY

    tilt = max(
        0.01,
        math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg)),
    )
    elev_err = float(state.get("last_elev_err", 0.0))
    elev_err_c = float(np.clip(elev_err, -ELEV_ERR_CLAMP_M, ELEV_ERR_CLAMP_M))
    thrust = (
        hover_thrust
        + float(state.get("elev_i", 0.0))
        - elev_err_c * K_P_THRUST
        + d_vert * K_D_THRUST
    ) / tilt
    # Descent-rate cap: sinking faster than MAX_DESCENT_RATE_MPS => don't let
    # the command sit below the hover trim (that keeps accelerating the sink).
    # Bounds the descent so it settles near the cap instead of diving through
    # a low gate's bottom bar.
    if not math.isnan(vD) and vD > MAX_DESCENT_RATE_MPS:
        hover_floor = (hover_thrust + float(state.get("elev_i", 0.0))) / tilt
        thrust = max(thrust, hover_floor)
    # Floor safety net: below FLOOR_CLEARANCE_M above the arena floor, blend in
    # climb thrust so an overshoot below a low gate (or an open-loop blind sink)
    # can't touch the ground. Suppressed while a fresh gate still sits below us
    # (>0.3 m): that descent is the intended course, not a fall.
    if not math.isnan(floor_clearance_m) and floor_clearance_m < FLOOR_CLEARANCE_M:
        gate_below = elev_fresh and float(state.get("last_elev_err", 0.0)) > 0.3
        if not gate_below:
            floor_frac = float(
                np.clip(
                    (FLOOR_CLEARANCE_M - floor_clearance_m) / FLOOR_CLEARANCE_M,
                    0.0,
                    1.0,
                )
            )
            climb_t = (
                hover_thrust
                + float(state.get("elev_i", 0.0))
                + FLOOR_CLIMB_THRUST * floor_frac
            ) / tilt
            thrust = max(thrust, climb_t)
    thrust = float(np.clip(thrust, 0.0, 1.0))

    dbg = {
        "bearing_deg": bearing_body,
        "blend": blend,
        "elev_err": elev_err,
        "desired_roll": desired_roll,
        "vY_fused": vY,
        "d_lat": d_lat,
        "d_vert": d_vert,
        "vision_valid": vision_valid,
        "reliable": reliable,
        "bx": bx,
        "by": by,
        "bz": bz,
        "v_target": v_target,
        "pitch_des_deg": pitch_des_deg,
        "elev_i": float(state.get("elev_i", 0.0)),
        "source": (vision or {}).get("source", ""),
        "infer_ms": (vision or {}).get("infer_ms"),
    }
    return roll_cmd_deg, pitch_cmd_deg, yaw_cmd_deg, thrust, dbg


def _fresh_hold_state() -> dict:
    return {
        "last_elev_err": 0.0,
        "elev_i": ELEV_I_SEED,
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
        # Silence the per-frame "[vision] GATE cx=..." HSV spam so the [gp]
        # guidance lines (and src=TRACK) are readable.
        data["_quiet_vision"] = True
        self.n_passed = 0
        self.phase = Phase.WAIT_FOR_DATA
        self.est = GPEstimation(data)
        self.vel_tracker = VisionVelocityTracker()
        self.gate_smoother = GateEstimateSmoother()
        self._cmd_slew = CommandSlew()
        self._hold = _fresh_hold_state()
        # Blue-ribbon fallback (on by default; GP_TRACKLINE=0 to disable). Only
        # ever engages when no real gate is usable, so it can add steering to a
        # starvation window but never override a gate.
        self._trackline = (
            TrackVirtualGate()
            if os.environ.get("GP_TRACKLINE", "1").strip() not in ("0", "false", "no")
            else None
        )
        self._track_active = False  # debug: was the last guidance from the ribbon
        self._post_pass_until = 0  # tick until which the ribbon overrides gates
        self._blind_ticks = 0  # consecutive ticks with no usable gate
        self._course_cue = 0.0  # EMA lateral course direction (neg=left)
        self._search_active = False  # debug: arcing to reacquire after a pass
        # Collision backoff DISABLED by default (GP_BACKOFF=1 to restore): a hit
        # made the drone reverse ~3 m, losing all progress and flying backward
        # into oblivion instead of just continuing toward the gate.
        self._backoff_on = os.environ.get("GP_BACKOFF", "0").strip() in (
            "1",
            "true",
            "yes",
        )
        self._backoff_start = 0.0
        self._backoff_dist = 0.0
        self._backoff_last_t = 0.0
        self._flying_since: float | None = None
        self._go_start_ms: int | None = None
        self._floor_z0: float | None = None  # GO-time NED z = flat-floor level
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
        # Original AndurilGP wire behavior: degree commands on the attitude
        # quaternion at 60 Hz (the encoding that flew the course).
        controller.control_hz = GP_CONTROL_HZ
        controller.set_control_mode("attitude_quat")
        controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
        print("[gp] AndurilGP controls pilot ready (make control-flight)", flush=True)

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
        if self._trackline is not None:
            self._trackline.reset()
        self._track_active = False
        self._post_pass_until = 0
        self._blind_ticks = 0
        self._course_cue = 0.0
        self._search_active = False
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
        self._flying_since = None
        self._go_start_ms = None
        self._floor_z0 = None
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
                "pitch_des elev_i source gate".split()
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
            ei = dbg.get("elev_i", 0.0) or 0.0
            src = str(dbg.get("source", "") or "")
            self._log_wr.writerow(
                [f"{now:.3f}"]
                + [f"{v:.3f}" for v in att]
                + [f"{v:.3f}" for v in cmds]
                + [f"{thrust:.4f}"]
                + [f"{dbg[k]:.3f}" for k in ("bx", "by", "bz")]
                + [f"{dbg['blend']:.3f}", f"{dbg['d_lat']:.4f}", f"{dbg['d_vert']:.4f}"]
                + [f"{vY:.3f}", f"{vD:.3f}", f"{vX:.3f}", f"{vt:.3f}", f"{pd:.3f}"]
                + [f"{ei:.4f}", src, str(self.n_passed)]
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
        yaw_rate_rps = math.radians(float(snap["rates_body_dps"][2]))
        dt = 1.0 / GP_CONTROL_HZ
        flying_t = (
            time.time() - self._flying_since
            if self._flying_since is not None
            else float("nan")
        )

        if self.phase == Phase.BACKOFF:
            self._tick_backoff(roll_deg, pitch_deg, yaw_deg, vX, vY, vD, dt)
            return

        # Fresh MAVLink COLLISION (mavlink_rx writes the key). Backoff is OFF by
        # default — reversing lost all progress and flew the drone backward into
        # oblivion. Just consume the event and keep flying the guidance forward
        # toward the gate. GP_BACKOFF=1 restores the old reverse-and-reacquire.
        if self.data.get("collision") is not None:
            if self._backoff_on:
                self._enter_backoff()
                self._tick_backoff(roll_deg, pitch_deg, yaw_deg, vX, vY, vD, dt)
                return
            self.data.pop("collision", None)

        # Read the blue ribbon EVERY frame (cheap, cached per camera frame) for
        # both the fallback and diagnostics — synth caches its raw detection in
        # `_trackline._last` so the debug line can show whether the ribbon is
        # actually seen mid-course.
        track_vg = (
            self._trackline.synth(self.data) if self._trackline is not None else None
        )

        vision = self.gate_smoother.update(self.data)
        # Drop far/phantom locks (post-pass background gate, garbage YOLO) BEFORE
        # they can steer or feed the velocity tracker — banking at a 70 m phantom
        # is what crashed the run after gate 4.
        if not _plausible_gate(vision):
            vision = None
        vision_vel = self.vel_tracker.update(vision)

        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self.n_passed:
            self.n_passed = active

        # Blue-ribbon fallback: engage ONLY when no usable gate (a pure safety
        # net). The earlier post-pass OVERRIDE was reverted — with the up-tilted
        # camera the ribbon heading is too weak to reliably steer turns, and
        # forcing it over a good gate banked the wrong way. Gate-following (with
        # phantom rejection + predictive aim) is what reached 4 gates.
        self._track_active = False
        if track_vg is not None and not _gate_usable(vision):
            vision = track_vg
            vision_vel = None
            self._track_active = True

        # Track the lateral direction the course is trending (neg = left) from a
        # usable gate's bearing or the ribbon offset, to steer the post-pass
        # search the right way. (offset<0 = ribbon left = course turns left.)
        cue = None
        if _gate_usable(vision) and abs(vision["body_y_m"]) > 0.4:
            cue = math.copysign(1.0, vision["body_y_m"])
        elif (
            self._trackline is not None
            and self._trackline._last is not None
            and abs(self._trackline._last.get("offset", 0.0)) > 0.15
        ):
            cue = math.copysign(1.0, self._trackline._last["offset"])
        if cue is not None:
            self._course_cue += SEARCH_CUE_ALPHA * (cue - self._course_cue)

        # Post-pass turn search: blind for a beat after a pass -> ARC toward the
        # course direction so the off-axis next gate sweeps into the camera,
        # instead of coasting straight past the turn. A real gate (or the ribbon)
        # always wins; this only fires when there is nothing else to steer to.
        self._search_active = False
        if _gate_usable(vision) or self._track_active:
            self._blind_ticks = 0
        else:
            self._blind_ticks += 1
            # Only arc when we actually have a course-direction cue; with no cue,
            # hold straight (don't guess a turn onto a gate that's dead ahead).
            if (
                self.n_passed >= 1
                and self._blind_ticks >= SEARCH_START_TICKS
                and abs(self._course_cue) > 0.15
            ):
                side = math.copysign(1.0, self._course_cue)
                vision = {
                    "body_x_m": SEARCH_LOOKAHEAD_M,
                    "body_y_m": side * SEARCH_LAT_M,
                    "body_z_m": 0.0,
                    "frame_id": None,
                    "reliable": True,
                    "source": "search",
                    "normal_body": None,
                    "method": None,
                }
                vision_vel = None
                self._search_active = True

        # Flat-floor clearance for the floor safety net. Capture GO-time NED z
        # (drone on the pad = ground) once, then track height above it. z0
        # persists across backoffs — the floor doesn't move — and is cleared
        # only on a full race reset. Absent telemetry leaves clearance NaN
        # (guard inert).
        floor_clearance = float("nan")
        lp = self.data.get("local_position_ned") or self.data.get("odometry")
        if lp is not None and lp.get("z") is not None:
            z_down = float(lp["z"])
            if self._floor_z0 is None:
                self._floor_z0 = z_down
            floor_clearance = self._floor_z0 - z_down

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
            floor_clearance_m=floor_clearance,
            yaw_rate_rps=yaw_rate_rps,
        )
        # In search, keep the yaw (reorient toward the next gate) but soften the
        # bank so it faces the gate instead of slamming sideways past it.
        if self._search_active:
            roll_cmd *= SEARCH_ROLL_SCALE
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
            infer = dbg.get("infer_ms")
            infer_s = f" yolo={infer:.0f}ms" if infer is not None else ""
            steer = (
                "SEARCH"
                if self._search_active
                else ("TRACK" if self._track_active else src)
            )
            trk = self._trackline._last if self._trackline is not None else None
            trk_s = (
                f"trk(o={trk['offset']:+.2f},a={trk['angle']:+.2f},s={trk['strength']:.2f})"
                if trk is not None
                else "trk=none"
            )
            print(
                f"[gp] att=({roll_deg:+.1f}r {pitch_deg:+.1f}p {yaw_deg:+.0f}y) "
                f"gate=({dbg['bx']:+.1f},{dbg['by']:+.1f},{dbg['bz']:+.1f}) "
                f"vX={vX:+.2f}/{vt:.2f} vY={dbg['vY_fused']:+.2f} src={steer}{infer_s} "
                f"ycmd={yaw_cmd:+.0f} cue={self._course_cue:+.2f} blind={self._blind_ticks} "
                f"droll={dbg['desired_roll']:+.1f} T={thrust:.3f} {trk_s}",
                flush=True,
            )

    def _resume_flying(self) -> None:
        """Shared FLYING entry: re-arm lean ramp + full AHRS/vel reset."""
        self.phase = Phase.FLYING
        self._flying_since = time.time()
        self.est.reset()
        # elev_i is learned hover trim (vehicle-level, not gate-level): losing
        # it on every backoff kept the integrator near zero all flight and the
        # ~0.5 m low-riding bias never trimmed out (log-verified).
        ei = float(self._hold.get("elev_i", 0.0))
        self._hold = _fresh_hold_state()
        self._hold["elev_i"] = ei
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._cmd_slew.reset()
        self.data.pop("collision", None)

    def _enter_flying(self, start_ms: int) -> None:
        """Countdown complete → FLYING with clean speed state."""
        print("Countdown complete! Flying!", flush=True)
        self._go_start_ms = start_ms
        self._finish_noted = False
        self._resume_flying()

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
        self.data.pop("collision", None)
        # Drop vision D-terms so re-acquire after the reverse isn't polluted
        # (but keep the learned hover trim — see _resume_flying).
        ei = float(self._hold.get("elev_i", 0.0))
        self._hold = _fresh_hold_state()
        self._hold["elev_i"] = ei
        self.vel_tracker.reset()
        self.gate_smoother.reset()
        self._cmd_slew.reset()
        print(
            f"[gp] COLLISION — backing off ~{BACKOFF_DIST_M:.0f} m",
            flush=True,
        )

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

        # Regulate reverse speed — fixed nose-up used to hit 20–30 km/h.
        rev = max(0.0, -vX)
        if rev > BACKOFF_MAX_SPEED_MPS:
            pitch_target = 0.0
        elif rev > 0.8 * BACKOFF_MAX_SPEED_MPS:
            pitch_target = 0.3 * BACKOFF_PITCH_DEG
        else:
            pitch_target = BACKOFF_PITCH_DEG
        pitch_cmd = (pitch_target - pitch_deg) * KP
        roll_cmd = (0.0 - roll_deg) * KR
        yaw_cmd = 0.0
        # Include the learned hover trim: raw 0.264 is ~0.006 below measured
        # hover, so every backoff slowly sank toward the floor-wedge cycle.
        thrust = HOVER_THRUST + float(self._hold.get("elev_i", 0.0))
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
                "bx": float("nan"),
                "by": float("nan"),
                "bz": float("nan"),
                "blend": 0.0,
                "d_lat": 0.0,
                "d_vert": 0.0,
                "source": "backoff",
            },
            vX=vX,
        )

        done_dist = self._backoff_dist >= BACKOFF_DIST_M and elapsed >= BACKOFF_MIN_S
        done_time = elapsed >= BACKOFF_MAX_S
        if done_dist or done_time:
            print(
                f"[gp] backoff done dist={self._backoff_dist:.1f}m "
                f"t={elapsed:.1f}s — resuming chase",
                flush=True,
            )
            self._resume_flying()

    def shutdown(self) -> None:
        self.est.stop()
        self._close_log()
