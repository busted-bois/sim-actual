"""Blue-line corridor pilot — attitude_quat setpoints from HSV dual-line errors.

Uses the same AndurilGP wire encoding as gp_pilot:
  roll_cmd  = (desired_roll - roll_meas) * KR
  pitch_cmd = (pitch_des    - pitch_meas) * KP
  yaw_cmd   = yaw_err * KY
Absolute RPY setpoints do not turn the live plant; error-space cmds do.

Blue line primary; YOLO gate bearing is a bounded secondary assist
(BL_GATE_ASSIST=0 off). Cap forward speed at 25 km/h via speed-PD → pitch.

Race-gated: hold zero thrust until a race that started AFTER pilot startup
exists, then fly immediately (no countdown hold — user wants wheels-up right
after each manual Restart Race; the physics-live gate still holds while the
sim keeps the drone frozen). Per-attempt telemetry lands in
rl/data/bl_log_*.csv.
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

from simulator.gp_pilot import CommandSlew, GP_CONTROL_HZ, HOVER_THRUST, KP, KR, KY
from simulator.gp_vision import GateEstimateSmoother, best_pose_gate

BL_CONTROL_HZ = GP_CONTROL_HZ

# Image-error → desired bank / yaw-error (deg). Positive cx/hdg = corridor right
# → positive desired_roll / yaw_err; wire flips via KR/KY=-1 (GP convention).
# Gains raised twice after log analysis (bl_log_20260722_*): cx clips ~±0.5 and
# hdg is 0 in most track rows; measured sim response ≈2.2 deg/s of yaw per deg
# of wire cmd, so the 40-deg clamp targets ~85 deg/s peak turn rate.
K_ROLL_CX = 36.0
K_ROLL_HDG = 0.55  # deg bank per deg heading_err
K_YAW_CX = 40.0
K_YAW_HDG = 1.2  # deg yaw_err per deg heading_err
# Lateral-rate (cx) damping on roll: the baseline has yaw-rate (gz) damping but
# NONE on the roll-induced TRANSLATION that the gz gyro can't see — log-verified
# cx +-0.4 weave at ~0.68 sign-flips/s. EMA'd d(cx)/dt, hard-capped so it only
# damps the weave and can never dominate the bank command.
K_ROLL_DCX = 6.0  # deg bank per (cx unit / s)
DCX_EMA_ALPHA = 0.35
DCX_ROLL_MAX_DEG = 8.0
# Yaw-rate damping from the IMU z-gyro (streams in EVERY sim mode). This
# sim's gyros are sign-inverted (flightlab signs + log-verified: commanding
# right yields negative gz), so actual right-yaw rate = -gz and the damping
# term is +K*gz_raw — it always opposes rotation. Kills the track-mode weave
# (log-verified cx +-0.4 oscillation at +-20-45 deg/s) so higher P holds.
K_YAW_D_GZ = 0.12  # deg of yaw_err per deg/s of raw gz

K_PITCH_CY = 8.0
K_THRUST_CY = 0.10

CRUISE_PITCH_DEG = -2.0
CREEP_PITCH_DEG = -0.5
SEARCH_YAW_ERR_DEG = 35.0
HOLD_YAW_ERR_DEG = 12.0
LOSS_HOLD_S = 0.2
SEARCH_AFTER_S = 0.5

# Wire-command slew for this pilot. GP's 90 deg/s (1.5 deg/tick at a nominal
# 60 Hz) halves again at the real ~32 Hz loop rate — log-verified: the corner
# appears as a one-frame yaw_err step and the slewed cmd reached only 56% of
# it before the line left the FOV. 360 deg/s builds the full 40-deg command
# in ~0.2 s; the gz damping term keeps the agility from becoming oscillation.
BL_CMD_SLEW_DEG_S = 360.0

MAX_BANK_DEG = 26.0
YAW_ERR_MAX_DEG = 40.0
PITCH_DES_MIN_DEG = -4.0
PITCH_DES_MAX_DEG = 8.0
PITCH_WIRE_MAX_DEG = 18.0
THRUST_MIN = 0.15
THRUST_MAX = 0.55

CY_TARGET = 0.05

# 25 km/h hard cap (user). Cruise default 12 km/h — 18 carried too much speed
# into the gate-2 left-hander to brake in time (BL_CRUISE_KMH overrides).
MAX_SPEED_MPS = 25.0 / 3.6
CRUISE_SPEED_MPS = float(os.environ.get("BL_CRUISE_KMH", "12")) / 3.6
CORNER_SPEED_MPS = 8.0 / 3.6
BLIND_SPEED_MPS = 6.0 / 3.6
K_SPEED_P = 2.5
K_SPEED_D = 0.6
UNTRUSTED_VX_MPS = 0.4

_TURN_HDG_SCALE_DEG = 20.0
# De-noised: hdg is dead (0) most of the flight, so turn_mag was driven almost
# entirely by cx WEAVE noise (>0.7 for 77% of track time), permanently braking
# AND — via the thrust attenuation below — starving the gate-2 climb. Raised
# 0.7->1.1 so weave contributes less; the |hdg| term still flags real corners.
_TURN_CX_SCALE = 1.1
# Corner state persistence: raw turn_mag whipsaws with per-frame vision noise
# (log-verified 2.22<->3.33 v_target flicker mid-turn); latch the peak and
# decay it slowly so the brake holds through the whole corner.
TURN_MAG_DECAY_PER_S = 0.8  # latched turn_mag full->0 in 1.25 s
# Open-loop corner brake for sessions with no velocity telemetry (vX nan —
# the speed-PD is dead there): nose-up by this much at full turn_mag.
OPEN_BRAKE_DEG = 3.5  # full corner: pitch_des = -2 + 3.5 = +1.5 (real brake)

# --- YOLO gate assist (secondary; blue line stays primary) -------------------
GATE_ASSIST_MIN_RANGE_M = 2.5  # smoother blanks <2.0 m; margin vs bearing blow-up
GATE_ASSIST_MAX_RANGE_M = 35.0  # gates are 24-29 m apart; beyond = PnP noise
GATE_ASSIST_MAX_BEARING_DEG = 45.0  # half-FOV; beyond = clip artifact
GATE_ASSIST_FRAME_TIMEOUT_S = 0.5  # wall-clock guard: frozen camera = no assist
GATE_HINT_MEMORY_S = 3.0  # lost gate's bearing sign steers search this long
# Lost-line "gate" mode (replaces blind search when a fresh gate is visible).
GATE_STEER_YAW_GAIN = 1.0  # deg yaw_err per deg bearing
GATE_STEER_YAW_MAX_DEG = 40.0  # = global clamp
GATE_STEER_ROLL_GAIN = 0.5
GATE_STEER_BANK_MAX_DEG = 18.0  # < MAX_BANK 26
# Track-mode anticipatory bias (bounded, agreement-gated): start the turn
# early enough that the line never leaves the FOV in the first place. A gate
# may bias only when the line CONFIRMS the same-side bend, or when it is
# plausibly the on-path next gate (small bearing, far) — a straight line must
# never "agree" with a big-bearing det (log-verified phantom gates at
# +16..+42 deg / 9-31 m biased the drone RIGHT at the left corner).
GATE_BIAS_MIN_BEARING_DEG = 8.0  # engage only when gate clearly off-center
GATE_AGREE_MIN_HDG_DEG = 2.0  # line bend must be real to confirm a side
GATE_BIAS_ONPATH_DEG = 20.0  # straight-line anticipation: bearing cone
GATE_BIAS_ONPATH_MIN_RNG_M = 14.0  # ...and not point-blank (phantoms sit 9-13 m)
GATE_BIAS_YAW_GAIN = 0.3
GATE_BIAS_YAW_MAX_DEG = 10.0  # 1/4 of YAW_ERR_MAX — cannot fight a good line lock
GATE_BIAS_ROLL_GAIN = 0.25
GATE_BIAS_ROLL_MAX_DEG = 9.0
GATE_TURN_MAG_BEARING_DEG = 20.0  # |bearing|/this adds turn_mag → early slow-down
GATE_TURN_MAG_MAX = 1.0  # an agreeing gate bearing may brake fully to corner speed

# --- Race-start gating (GP semantics: only a FRESH countdown flies) ----------
CLOCK_RESET_SLACK_MS = 500  # sim_boot rewinding beyond this = manual reset
DISARM_PERSIST_S = 1.0  # ignore 1 Hz heartbeat armed-flag blips

# --- Post-collision reverse (mirrors gp_pilot BACKOFF, but time-primary) -----
# Trigger arming. mavlink_rx applies no throttle/id/severity filter, and the
# vendor README warns ground contact "fires hundreds of times per second while
# the drone is on the pad". So the trigger is armed by TIME SINCE GO, plus a
# floor-clearance veto when telemetry exists (inert when it does not).
REVERSE_ARM_AFTER_GO_S = 3.0  # pad chatter window; a hit this early is unreversible
REVERSE_MIN_CLEARANCE_M = 0.7  # NaN under the VQ2 block -> guard inert, never required
COLLISION_FRESH_S = 0.25  # wall-clock guard on `ts`, belt-and-braces with the pop
REVERSE_COOLDOWN_S = 2.0  # one reverse per hit, not a reverse loop
REVERSE_MAX_PER_ATTEMPT = 3  # after this, let the search ladder fly unassisted
# Severity gate OFF by default: MAVLink defines horizontal_minimum_delta as a
# DISTANCE (small = severe) but vendor train_controller.py treats the same field
# as an IMPULSE (large = severe). A threshold with the wrong sign either never
# fires or always fires — log the real distribution first, then enable.
REVERSE_MIN_DELTA = 0.0

# Reverse profile. vX is nan in every VQ2-blocked flight (log-verified 748/748
# rows), so the closed-loop regulator is the bonus path and the open-loop lean
# is what actually flies. CRUISE_PITCH_DEG=-2.0 gives ~12 km/h forward, so a
# symmetric +2.0 would not be "low speed": arrest hard first, then drift back.
REVERSE_ARREST_S = 0.5
REVERSE_ARREST_PITCH_DEG = 4.0  # = gp_pilot BACKOFF_PITCH_DEG; kills forward momentum
REVERSE_PITCH_DEG = 1.5  # then a gentle backwards lean
REVERSE_MAX_SPEED_MPS = 1.5  # ~5.4 km/h cap; only enforceable when vX is real
REVERSE_DIST_M = 3.0  # secondary exit, only when vX is real
REVERSE_MIN_S = 0.6  # never bail out under this
REVERSE_MAX_S = 2.5  # PRIMARY exit — shorter than gp's 4.0: we reverse BLIND
REVERSE_QUIET_S = 0.25  # never resume while collisions are still arriving
REVERSE_TRIM_ALPHA = 0.01  # ~3 s tau at 32 Hz; blueline's stand-in for gp's elev_i
REVERSE_TRIM_MIN = HOVER_THRUST - 0.02
REVERSE_TRIM_MAX = HOVER_THRUST + 0.06

# Re-acquisition = "sees the next gate CLEARLY". Counted in DISTINCT CAMERA
# FRAMES, never in control ticks: the loop runs ~32 Hz against a ~30 Hz camera,
# so tick counting would score one frame two or three times over.
REACQ_MIN_FRAMES = 4
REACQ_STABLE_S = 0.25  # ...and the streak must span at least this long
REACQ_GAP_S = 0.25  # a longer hole restarts the streak (continuous, not cumulative)
REACQ_MAX_BEARING_DEG = 25.0  # < GateAssist's 45: steerable, not peripheral
REACQ_MIN_RANGE_M = 3.0  # > GATE_ASSIST_MIN_RANGE_M: margin off the gate we just hit
REACQ_MAX_RANGE_M = 25.0  # < 35: the 30 m+ band was log-verified PnP noise
REACQ_MAX_LAG_FR = 3  # inference not badly behind the camera clock
REACQ_MIN_CONF = 0.60  # > gp_vision MIN_BOX_CONF 0.5
REACQ_MAX_REPROJ_PX = 6.0  # < gp_vision MAX_REPROJ_PX 10.0
# Rejects gate_pnp's "edge-pair" fallback, which reports n_visible=2 with a
# HARDCODED reproj_px=0.0 — a "perfect" reprojection there means nothing.
REACQ_MIN_N_VISIBLE = 3
REACQ_BEARING_JITTER_DEG = 15.0  # consecutive accepted frames = the same object
REACQ_RANGE_JITTER_M = 4.0
REACQ_LINE_CX_MAX = 0.7  # |cx| beyond this = corridor leaving the FOV, not a lock


def _turn_hint_sign(cx: float, hdg_rad: float) -> float:
    hdg_deg = math.degrees(hdg_rad)
    if abs(hdg_deg) >= abs(cx) * 30.0 and abs(hdg_deg) > 1e-3:
        return 1.0 if hdg_deg > 0.0 else -1.0
    if abs(cx) > 1e-4:
        return 1.0 if cx > 0.0 else -1.0
    return 1.0


def _read_att_deg(data: dict) -> tuple[float, float, float]:
    att = data.get("attitude")
    if att is not None:
        return (
            math.degrees(float(att["roll"])),
            math.degrees(float(att["pitch"])),
            math.degrees(float(att["yaw"])),
        )
    return 0.0, 0.0, 0.0


def _read_vx_body(data: dict) -> float:
    """Body-forward speed (m/s) from NED velocity + yaw."""
    vel = data.get("vel_ned")
    if vel is None:
        src = data.get("odometry") or data.get("local_position_ned")
        if src is None:
            return float("nan")
        vel = (float(src["vx"]), float(src["vy"]), float(src["vz"]))
    yaw = data.get("yaw_rad")
    if yaw is None:
        att = data.get("attitude")
        if att is None:
            return float("nan")
        yaw = float(att["yaw"])
    vx, vy, _ = float(vel[0]), float(vel[1]), float(vel[2])
    return float(vx * math.cos(yaw) + vy * math.sin(yaw))


class GateAssist:
    """YOLO-pose gates → bounded bearing estimate (secondary to the line).

    Wraps GateEstimateSmoother on a {pose, frame} view of the shared dict so
    its anduril/HSV fallbacks can never fire — YOLO-only by construction.
    Staleness (YOLO_STALE_GAP_FR), nearest-ahead pick, near-gate pass
    suppression, EMA and target-identity lock all come from the smoother.
    """

    def __init__(self):
        self._smoother = GateEstimateSmoother()

    def reset(self) -> None:
        self._smoother.reset()

    def update(self, data: dict, now: float) -> dict | None:
        pose = data.get("pose")
        frame = data.get("frame")
        if pose is None or frame is None:
            return None
        received_at = frame.get("received_at")
        if (
            received_at is not None
            and now - float(received_at) > GATE_ASSIST_FRAME_TIMEOUT_S
        ):
            return None  # camera feed frozen — frame-id staleness can't tell
        # Raw packet lag, before the smoother re-stamps onto the camera clock.
        lag_fr = None
        pose_fid = pose.get("frame_id")
        cam_fid = frame.get("frame_id")
        if pose_fid is not None and cam_fid is not None:
            lag_fr = int(cam_fid) - int(pose_fid)
        est = self._smoother.update({"pose": pose, "frame": frame})
        if est is None or est.get("source") != "yolo":
            return None
        bx = float(est["body_x_m"])
        by = float(est["body_y_m"])
        bz = float(est["body_z_m"])
        rng = math.sqrt(bx * bx + by * by + bz * bz)
        bearing_deg = math.degrees(math.atan2(by, bx))
        if (
            rng < GATE_ASSIST_MIN_RANGE_M
            or rng > GATE_ASSIST_MAX_RANGE_M
            or abs(bearing_deg) > GATE_ASSIST_MAX_BEARING_DEG
        ):
            return None
        return {
            "bearing_deg": bearing_deg,
            "range_m": rng,
            "infer_ms": est.get("infer_ms"),
            "lag_fr": lag_fr,
            "frame_id": est.get("frame_id"),
        }


class ReacquireMonitor:
    """Stable re-acquisition of the gate / line, for the post-collision reverse.

    A single flickering frame must never end the reverse, so a cue counts as
    re-acquired only after REACQ_MIN_FRAMES DISTINCT CAMERA FRAMES, spanning at
    least REACQ_STABLE_S, with no gap longer than REACQ_GAP_S and consistent
    bearing/range throughout. Distinct frame IDs, not ticks: the control loop
    runs ~32 Hz against a ~30 Hz camera, so counting ticks would score a single
    frame two or three times and "stability" would mean nothing.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._gate = {"fid": None, "n": 0, "t0": 0.0, "t": 0.0, "brg": 0.0, "rng": 0.0}
        self._line = {"fid": None, "n": 0, "t0": 0.0, "t": 0.0}

    @staticmethod
    def _restart(st: dict, now: float, **kw) -> bool:
        st.update(n=1, t0=now, t=now, **kw)
        return False

    def update_gate(self, gate: dict | None, pose: dict | None, now: float) -> bool:
        """True once a gate has been seen clearly and steadily enough to resume.

        `gate` is GateAssist's output (already YOLO-only, camera-fresh, range and
        bearing gated); `pose` is best_pose_gate()'s raw pick, used only for the
        detection-quality fields the smoother throws away.
        """
        st = self._gate
        if gate is None:
            return False
        fid = gate.get("frame_id")
        if fid is not None and fid == st["fid"]:
            return st["n"] >= REACQ_MIN_FRAMES and now - st["t0"] >= REACQ_STABLE_S
        brg, rng = float(gate["bearing_deg"]), float(gate["range_m"])
        lag = gate.get("lag_fr")
        ok = (
            abs(brg) <= REACQ_MAX_BEARING_DEG
            and REACQ_MIN_RANGE_M <= rng <= REACQ_MAX_RANGE_M
            and (lag is None or lag <= REACQ_MAX_LAG_FR)
            and _pose_quality_ok(pose)
        )
        st["fid"] = fid
        if not ok:
            st["n"] = 0
            return False
        # Continuity: a phantom that flickers in and out can never accumulate,
        # because a break restarts the streak at 1 rather than extending it.
        if (
            st["n"] == 0
            or now - st["t"] > REACQ_GAP_S
            or abs(brg - st["brg"]) > REACQ_BEARING_JITTER_DEG
            or abs(rng - st["rng"]) > REACQ_RANGE_JITTER_M
        ):
            return self._restart(st, now, brg=brg, rng=rng)
        st.update(n=st["n"] + 1, t=now, brg=brg, rng=rng)
        return st["n"] >= REACQ_MIN_FRAMES and now - st["t0"] >= REACQ_STABLE_S

    def update_line(self, vision: dict | None, now: float) -> bool:
        """True once the blue line is back, steadily and not just a corner sliver."""
        st = self._line
        if not (vision and vision.get("found")):
            return False
        fid = vision.get("frame_id")
        if fid is not None and fid == st["fid"]:
            return st["n"] >= REACQ_MIN_FRAMES and now - st["t0"] >= REACQ_STABLE_S
        st["fid"] = fid
        if abs(float(vision.get("cx_norm", 0.0))) > REACQ_LINE_CX_MAX:
            st["n"] = 0
            return False
        if st["n"] == 0 or now - st["t"] > REACQ_GAP_S:
            return self._restart(st, now)
        st.update(n=st["n"] + 1, t=now)
        return st["n"] >= REACQ_MIN_FRAMES and now - st["t0"] >= REACQ_STABLE_S


def _pose_quality_ok(pose: dict | None) -> bool:
    """Detection quality from the raw YOLO/PnP pick.

    The smoother's own `reliable`/`pnp_ok` are hardcoded True, so they say
    nothing; these are the fields that actually carry quality.
    """
    if pose is None:
        return False
    if float(pose.get("conf", 0.0)) < REACQ_MIN_CONF:
        return False
    p = pose.get("pose")
    if not p:
        return False
    return (
        int(p.get("n_visible", 0)) >= REACQ_MIN_N_VISIBLE
        and float(p.get("reproj_px", 1e9)) <= REACQ_MAX_REPROJ_PX
    )


def compute_blueline_guidance(
    *,
    vision: dict | None,
    lost_s: float,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    vX: float = float("nan"),
    dt: float = 1.0 / BL_CONTROL_HZ,
    hover_thrust: float = HOVER_THRUST,
    last_cx: float = 0.0,
    last_hdg: float = 0.0,
    state: dict | None = None,
    gate: dict | None = None,
    gate_hint_sign: float | None = None,
    gz_dps: float = float("nan"),
) -> tuple[float, float, float, float, dict]:
    """Vision + IMU → attitude-quat wire cmds (GP encoding)."""
    if state is None:
        state = {}
    dbg: dict = {"mode": "lost"}
    found = bool(vision and vision.get("found"))

    cx = hdg_rad = cy = dcx = 0.0
    turn_mag = 0.0
    if not found:
        # Rate state must not bridge a loss: a re-acquire cx jump would fire a
        # huge one-tick D spike otherwise.
        state.pop("prev_cx", None)
        state["dcx_ema"] = 0.0
    if found:
        cx = float(vision.get("cx_norm", 0.0))
        cy = float(vision.get("cy_norm", 0.0))
        hdg_rad = float(vision.get("heading_err", 0.0))
        hdg_deg = math.degrees(hdg_rad)
        # TODO(inner-edge-ab): shadow the estimator we are NOT flying. Logged
        # only — never fed to the control law. Defaults are nan, NOT the flown
        # value: a log row where the two agree must mean the estimators agreed,
        # not that the key was missing.
        dbg["cx_cent"] = float(vision.get("cx_centroid", float("nan")))
        dbg["hdg_cent"] = float(vision.get("heading_centroid", float("nan")))
        dbg["cx_edge"] = float(vision.get("cx_edge", float("nan")))
        dbg["bl_src"] = str(vision.get("source", ""))
        # EMA'd lateral rate for the roll D-term (bounded contribution).
        prev_cx = state.get("prev_cx")
        dcx_raw = 0.0 if prev_cx is None else (cx - prev_cx) / max(dt, 1e-3)
        state["prev_cx"] = cx
        dcx = DCX_EMA_ALPHA * dcx_raw + (1.0 - DCX_EMA_ALPHA) * float(
            state.get("dcx_ema", 0.0)
        )
        state["dcx_ema"] = dcx
        turn_mag = float(
            np.clip(
                abs(hdg_deg) / _TURN_HDG_SCALE_DEG + abs(cx) / _TURN_CX_SCALE,
                0.0,
                1.0,
            )
        )
        desired_roll = float(
            np.clip(
                K_ROLL_CX * cx
                + K_ROLL_HDG * hdg_deg
                + np.clip(K_ROLL_DCX * dcx, -DCX_ROLL_MAX_DEG, DCX_ROLL_MAX_DEG),
                -MAX_BANK_DEG,
                MAX_BANK_DEG,
            )
        )
        yaw_err = float(
            np.clip(
                K_YAW_CX * cx + K_YAW_HDG * hdg_deg,
                -YAW_ERR_MAX_DEG,
                YAW_ERR_MAX_DEG,
            )
        )
        # Anticipatory gate bias: a fresh off-center gate that agrees with the
        # line's bend starts the turn (and the slow-down) before the corner
        # tightens enough to sweep the line out of the FOV. Clamped small so it
        # can never fight a good line lock.
        if gate is not None:
            gb = float(gate["bearing_deg"])
            gate_seen = abs(gb) >= GATE_BIAS_MIN_BEARING_DEG
            if gate_seen:
                # Direction-neutral: ANY big-bearing det means a corner is
                # near — slowing early is safe even for a phantom det.
                turn_mag = max(
                    turn_mag,
                    min(abs(gb) / GATE_TURN_MAG_BEARING_DEG, GATE_TURN_MAG_MAX),
                )
                dbg["gate_slow"] = True
            # Directional steering bias stays strict: the line must confirm
            # the same-side bend, or the det must plausibly be the on-path
            # next gate (small bearing, not point-blank).
            bend_agrees = gb * hdg_deg > 0.0 and abs(hdg_deg) >= GATE_AGREE_MIN_HDG_DEG
            onpath_far = (
                abs(gb) <= GATE_BIAS_ONPATH_DEG
                and float(gate.get("range_m", 0.0)) >= GATE_BIAS_ONPATH_MIN_RNG_M
            )
            if gate_seen and (bend_agrees or onpath_far):
                yaw_err = float(
                    np.clip(
                        yaw_err
                        + np.clip(
                            GATE_BIAS_YAW_GAIN * gb,
                            -GATE_BIAS_YAW_MAX_DEG,
                            GATE_BIAS_YAW_MAX_DEG,
                        ),
                        -YAW_ERR_MAX_DEG,
                        YAW_ERR_MAX_DEG,
                    )
                )
                desired_roll = float(
                    np.clip(
                        desired_roll
                        + np.clip(
                            GATE_BIAS_ROLL_GAIN * gb,
                            -GATE_BIAS_ROLL_MAX_DEG,
                            GATE_BIAS_ROLL_MAX_DEG,
                        ),
                        -MAX_BANK_DEG,
                        MAX_BANK_DEG,
                    )
                )
                dbg["gate_bias"] = True
        v_target = 0.0  # set after the turn_mag latch below
        dbg["mode"] = "track"
    elif gate is not None:
        # Line lost but a fresh YOLO gate is in view: steer at the gate instead
        # of blind-searching. Corner crawl — this is real guidance, not a hunt.
        dbg["mode"] = "gate"
        gb = float(gate["bearing_deg"])
        yaw_err = float(
            np.clip(
                GATE_STEER_YAW_GAIN * gb,
                -GATE_STEER_YAW_MAX_DEG,
                GATE_STEER_YAW_MAX_DEG,
            )
        )
        desired_roll = float(
            np.clip(
                GATE_STEER_ROLL_GAIN * gb,
                -GATE_STEER_BANK_MAX_DEG,
                GATE_STEER_BANK_MAX_DEG,
            )
        )
        v_target = CORNER_SPEED_MPS
        cy = CY_TARGET
    else:
        sign = (
            gate_hint_sign
            if gate_hint_sign is not None
            else _turn_hint_sign(last_cx, last_hdg)
        )
        if lost_s < LOSS_HOLD_S:
            dbg["mode"] = "hold"
            yaw_err = sign * HOLD_YAW_ERR_DEG
        elif lost_s < SEARCH_AFTER_S:
            dbg["mode"] = "creep"
            yaw_err = sign * HOLD_YAW_ERR_DEG
        else:
            dbg["mode"] = "search"
            yaw_err = sign * SEARCH_YAW_ERR_DEG
        desired_roll = 0.0
        v_target = BLIND_SPEED_MPS
        cy = CY_TARGET

    # Latch the corner state: raw turn_mag whipsaws with per-frame vision noise
    # and drops to 0 on line loss — decay instead, so the brake holds through
    # the whole corner and across YOLO detection gaps (log-verified whipsaw).
    turn_mag = float(
        max(turn_mag, float(state.get("turn_mag_lat", 0.0)) - TURN_MAG_DECAY_PER_S * dt)
    )
    state["turn_mag_lat"] = turn_mag
    if dbg["mode"] == "track":
        v_target = CRUISE_SPEED_MPS + turn_mag * (CORNER_SPEED_MPS - CRUISE_SPEED_MPS)

    # Yaw-rate damping (see K_YAW_D_GZ): +K*gz_raw opposes the actual rotation
    # because this sim's gyros are sign-inverted. Applies in every mode.
    if not math.isnan(gz_dps):
        dbg["yaw_damp"] = K_YAW_D_GZ * gz_dps
        yaw_err = float(
            np.clip(yaw_err + K_YAW_D_GZ * gz_dps, -YAW_ERR_MAX_DEG, YAW_ERR_MAX_DEG)
        )

    cy_err = cy - CY_TARGET

    # Speed-PD → desired pitch (always when vX known); hard cap 25 km/h.
    if math.isnan(vX):
        pitch_des = CRUISE_PITCH_DEG if found else CREEP_PITCH_DEG
        pitch_des = pitch_des - K_PITCH_CY * cy_err * (1.0 - 0.5 * turn_mag)
        if turn_mag > 0.0:
            # No velocity feedback: brake open-loop. The old +1.5 deg*turn_mag
            # still left the nose DOWN at full corner (log-verified overspeed).
            pitch_des = pitch_des + turn_mag * OPEN_BRAKE_DEG
    else:
        prev_vx = state.get("prev_vx")
        a_fwd = 0.0 if prev_vx is None else (vX - prev_vx) / max(dt, 1e-3)
        state["prev_vx"] = vX
        # EMA the one-tick acceleration — raw 60 Hz differencing is too noisy
        # for a D-term (GP pilot pattern).
        a_fwd_ema = 0.3 * a_fwd + 0.7 * float(state.get("a_fwd_ema", 0.0))
        state["a_fwd_ema"] = a_fwd_ema
        pitch_des = (
            CRUISE_PITCH_DEG
            - K_SPEED_P * (v_target - vX)
            - K_SPEED_D * a_fwd_ema
            - K_PITCH_CY * cy_err * 0.5
        )
        if vX > MAX_SPEED_MPS * 1.15:
            pitch_des = PITCH_DES_MAX_DEG
        elif vX > MAX_SPEED_MPS:
            pitch_des = max(pitch_des, 0.5 * PITCH_DES_MAX_DEG)
        if abs(vX) < UNTRUSTED_VX_MPS:
            pitch_des = max(pitch_des, CRUISE_PITCH_DEG)

    pitch_des = float(np.clip(pitch_des, PITCH_DES_MIN_DEG, PITCH_DES_MAX_DEG))

    roll_cmd = float((desired_roll - roll_deg) * KR)
    pitch_cmd = float(
        np.clip((pitch_des - pitch_deg) * KP, -PITCH_WIRE_MAX_DEG, PITCH_WIRE_MAX_DEG)
    )
    yaw_cmd = float(yaw_err * KY)

    # Attenuation cut 0.7->0.3: the noise-pinned turn_mag was running the climb
    # correction at ~35% authority, and gate 2 is the STEEPEST-climb leg (log:
    # thrust never left 0.30 while the drone sat 0.5 low). Identical on straights
    # (turn_mag->0), so gate 1 (a level leg) is unaffected. K_THRUST_CY untouched.
    thrust = float(
        np.clip(
            hover_thrust - K_THRUST_CY * cy_err * (1.0 - 0.3 * turn_mag),
            THRUST_MIN,
            THRUST_MAX,
        )
    )

    dbg.update(
        cx=cx,
        cy=cy,
        hdg=hdg_rad,
        dcx=dcx,
        gate_bearing=None if gate is None else float(gate["bearing_deg"]),
        gate_range=None if gate is None else float(gate["range_m"]),
        turn_mag=turn_mag,
        desired_roll=desired_roll,
        yaw_err=yaw_err,
        pitch_des=pitch_des,
        v_target=v_target,
        vX=vX,
        roll=roll_cmd,
        pitch=pitch_cmd,
        yaw=yaw_cmd,
        thrust=thrust,
    )
    return roll_cmd, pitch_cmd, yaw_cmd, thrust, dbg


class BlueLinePilot:
    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self.n_passed = 0
        self._cmd_slew = CommandSlew(hz=BL_CONTROL_HZ, deg_s=BL_CMD_SLEW_DEG_S)
        self._lost_since: float | None = None
        self._last_vision: dict | None = None
        self._last_cx = 0.0
        self._last_hdg = 0.0
        self._hold: dict = {}
        self._tick = 0
        self._debug = os.environ.get("BL_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._gate_assist = GateAssist()
        self._last_gate_bearing_deg: float | None = None
        self._last_gate_seen_t = 0.0
        self._assist_enabled = os.environ.get(
            "BL_GATE_ASSIST", "1"
        ).strip().lower() not in ("0", "false", "no")
        # Post-collision reverse state.
        self._reacq = ReacquireMonitor()
        self._fly_since: float | None = None
        self._reverse_t0 = 0.0
        self._reverse_hit_t = 0.0
        self._reverse_end_t = -1e9
        self._reverse_dist = 0.0
        self._reverse_last_t = 0.0
        self._reverse_count = 0
        self._last_reverse_pitch_target = 0.0
        self._floor_z0: float | None = None
        # Blueline has no elevation integrator; raw HOVER_THRUST sits ~0.006
        # below true hover, which sank gp's every backoff. Track a slow EMA of
        # the thrust the cy loop has been commanding and reverse on that.
        self._hover_trim = HOVER_THRUST
        self._col_stats = {"n": 0, "ids": {}, "dmin": 1e9, "dmax": -1e9, "t": 0.0}
        # Race-start gate: hold until a fresh race exists, then fly at once.
        self._phase = "wait"
        self._wait_anchor_ms: int | None = None
        self._go_start_ms: int | None = None
        self._last_sim_ms: int | None = None
        self._disarm_since: float | None = None
        self._wait_note_printed = False
        self._wait_imu_ts_prev = None
        self._frozen_noted = False
        # Per-attempt CSV telemetry (rl/data/bl_log_*.csv).
        self._log = None
        self._log_wr = None
        self._log_last_flush = 0.0
        self._openloop_noted = False

        controller.control_hz = BL_CONTROL_HZ
        controller.set_control_mode("attitude_quat")
        controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
        print(
            f"[blueline] corridor pilot ready "
            f"(cruise {CRUISE_SPEED_MPS * 3.6:.0f} km/h, "
            f"max {MAX_SPEED_MPS * 3.6:.0f} km/h, GP wireform, "
            f"gate-assist={'on' if self._assist_enabled else 'off'}) — "
            f"holding for race start",
            flush=True,
        )

    @property
    def gates_passed(self) -> int:
        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self.n_passed:
            self.n_passed = active
        return self.n_passed

    def on_attempt_start(self) -> None:
        self.reset_for_attempt()

    def reset_for_attempt(self) -> None:
        self.n_passed = 0
        self._cmd_slew.reset()
        self._lost_since = None
        self._last_vision = None
        self._last_cx = 0.0
        self._last_hdg = 0.0
        self._hold = {}
        self._tick = 0
        self._gate_assist.reset()
        self._last_gate_bearing_deg = None
        self._last_gate_seen_t = 0.0
        self._disarm_since = None
        self._openloop_noted = False
        self._reacq.reset()
        self._reverse_count = 0
        self._reverse_end_t = -1e9
        self._reverse_dist = 0.0
        self._floor_z0 = None
        self._hover_trim = HOVER_THRUST
        self.data.pop("collision", None)
        self._close_log()
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, HOVER_THRUST)

    def shutdown(self) -> None:
        self._close_log()
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)

    # --- per-attempt CSV telemetry -------------------------------------------

    def _open_log(self) -> None:
        self._close_log()
        try:
            os.makedirs(os.path.join("rl", "data"), exist_ok=True)
            path = os.path.join("rl", "data", time.strftime("bl_log_%Y%m%d_%H%M%S.csv"))
            self._log = open(path, "w", newline="")
            self._log_wr = csv.writer(self._log)
            self._log_wr.writerow(
                "t mode bl_src cx cx_cent cx_edge cy hdg_deg hdg_cent_deg dcx "
                "turn_mag gate_brg gate_rng vX v_target "
                "pitch_des des_roll yaw_err cmd_roll cmd_pitch cmd_yaw thrust "
                "att_roll att_pitch gz_dps lost_s n_passed rev_n".split()
            )
            print(f"[blueline] flight log -> {path}", flush=True)
        except OSError as e:  # telemetry must never ground the pilot
            print(f"[blueline] flight log unavailable: {e}", flush=True)
            self._log, self._log_wr = None, None

    def _close_log(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
        self._log, self._log_wr = None, None

    def _log_tick(
        self, dbg, roll, pitch, yaw, thrust, att_roll, att_pitch, gz_dps, lost_s
    ) -> None:
        if self._log_wr is None:
            return
        try:
            now = time.time()
            gb = dbg.get("gate_bearing")
            gr = dbg.get("gate_range")
            self._log_wr.writerow(
                [f"{now:.3f}", str(dbg.get("mode", "")), str(dbg.get("bl_src", ""))]
                + [
                    f"{dbg.get(k, float('nan')):.3f}"
                    for k in ("cx", "cx_cent", "cx_edge", "cy")
                ]
                + [
                    f"{math.degrees(dbg.get('hdg', 0.0)):.2f}",
                    f"{math.degrees(dbg.get('hdg_cent', 0.0)):.2f}",
                    f"{dbg.get('dcx', 0.0):.3f}",
                    f"{dbg.get('turn_mag', 0.0):.3f}",
                    "" if gb is None else f"{gb:.2f}",
                    "" if gr is None else f"{gr:.2f}",
                ]
                + [
                    f"{dbg.get(k, float('nan')):.3f}"
                    for k in ("vX", "v_target", "pitch_des", "desired_roll", "yaw_err")
                ]
                + [f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}", f"{thrust:.4f}"]
                + [f"{att_roll:.2f}", f"{att_pitch:.2f}", f"{gz_dps:.2f}"]
                + [f"{lost_s:.2f}", str(self.gates_passed), str(self._reverse_count)]
            )
            if now - self._log_last_flush >= 1.0:
                self._log.flush()
                self._log_last_flush = now
        except (OSError, ValueError, KeyError):
            self._close_log()

    # --- collision intake ----------------------------------------------------

    def _drain_collision(self, now_wall: float) -> dict | None:
        """Pop the collision key — the SINGLE consumer — and fold it into stats.

        Called every tick in BOTH phases, including "wait". Draining while we
        hold on the pad is what guarantees the key is empty at GO, so ground
        chatter cannot arm the very first fly tick. mavlink_rx overwrites the
        key in place and never clears it, so an unconsumed hit would otherwise
        sit in the dict looking fresh forever.
        """
        col = self.data.pop("collision", None)
        if col is None:
            return None
        st = self._col_stats
        st["n"] += 1
        cid = col.get("id")
        st["ids"][cid] = st["ids"].get(cid, 0) + 1
        d = float(col.get("delta", 0.0))
        st["dmin"], st["dmax"] = min(st["dmin"], d), max(st["dmax"], d)
        if now_wall - st["t"] >= 1.0:
            # Rate-limited: ground contact can arrive hundreds of times a second.
            print(
                f"[blueline] COLLISION x{st['n']} phase={self._phase} "
                f"ids={st['ids']} delta=[{st['dmin']:.3f},{st['dmax']:.3f}] "
                f"threat={col.get('threat_level')} type={col.get('type')}",
                flush=True,
            )
            st.update(n=0, ids={}, dmin=1e9, dmax=-1e9, t=now_wall)
        return col

    def _floor_clearance(self) -> float:
        """Height above the GO-time floor, or nan when position telemetry is off."""
        src = self.data.get("local_position_ned") or self.data.get("odometry")
        if src is None or src.get("z") is None:
            return float("nan")
        z = float(src["z"])
        if self._floor_z0 is None:
            self._floor_z0 = z
        return self._floor_z0 - z  # NED: z is down

    def _collision_should_reverse(self, col, now: float, now_wall: float) -> bool:
        if col is None:
            return False
        ts = col.get("ts")
        if ts is not None and now_wall - float(ts) > COLLISION_FRESH_S:
            return False  # stale packet
        if self._fly_since is None or now - self._fly_since < REVERSE_ARM_AFTER_GO_S:
            return False  # still in the pad-chatter window
        if now - self._reverse_end_t < REVERSE_COOLDOWN_S:
            return False  # one reverse per hit, not a loop
        if self._reverse_count >= REVERSE_MAX_PER_ATTEMPT:
            return False  # stop reversing; let the ladder fly
        clr = self._floor_clearance()
        if not math.isnan(clr) and clr < REVERSE_MIN_CLEARANCE_M:
            return False  # on the ground: reversing achieves nothing
        if REVERSE_MIN_DELTA > 0.0 and abs(float(col.get("delta", 0.0))) < (
            REVERSE_MIN_DELTA
        ):
            return False  # inert by default — see REVERSE_MIN_DELTA
        return True

    # --- post-collision reverse ----------------------------------------------

    def _enter_reversing(self, now: float) -> None:
        self._phase = "reversing"
        self._reverse_t0 = self._reverse_last_t = self._reverse_hit_t = now
        self._reverse_dist = 0.0
        self._reverse_count += 1
        # Post-impact vision state is garbage: the latched turn_mag, the cx
        # derivative, the speed-PD history and the smoother's pre-crash target
        # lock would all pollute the re-acquire. _last_cx/_last_hdg and the gate
        # hint are deliberately KEPT — we hold heading through the reverse, so
        # the pre-impact turn direction is still the best hint for the ladder.
        self._hold = {}
        self._cmd_slew.reset()
        self._gate_assist.reset()
        self._reacq.reset()
        self._last_vision = None
        self._lost_since = None
        self.data.pop("collision", None)
        print(
            f"[blueline] COLLISION #{self._reverse_count} — reversing "
            f"(max {REVERSE_MAX_S:.1f}s) until the gate or line is back",
            flush=True,
        )

    def _tick_reversing(self, now: float, col: dict | None = None) -> None:
        elapsed = now - self._reverse_t0
        if col is not None:
            self._reverse_hit_t = now  # still in contact — hold off resuming
        roll_deg, pitch_deg, _yaw = _read_att_deg(self.data)
        vX = _read_vx_body(self.data)
        step_dt = max(1.0 / BL_CONTROL_HZ, now - self._reverse_last_t)
        self._reverse_last_t = now
        vx_ok = not math.isnan(vX)
        if vx_ok:
            self._reverse_dist += max(0.0, -vX) * step_dt

        # POSITIVE pitch = nose-up = REVERSE (CRUISE_PITCH_DEG=-2.0 is forward).
        if vx_ok:
            # gp's 3-band regulator: a fixed nose-up ran away to 20-30 km/h.
            rev = max(0.0, -vX)
            if rev > REVERSE_MAX_SPEED_MPS:
                pitch_target = 0.0
            elif rev > 0.8 * REVERSE_MAX_SPEED_MPS:
                pitch_target = 0.3 * REVERSE_PITCH_DEG
            else:
                pitch_target = REVERSE_ARREST_PITCH_DEG
        else:
            # Open loop (vX dead): arrest the forward run, then drift back.
            pitch_target = (
                REVERSE_ARREST_PITCH_DEG
                if elapsed < REVERSE_ARREST_S
                else REVERSE_PITCH_DEG
            )

        self._last_reverse_pitch_target = pitch_target  # inspected by tests/debug
        pitch_cmd = float(
            np.clip(
                (pitch_target - pitch_deg) * KP,
                -PITCH_WIRE_MAX_DEG,
                PITCH_WIRE_MAX_DEG,
            )
        )
        roll_cmd = float((0.0 - roll_deg) * KR)  # level
        yaw_cmd = 0.0  # hold heading — backing up blind while rotating clips things

        vision = self.data.get("blue_line")
        thrust = self._hover_trim
        if vision and vision.get("found"):
            thrust = self._hover_trim - K_THRUST_CY * (
                float(vision.get("cy_norm", CY_TARGET)) - CY_TARGET
            )
        thrust = float(np.clip(thrust, THRUST_MIN, THRUST_MAX))

        gate = (
            self._gate_assist.update(self.data, now) if self._assist_enabled else None
        )
        if gate is not None:
            self._last_gate_bearing_deg = float(gate["bearing_deg"])
            self._last_gate_seen_t = now
        gate_clear = self._reacq.update_gate(gate, best_pose_gate(self.data), now)
        line_back = self._reacq.update_line(vision, now)

        roll_cmd, pitch_cmd, yaw_cmd, thrust = self._cmd_slew.apply(
            roll_cmd, pitch_cmd, yaw_cmd, thrust
        )
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        self.data["bl_gate_assist"] = {
            "mode": "reverse",
            "bearing_deg": None if gate is None else gate["bearing_deg"],
            "range_m": None if gate is None else gate["range_m"],
            "infer_ms": None if gate is None else gate.get("infer_ms"),
            "lag_fr": None if gate is None else gate.get("lag_fr"),
            "hint_sign": None,
        }
        imu = self.data.get("imu") or {}
        self._log_tick(
            {
                "mode": "reverse",
                "pitch_des": pitch_target,
                "vX": vX,
                "gate_bearing": None if gate is None else gate["bearing_deg"],
                "gate_range": None if gate is None else gate["range_m"],
            },
            roll_cmd,
            pitch_cmd,
            yaw_cmd,
            thrust,
            roll_deg,
            pitch_deg,
            math.degrees(float(imu.get("gz", float("nan")))),
            0.0,
        )

        if elapsed < REVERSE_MIN_S or now - self._reverse_hit_t < REVERSE_QUIET_S:
            return
        if gate_clear or line_back:
            self._resume_flying("gate" if gate_clear else "line", now, elapsed)
        elif (vx_ok and self._reverse_dist >= REVERSE_DIST_M) or (
            elapsed >= REVERSE_MAX_S
        ):
            # Time-primary: vX is nan under the VQ2 block, so _reverse_dist stays
            # pinned at 0.0 and a distance-only exit would never fire.
            self._resume_flying("timeout", now, elapsed)

    def _resume_flying(self, why: str, now: float, elapsed: float) -> None:
        print(
            f"[blueline] reverse done ({why}) t={elapsed:.1f}s "
            f"dist={self._reverse_dist:.1f}m — resuming",
            flush=True,
        )
        self._phase = "fly"
        self._reverse_end_t = now
        self._hold = {}
        self._cmd_slew.reset()
        self._reacq.reset()
        # Clearing both re-enters the existing ladder at "hold" next tick, which
        # then walks to creep/search on its own timers. _gate_assist is NOT reset:
        # it has been tracking through the reverse and its lock is the freshest
        # thing we have.
        self._last_vision = None
        self._lost_since = None
        self.data.pop("collision", None)

    # --- race-start gating ---------------------------------------------------

    def _publish_hold_state(self, mode: str) -> None:
        self.data["bl_gate_assist"] = {
            "mode": mode,
            "bearing_deg": None,
            "range_m": None,
            "infer_ms": None,
            "lag_fr": None,
            "hint_sign": None,
        }

    def _abort_to_wait(self, reason: str) -> None:
        print(f"[blueline] {reason} — holding for a fresh race start.", flush=True)
        self._phase = "wait"
        self._wait_anchor_ms = None
        self._go_start_ms = None
        self._last_sim_ms = None
        self._disarm_since = None
        self._wait_imu_ts_prev = None
        self._frozen_noted = False
        self._fly_since = None  # re-arms the post-GO collision grace
        self._reacq.reset()
        self._close_log()
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
        self._publish_hold_state("wait")

    def _tick_wait(self) -> None:
        """Hold zero thrust until a race that started AFTER US exists.

        Mirrors GPPilot's WAIT_FOR_START anchor (a stale already-running race
        never flies) but does NOT wait out the sim's 3-2-1 — the user wants
        wheels-up immediately after each manual Restart Race. When the sim
        keeps the drone frozen through its countdown, the physics-live gate
        below holds naturally (commands do nothing to frozen physics anyway).
        """
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.0)
        self._publish_hold_state("wait")
        race = self.data.get("race_status")
        if race is None:
            if not self._wait_note_printed:
                self._wait_note_printed = True
                print(
                    "[blueline] waiting for race — start/Restart Race in the sim.",
                    flush=True,
                )
            return
        sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
        start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
        finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
        if self._wait_anchor_ms is None:
            self._wait_anchor_ms = sim_ms
        elif sim_ms < self._wait_anchor_ms - CLOCK_RESET_SLACK_MS:
            # Manual reset while already holding: chase the rewound clock.
            self._wait_anchor_ms = sim_ms
        fresh = start_ms > 0 and start_ms >= self._wait_anchor_ms
        if not fresh or finish_ns >= 0:
            return
        # Physics-live gate: a stale race can idle the sim physics — IMU keeps
        # streaming but its sensor clock freezes and commands do nothing
        # (log-verified: a 6.7 s attempt with gz pinned at 0). Require the IMU
        # clock to have advanced since the previous tick before flying.
        imu = self.data.get("imu")
        if imu is not None:
            ts = imu.get("time_us")
            prev = self._wait_imu_ts_prev
            self._wait_imu_ts_prev = ts
            if ts is not None and (prev is None or ts == prev):
                if not self._frozen_noted and prev is not None:
                    self._frozen_noted = True
                    print(
                        "[blueline] sim physics is IDLE — click Restart Race.",
                        flush=True,
                    )
                return
        self._go_start_ms = start_ms
        self._last_sim_ms = sim_ms
        self._frozen_noted = False
        self.reset_for_attempt()
        self._open_log()
        self._phase = "fly"
        self._fly_since = time.monotonic()  # arms the collision trigger, +grace
        print("[blueline] GO — race started.", flush=True)

    def _should_abort_flying(self) -> str | None:
        race = self.data.get("race_status")
        if race is not None:
            sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
            start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
            finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
            if (
                self._last_sim_ms is not None
                and sim_ms < self._last_sim_ms - CLOCK_RESET_SLACK_MS
            ):
                return "sim clock reset"
            self._last_sim_ms = sim_ms
            if (
                self._go_start_ms is not None
                and start_ms > 0
                and start_ms != self._go_start_ms
                and finish_ns < 0
            ):
                return "new race start"
            if finish_ns >= 0:
                return "race finished"
        if not bool(self.data.get("armed", False)):
            now = time.time()
            if self._disarm_since is None:
                self._disarm_since = now
            elif now - self._disarm_since >= DISARM_PERSIST_S:
                return "disarmed (sim reset?)"
        else:
            self._disarm_since = None
        return None

    def tick(self) -> None:
        self._tick += 1
        now, now_wall = time.monotonic(), time.time()
        if self._phase == "wait":
            self._drain_collision(now_wall)  # keep pad chatter from arming GO
            self._tick_wait()
            return
        # Abort stays FIRST so a hard-crash disarm still wins over a reverse in
        # progress (the sim disarms on a real crash; you cannot reverse then).
        reason = self._should_abort_flying()
        if reason is not None:
            self._abort_to_wait(reason)
            return
        col = self._drain_collision(now_wall)
        if self._phase == "reversing":
            self._tick_reversing(now, col)
            return
        if self._collision_should_reverse(col, now, now_wall):
            self._enter_reversing(now)
            self._tick_reversing(now)
            return
        vision = self.data.get("blue_line")
        if vision and vision.get("found"):
            self._last_vision = vision
            self._last_cx = float(vision.get("cx_norm", 0.0))
            self._last_hdg = float(vision.get("heading_err", 0.0))
            self._lost_since = None
            lost_s = 0.0
            use = vision
        else:
            if self._lost_since is None:
                self._lost_since = now
            lost_s = now - self._lost_since
            # Do not replay stale centered frames through corners — only a
            # short hold, then signed search from last cx/hdg.
            if lost_s < LOSS_HOLD_S and self._last_vision is not None:
                use = self._last_vision
                lost_s = 0.0
            else:
                use = None

        gate = (
            self._gate_assist.update(self.data, now) if self._assist_enabled else None
        )
        if gate is not None:
            self._last_gate_bearing_deg = float(gate["bearing_deg"])
            self._last_gate_seen_t = now
        # Recently-lost gate still tells us which way the course bends — its
        # bearing sign beats _turn_hint_sign's heuristic (default-right) search.
        gate_hint_sign = None
        if (
            gate is None
            and self._last_gate_bearing_deg is not None
            and now - self._last_gate_seen_t <= GATE_HINT_MEMORY_S
            and abs(self._last_gate_bearing_deg) > 1e-6
        ):
            gate_hint_sign = 1.0 if self._last_gate_bearing_deg > 0.0 else -1.0

        roll_deg, pitch_deg, _yaw_meas = _read_att_deg(self.data)
        vX = _read_vx_body(self.data)
        imu = self.data.get("imu") or {}
        gz_dps = math.degrees(float(imu.get("gz", float("nan"))))
        if not self._openloop_noted and self._tick >= 60:
            self._openloop_noted = True
            if math.isnan(vX) and self.data.get("attitude") is None:
                print(
                    "[blueline] no attitude/velocity telemetry — open-loop "
                    "speed control (VQ2 block?)",
                    flush=True,
                )

        roll, pitch, yaw, thrust, dbg = compute_blueline_guidance(
            vision=use,
            lost_s=lost_s,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            vX=vX,
            dt=1.0 / BL_CONTROL_HZ,
            hover_thrust=HOVER_THRUST,
            last_cx=self._last_cx,
            last_hdg=self._last_hdg,
            state=self._hold,
            gate=gate,
            gate_hint_sign=gate_hint_sign,
            gz_dps=gz_dps,
        )
        self.data["bl_gate_assist"] = {
            "mode": dbg.get("mode"),
            "bearing_deg": None if gate is None else gate["bearing_deg"],
            "range_m": None if gate is None else gate["range_m"],
            "infer_ms": None if gate is None else gate.get("infer_ms"),
            "lag_fr": None if gate is None else gate.get("lag_fr"),
            "hint_sign": gate_hint_sign,
        }
        roll, pitch, yaw, thrust = self._cmd_slew.apply(roll, pitch, yaw, thrust)
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(roll, pitch, yaw, thrust)
        # Learn the hover thrust the cy loop has been holding, for the reverse to
        # fly on. Never reset when entering the reverse — that is the point.
        self._hover_trim = float(
            np.clip(
                self._hover_trim + REVERSE_TRIM_ALPHA * (thrust - self._hover_trim),
                REVERSE_TRIM_MIN,
                REVERSE_TRIM_MAX,
            )
        )
        self._log_tick(
            dbg, roll, pitch, yaw, thrust, roll_deg, pitch_deg, gz_dps, lost_s
        )

        if self._debug and self._tick % 30 == 0:
            vx_s = f"{vX:+.2f}" if not math.isnan(vX) else "nan"
            if gate is not None:
                inf = gate.get("infer_ms")
                lag = gate.get("lag_fr")
                gate_s = (
                    f"{gate['bearing_deg']:+.1f}d r={gate['range_m']:.1f}m "
                    f"yolo={0.0 if inf is None else float(inf):.0f}ms "
                    f"lag={'?' if lag is None else lag}f"
                )
            else:
                gate_s = "none"
            print(
                f"[blueline] {dbg.get('mode')} cx={dbg.get('cx', 0):+.2f} "
                f"hdg={dbg.get('hdg', 0):+.2f} vX={vx_s} gate={gate_s} "
                f"cmd=({roll:+.1f},{pitch:+.1f},{yaw:+.1f}) "
                f"att=({roll_deg:+.1f},{pitch_deg:+.1f})",
                flush=True,
            )

        _ = self.gates_passed
