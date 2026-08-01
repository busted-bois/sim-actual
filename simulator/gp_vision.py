"""Vision adapter → AndurilGP-style vision_gate_estimate fields.

GPPilot (make control-flight) prefers the YOLO-pose packet (8-keypoint PnP,
aims at the gate opening) whenever inference is keeping up with the camera.
Anduril HSV-red detection and the legacy HSV gate_target remain as fallbacks
for stale/absent YOLO frames and for tools that do not run GatePoseRunner.
"""

from __future__ import annotations

import math
import os

import numpy as np


def env_f(name: str, default: float) -> float:
    """Env-overridable float, ignoring anything unparseable."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_flag(name: str, default: bool = True) -> bool:
    """Env-overridable on/off switch. Everything but 0/false/no/off is on."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


GATE_OUTER_W_M = 2.7
FX = 320.0
CX = 320.0
CY = 180.0
IMG_W = 640
IMG_H = 360
CAM_TILT_DEG = 20.0
MIN_BOX_CONF = 0.5
MAX_REPROJ_PX = 10.0

# YOLO-primary source selection.
# 5 (not 3): inference sits right at the 3-frame boundary, which made the
# source alternate yolo<->anduril nearly every tick (log-verified sawtooth).
YOLO_STALE_GAP_FR = 5  # pose packet this many frames behind the camera = stale
BEARING_COST_M_PER_RAD = 5.0  # nearest-ahead pick: range + this per rad off-axis
# Only ever consider the N nearest gates. A gate 3-4 course-lengths out can
# briefly outscore the one dead ahead at handoff (big box, low bearing) and
# yank the aim toward it while we're threading the near gate — the log-verified
# gate-2 clip. The next gate we could legitimately switch to is always the
# nearest or 2nd-nearest, so the far ones are pure distractors: drop them.
MAX_GATES_CONSIDERED = 2
# Pass-through suppression (Anduril-tracker parity for the YOLO path).
YOLO_NEAR_BX_M = 2.0  # gate centre closer than this → threading it: go blind
YOLO_NEAR_BOX_FRAC = 0.85  # box filling this fraction of the frame = on top of gate
YOLO_PASS_COOLDOWN_FR = 10  # camera frames to stay blind after a near-gate pass

# --- Source continuity (measured on gp_log_20260728_192003) -------------------
# The ladder is re-picked every camera frame, and there is no NVIDIA GPU here
# (AMD integrated; torch reports cuda_available=False), so YOLO runs on CPU at
# 100-350 ms/frame = 3-10 camera frames of lag and YOLO_STALE_GAP_FR trips
# constantly. Result: 15.9 source changes/s, 82% of them gate<->gate, median
# lock streak 0.17 s. The estimators disagree about the SAME gate by far more
# than the gate moves between frames:
#     within one source   |dby| p50 0.000  p90 0.074 m
#     across a switch     |dby| p50 0.180  p90 0.809 m   (|dbx| p90 2.25 m)
#
# GateEstimateSmoother already snaps + sets track_break on a flip, which covers
# the DERIVATIVE. Nothing covered the PROPORTIONAL term: at the median lock
# range (bx 8.9 m) a 0.809 m step is 5.18 deg of bearing = 12.9 deg of roll
# demand out of a 14 deg cap, applied in whichever direction the two estimators
# happened to disagree. Three mechanisms, each independently switchable:
#
#   PROPAGATE  keep a stale YOLO packet alive by advancing it with body motion,
#              so the sawtooth never starts (fixes the outbound half).
#   STICKY     once fallen back, require N consecutive frames of the preferred
#              source before returning to it (fixes the return half).
#   BLEND      when a switch does happen, publish the old position and bleed the
#              offset to zero, so the P term sees a ramp instead of a step.
GP_YOLO_PROPAGATE = env_flag("GP_YOLO_PROPAGATE", True)
YOLO_PROPAGATE_MAX_GAP_FR = 12  # past this the packet is too old to trust at all
GP_SRC_STICKY_FR = int(env_f("GP_SRC_STICKY_FR", 3))  # 0 disables
GP_SRC_BLEND_S = env_f("GP_SRC_BLEND_S", 0.3)  # 0 disables
SRC_BLEND_MAX_M = 3.0  # a bigger jump is a different gate, not an offset
CAM_HZ = 30.0


def gate_body_from_pinhole(
    u_px: float, v_px: float, box_w_px: float, gate_w_m: float = GATE_OUTER_W_M
) -> tuple[float, float, float] | None:
    """Pinhole range from outer width + cam-tilt → body FRD (bx, by, bz)."""
    if box_w_px < 1.0:
        return None
    z_cam = (gate_w_m * FX) / box_w_px
    x_cam = (u_px - CX) / FX * z_cam
    y_cam = (v_px - CY) / FX * z_cam
    c = math.cos(math.radians(CAM_TILT_DEG))
    s = math.sin(math.radians(CAM_TILT_DEG))
    bx = c * z_cam + s * y_cam
    by = x_cam
    bz = -s * z_cam + c * y_cam
    return float(bx), float(by), float(bz)


def best_pose_gate(data: dict) -> dict | None:
    """Quality-filtered YOLO gate most likely to be the NEXT gate.

    Highest-confidence is the wrong pick at gate handoff — a big far gate can
    outscore the one dead ahead. Pick the nearest gate with a mild bearing
    penalty (BEARING_COST_M_PER_RAD) so the chase stays on the course line.
    """
    pose_pkt = data.get("pose") or {}
    gates = pose_pkt.get("gates") or []
    # Collect valid, in-front gates with their range, then restrict to the
    # MAX_GATES_CONSIDERED nearest before scoring — a far gate is never a
    # legitimate next target and only serves to distract the selector.
    candidates = []
    for g in gates:
        p = g.get("pose")
        if not p:
            continue
        conf = float(g.get("conf", 0.0))
        if conf < MIN_BOX_CONF:
            continue
        reproj = float(p.get("reproj_px", 0.0))
        if reproj > MAX_REPROJ_PX:
            continue
        gb = np.asarray(p["gate_pos_body"], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(gb)) or gb[0] <= 0.1:
            continue
        rng = float(np.linalg.norm(gb))
        candidates.append((rng, g, gb))
    candidates.sort(key=lambda c: c[0])
    best = None
    best_cost = float("inf")
    for rng, g, gb in candidates[:MAX_GATES_CONSIDERED]:
        bearing = abs(math.atan2(gb[1], gb[0]))
        cost = rng + BEARING_COST_M_PER_RAD * bearing
        if cost < best_cost:
            best_cost = cost
            best = g
    return best


def _propagate_body(gb: np.ndarray, gap_fr: int, motion: dict | None) -> np.ndarray:
    """Advance a gate's body-frame position by our own motion over `gap_fr`.

    The gate is static in the world, so between the frame YOLO looked at and
    now, its BODY-frame position moved by exactly minus our own displacement,
    plus the rotation our yaw applied to the frame itself. Without a motion
    estimate this is the identity — a stale packet held in place, which is
    still closer to the truth than an estimator carrying a ~0.8 m offset.
    """
    if not motion or gap_fr <= 0:
        return gb
    dt = gap_fr / CAM_HZ
    vx = float(motion.get("vX", 0.0) or 0.0)
    vy = float(motion.get("vY", 0.0) or 0.0)
    vd = float(motion.get("vD", 0.0) or 0.0)
    if not all(math.isfinite(v) for v in (vx, vy, vd)):
        return gb
    out = gb - np.array([vx * dt, vy * dt, vd * dt], dtype=np.float64)
    # Yaw rotates the body frame under the (world-static) gate.
    wz = float(motion.get("yaw_rate_rps", 0.0) or 0.0)
    if math.isfinite(wz) and wz != 0.0:
        a = -wz * dt
        ca, sa = math.cos(a), math.sin(a)
        out = np.array(
            [ca * out[0] - sa * out[1], sa * out[0] + ca * out[1], out[2]],
            dtype=np.float64,
        )
    return out


def _yolo_pose_estimate(data: dict, motion: dict | None = None) -> dict | None:
    """YOLO pose packet → estimate dict (no suppression logic).

    Past YOLO_STALE_GAP_FR the packet is not discarded outright: with inference
    on CPU it is stale most of the time, and dropping it is what hands the pilot
    to a different estimator 13 times a second. Instead it is advanced by our own
    motion (`motion`) out to YOLO_PROPAGATE_MAX_GAP_FR and flagged `propagated`.
    """
    pose_pkt = data.get("pose") or {}
    fid = pose_pkt.get("frame_id")
    if fid is None:
        return None
    latest = (data.get("frame") or {}).get("frame_id")
    gap = 0 if latest is None else int(latest - fid)
    propagated = False
    if gap > YOLO_STALE_GAP_FR:
        if not GP_YOLO_PROPAGATE or gap > YOLO_PROPAGATE_MAX_GAP_FR:
            return None  # inference fell behind — don't servo on an old world
        propagated = True
    g = best_pose_gate(data)
    if g is None:
        return None
    p = g["pose"]
    gb = np.asarray(p["gate_pos_body"], dtype=np.float64).reshape(3)
    if propagated:
        gb = _propagate_body(gb, gap, motion)
        if not np.all(np.isfinite(gb)) or gb[0] <= 0.1:
            return None
    return {
        "frame_id": fid,
        "body_x_m": float(gb[0]),
        "body_y_m": float(gb[1]),
        "body_z_m": float(gb[2]),
        "propagated": propagated,
        "pnp_ok": True,
        "pnp_rvec": None,
        "normal_body": np.asarray(p["normal_body"], dtype=np.float64).reshape(3),
        "u_px": None,
        "v_px": None,
        "reliable": True,
        "source": "yolo",
        # PnP method: "edge-pair" is a single-edge fallback whose normal is a
        # fixed camera-tilt placeholder, not a measured plane — consumers that
        # read the full normal vector (RL obs) must not trust it as a real
        # orientation. The GP guidance ignores the normal's z so it never cared.
        "method": p.get("method"),
        "infer_ms": pose_pkt.get("infer_ms"),
    }


class YoloGateTracker:
    """Pass-through suppression for the YOLO path.

    The Anduril tracker goes blind while threading a gate (huge red area) and
    holds a cooldown after, so the pilot never yanks toward the just-passed
    gate's edge. YOLO needs the same: when any confident gate is very close
    (pose bx < YOLO_NEAR_BX_M) or its box fills the frame, publish nothing and
    tell the caller to suppress ALL sources, then stay blind for
    YOLO_PASS_COOLDOWN_FR camera frames.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._in_gate = False
        self._suppress_until_fid: int | None = None

    @staticmethod
    def _near_gate(gates) -> bool:
        for g in gates:
            if float(g.get("conf", 0.0)) < MIN_BOX_CONF:
                continue
            p = g.get("pose")
            if p is not None:
                gb = np.asarray(p["gate_pos_body"], dtype=np.float64).reshape(3)
                if np.all(np.isfinite(gb)) and 0.0 < float(gb[0]) < YOLO_NEAR_BX_M:
                    return True
            b = g.get("box")
            if b is not None:
                b = np.asarray(b, dtype=np.float64).reshape(-1)
                if (
                    b[2] - b[0] >= YOLO_NEAR_BOX_FRAC * IMG_W
                    or b[3] - b[1] >= YOLO_NEAR_BOX_FRAC * IMG_H
                ):
                    return True
        return False

    def force_cooldown(self, fid: int | None) -> None:
        """External pass signal (near->far target handoff detected by the
        smoother): hold the same blind cooldown as a detected near-gate pass."""
        self._in_gate = False
        if fid is not None:
            self._suppress_until_fid = fid + YOLO_PASS_COOLDOWN_FR

    def update(
        self, data: dict, motion: dict | None = None
    ) -> tuple[dict | None, bool]:
        """(estimate, suppress). suppress=True means a gate is being threaded —
        the caller must return None without falling back to other sources."""
        pose_pkt = data.get("pose") or {}
        fid = pose_pkt.get("frame_id")
        if fid is None:
            return None, False
        latest = (data.get("frame") or {}).get("frame_id")
        ref_fid = latest if latest is not None else fid
        # "Too old to use at all" — which now means past the PROPAGATION
        # ceiling, not past YOLO_STALE_GAP_FR. Tying near-gate suppression to
        # the same window a propagated packet is trusted over keeps the
        # pass-through blind spot intact: a packet good enough to steer on is
        # good enough to notice we are on top of the gate.
        horizon = YOLO_PROPAGATE_MAX_GAP_FR if GP_YOLO_PROPAGATE else YOLO_STALE_GAP_FR
        stale = latest is not None and latest - fid > horizon
        if not stale and self._near_gate(pose_pkt.get("gates") or []):
            self._in_gate = True
            return None, True
        if self._in_gate:
            self._in_gate = False
            self._suppress_until_fid = ref_fid + YOLO_PASS_COOLDOWN_FR
        if self._suppress_until_fid is not None:
            if ref_fid < self._suppress_until_fid:
                return None, True
            self._suppress_until_fid = None
        return _yolo_pose_estimate(data, motion), False


def _anduril_estimate(data: dict) -> dict | None:
    """data["anduril_gate"] — HSV-red detect_gate + PnP/pinhole."""
    anduril = data.get("anduril_gate")
    if anduril is None or anduril.get("body_x_m") is None:
        return None
    bx = float(anduril["body_x_m"])
    by = float(anduril["body_y_m"])
    bz = float(anduril["body_z_m"])
    if any(math.isnan(v) for v in (bx, by, bz)):
        return None
    out = dict(anduril)
    out.setdefault("source", "anduril")
    out.setdefault("reliable", True)
    return out


def _hsv_estimate(data: dict) -> dict | None:
    """Legacy HSV gate_target — rays when ranged, else pinhole on the box."""
    pose_pkt = data.get("pose") or {}
    frame_id = pose_pkt.get("frame_id")
    gt = data.get("gate_target") or {}
    if not gt.get("detected"):
        return None
    frame_id = gt.get("frame_id", frame_id)
    u = gt.get("u_px")
    v = gt.get("v_px")
    range_m = gt.get("range_m")
    bearing = gt.get("bearing_rad")
    elev = gt.get("elevation_rad")

    if range_m is not None and bearing is not None and elev is not None:
        br = float(bearing)
        el = float(elev)
        R = float(range_m)
        bx = R * math.cos(el) * math.cos(br)
        by = R * math.cos(el) * math.sin(br)
        bz = -R * math.sin(el)
        return {
            "frame_id": frame_id,
            "body_x_m": bx,
            "body_y_m": by,
            "body_z_m": bz,
            "pnp_ok": False,
            "pnp_rvec": None,
            "normal_body": None,
            "u_px": u,
            "v_px": v,
            "reliable": False,
            "source": "hsv",
        }

    if u is None or v is None:
        return None
    r_frac = float(gt.get("r_frac") or 0.0)
    box_w = math.sqrt(max(r_frac, 1e-6) * 640.0 * 360.0)
    body = gate_body_from_pinhole(float(u), float(v), box_w)
    if body is None:
        return None
    bx, by, bz = body
    return {
        "frame_id": frame_id,
        "body_x_m": bx,
        "body_y_m": by,
        "body_z_m": bz,
        "pnp_ok": False,
        "pnp_rvec": None,
        "normal_body": None,
        "u_px": float(u),
        "v_px": float(v),
        "reliable": False,
        "source": "hsv",
    }


def gate_candidates(
    data: dict,
    yolo_tracker: YoloGateTracker | None = None,
    motion: dict | None = None,
) -> tuple[dict[str, dict], bool]:
    """({source: estimate}, suppress) for every source that resolved this frame.

    Ladder order is the dict's insertion order, so `next(iter(...))` is the
    plain preference pick. Returning ALL of them (rather than the first) is what
    lets the stabilizer ask "is the source I already committed to still
    available?" — the question hysteresis needs and the old short-circuit
    ladder could not answer.
    """
    if yolo_tracker is not None:
        yolo, suppress = yolo_tracker.update(data, motion)
        if suppress:
            return {}, True
    else:
        yolo, suppress = _yolo_pose_estimate(data, motion), False

    out: dict[str, dict] = {}
    for est in (yolo, _anduril_estimate(data), _hsv_estimate(data)):
        if est is not None:
            out.setdefault(str(est.get("source")), est)
    return out, suppress


def vision_gate_estimate(
    data: dict,
    yolo_tracker: YoloGateTracker | None = None,
    motion: dict | None = None,
) -> dict | None:
    """Build Anduril-compatible vision estimate.

    Preference order for GPPilot:
      1. YOLO pose packet — 8-keypoint PnP aimed at the opening, fresh OR
         motion-propagated within YOLO_PROPAGATE_MAX_GAP_FR (see `motion`)
      2. data["anduril_gate"] — HSV-red detect_gate + PnP/pinhole
      3. HSV gate_target rays / pinhole

    With a yolo_tracker, near-gate pass-through suppression blanks ALL
    sources so the pilot flies through blind instead of chasing the near
    edge with a lower-grade detector.
    """
    cands, _suppress = gate_candidates(data, yolo_tracker, motion)
    return next(iter(cands.values()), None)


def gate_tilt_deg_from_normal(normal_body: np.ndarray) -> float:
    """Yaw offset (deg) between body-forward and the gate's fly-through axis.

    Sign-agnostic in the normal's facing: the live detectors point it BACK at
    the drone while rl.gp_expert's synthetic one points forward, so flip onto
    the +x half-space first and a head-on gate reads ~0 either way. The
    previous atan2(n[1], n[0]) put a detector (back-facing) normal at ±180° ->
    clipped to a sign-flapping ±30° yaw dither near gates — dormant while HSV
    rarely solved a normal, armed when YOLO (always has one) became primary.
    """
    n = np.asarray(normal_body, dtype=np.float64).reshape(3)
    fx, fy = float(n[0]), float(n[1])
    if fx < 0.0:
        fx, fy = -fx, -fy
    return float(np.clip(math.degrees(math.atan2(fy, fx)), -30.0, 30.0))


GATE_EMA_ALPHA = 0.35
PNP_STICKY_FRAMES = 5
# YOLO may skip camera frames when inference lags; the EMA/derivative code
# scales by the actual frame gap, so allow up to 6 (200 ms) before resetting.
EMA_MAX_FRAME_GAP = 6
# Target-identity gate (log-verified failure): with two gates in view the
# per-frame selector / detector fallbacks can hand the smoother a DIFFERENT
# gate tick-to-tick, and EMA-blending across that sweeps the aim point
# through empty space toward the gate edge. A jump is an identity change
# (not measurement noise) when the range ratio or the 3D direction to the
# gate moves beyond same-gate noise; depth noise moves range but not
# direction, so direction is the sharper discriminator.
BREAK_RANGE_RATIO = 1.6
BREAK_DIRECTION_DEG = 15.0
BREAK_CONFIRM_N = 3  # frames a new target must persist before we follow it
# An accepted near->far switch means we just crossed the old gate's plane:
# treat it as a gate pass (blind cooldown), not a retarget — this is the
# source-agnostic version of the YOLO near-gate suppression.
HANDOFF_NEAR_BX_M = 3.0
HANDOFF_DBX_M = 3.0


def _same_target(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two gate body positions plausibly describe the SAME gate."""
    ra = float(np.linalg.norm(a))
    rb = float(np.linalg.norm(b))
    if ra < 1e-6 or rb < 1e-6:
        return True
    if max(ra, rb) / min(ra, rb) > BREAK_RANGE_RATIO:
        return False
    cosang = float(np.clip(np.dot(a, b) / (ra * rb), -1.0, 1.0))
    return math.degrees(math.acos(cosang)) <= BREAK_DIRECTION_DEG


class GateEstimateSmoother:
    """EMA + PnP-sticky source selection + target-identity lock.

    Publishes at most one estimate per underlying detector measurement, on a
    monotonic camera-frame clock, and never blends across a target switch.
    """

    def __init__(self):
        self.yolo_tracker = YoloGateTracker()
        self.reset()

    def reset(self) -> None:
        self._clear_track()
        self.yolo_tracker.reset()

    def _clear_track(self) -> None:
        """Drop the current track WITHOUT touching the yolo_tracker.

        Calling full reset() on every None estimate wiped the tracker's
        in-gate/cooldown state each suppressed tick, which made the post-pass
        blind cooldown dead code — the pilot re-acquired the just-passed
        gate's edge with zero blind window (log-verified collisions).
        """
        self._ema: tuple[int, float, float, float] | None = None
        self._last_pnp_fid: int | None = None
        self._last_out: dict | None = None
        self._last_src_key: tuple | None = None
        self._pending: np.ndarray | None = None
        self._pending_n = 0
        self._src: str | None = None  # source currently committed to
        self._challenger: str | None = None
        self._challenger_n = 0
        self._offset: np.ndarray | None = None  # A4: bleeding-off switch step
        self._offset_end = 0  # camera fid the ramp finishes on
        self._offset_span = 0

    def _pick(self, cands: dict[str, dict]) -> dict | None:
        """Hysteresis over the preference ladder.

        The ladder's own first choice churns: with YOLO stale most frames the
        preferred source alternates yolo/anduril nearly every tick. Stay on the
        committed source while it is still resolving, and require the preferred
        one to hold for GP_SRC_STICKY_FR consecutive frames before going back.
        """
        if not cands:
            self._challenger, self._challenger_n = None, 0
            return None
        preferred = next(iter(cands))
        if GP_SRC_STICKY_FR <= 0 or self._src is None or self._src == preferred:
            self._challenger, self._challenger_n = None, 0
            return cands[preferred]
        if self._src not in cands:
            # Committed source produced nothing — no choice but to move.
            self._challenger, self._challenger_n = None, 0
            return cands[preferred]
        # Both available and the ladder prefers the other one: make it wait.
        if self._challenger == preferred:
            self._challenger_n += 1
        else:
            self._challenger, self._challenger_n = preferred, 1
        if self._challenger_n >= GP_SRC_STICKY_FR:
            self._challenger, self._challenger_n = None, 0
            return cands[preferred]
        return cands[self._src]

    def _start_blend(
        self, prev_vec: np.ndarray, vec: np.ndarray, cam_fid: int | None
    ) -> None:
        """Capture the step a source switch just introduced, to bleed off.

        Snapping is right for the target (the new source IS the better estimate)
        but wrong for the CONTROLLER, which reads the snap as a real lateral
        error and answers with up to 12.9 deg of roll out of a 14 deg cap. Publish
        the old position and walk it to the new one over GP_SRC_BLEND_S.
        """
        # Continue from where we are CURRENTLY publishing, not from the raw
        # previous source. Switches come closer together than the ramp is long
        # (median lock streak 0.17 s vs a 0.3 s ramp), and restarting from the
        # raw value makes the published jump BIGGER than no ramp at all.
        residual = np.zeros(3, dtype=np.float64)
        if (
            self._offset is not None
            and cam_fid is not None
            and cam_fid < self._offset_end
        ):
            residual = self._offset * (
                (self._offset_end - cam_fid) / self._offset_span
            )
        self._offset = None
        self._offset_end = self._offset_span = 0
        span = int(round(GP_SRC_BLEND_S * CAM_HZ))
        if span <= 0 or cam_fid is None:
            return
        delta = (prev_vec + residual) - vec
        if not np.all(np.isfinite(delta)):
            return
        if float(np.linalg.norm(delta)) > SRC_BLEND_MAX_M:
            return  # too big to be the same gate seen differently
        self._offset = delta
        self._offset_span = span
        self._offset_end = cam_fid + span

    def _present(self, est: dict | None, cam_fid: int | None) -> dict | None:
        """Publish a copy carrying the decaying switch offset.

        Runs on the CAMERA clock, not on publication events, and on every return
        path including the dedup short-circuit. YOLO only publishes once per
        inference (~9 camera frames on CPU), so a ramp advanced per-publication
        stalls at full offset for the whole stale window — which is exactly the
        span the ramp exists to cover.

        `track_break` is re-asserted for the whole ramp: it already means "do not
        differentiate across this" to both compute_guidance and
        VisionVelocityTracker, and a ramp is exactly as unsafe to differentiate
        as the step it replaced (0.8 m over 0.3 s reads as 2.7 m/s).
        """
        if est is None or self._offset is None:
            return est
        if cam_fid is None or cam_fid >= self._offset_end:
            self._offset = None
            return est
        frac = (self._offset_end - cam_fid) / self._offset_span
        out = dict(est)
        out["body_x_m"] = est["body_x_m"] + float(self._offset[0]) * frac
        out["body_y_m"] = est["body_y_m"] + float(self._offset[1]) * frac
        out["body_z_m"] = est["body_z_m"] + float(self._offset[2]) * frac
        out["src_blend"] = True
        out["track_break"] = True
        return out

    def update(self, data: dict, motion: dict | None = None) -> dict | None:
        cands, _suppress = gate_candidates(data, self.yolo_tracker, motion)
        est = self._pick(cands)
        if est is None:
            self._clear_track()
            return None
        # Read the camera clock up front: every return path below has to present
        # through _present(), and the blend ramp is measured on this clock.
        cam_fid = (data.get("frame") or {}).get("frame_id")
        # Dedup on the source's NATIVE id before re-stamping — one output per
        # underlying measurement even when the camera clock advances.
        src_key = (est.get("source"), est.get("frame_id"))
        if src_key == self._last_src_key:
            return self._present(self._last_out, cam_fid)
        self._last_src_key = src_key
        # Publish on the camera clock: yolo carries the (older) pose-packet
        # fid while anduril carries the camera fid, so raw fids REGRESS on
        # anduril->yolo flips — silently bypassing the EMA gap guard and the
        # guidance D-term guards (log-verified sawtooth).
        if cam_fid is not None:
            est["frame_id"] = cam_fid
        fid = est.get("frame_id")
        if est.get("pnp_ok"):
            self._last_pnp_fid = fid
        elif (
            fid is not None
            and self._last_pnp_fid is not None
            and 0 < fid - self._last_pnp_fid <= PNP_STICKY_FRAMES
            and est.get("source") != "anduril"
        ):
            # Sticky YOLO only — Anduril already has its own EMA/pass-suppress.
            return self._present(self._last_out, cam_fid)
        vec = np.array(
            [est["body_x_m"], est["body_y_m"], est["body_z_m"]], dtype=np.float64
        )
        if self._last_out is not None and est.get("source") != self._last_out.get(
            "source"
        ):
            # Estimator handoff (yolo <-> anduril/hsv): their systematic
            # offsets differ by up to ~0.9 m vertically on clipped views, and a
            # SAME-gate step passes _same_target (~8 deg), so the EMA would
            # sweep the aim point through the offset. Snap to the new source
            # instead. BUT a flip that also lands on a DIFFERENT gate must run
            # the 3-frame identity debounce + near->far pass cooldown via
            # _on_target_break — snapping blindly here re-opened the "bank
            # toward the far gate while threading the near one" crash.
            if self._ema is not None:
                pbx, pby, pbz = self._ema[1], self._ema[2], self._ema[3]
                prev_vec = np.array([pbx, pby, pbz], dtype=np.float64)
                if not _same_target(vec, prev_vec):
                    return self._on_target_break(est, vec, prev_vec, fid, cam_fid)
                # Same gate, different estimator: the snap below is right for
                # the target but is a step for the controller. Ramp it instead.
                self._start_blend(prev_vec, vec, cam_fid)
            est["track_break"] = True
            self._ema = None
            self._pending, self._pending_n = None, 0
        if self._ema is not None:
            prev_fid, pbx, pby, pbz = self._ema
            prev_vec = np.array([pbx, pby, pbz], dtype=np.float64)
            if not _same_target(vec, prev_vec):
                return self._on_target_break(est, vec, prev_vec, fid, cam_fid)
            self._pending, self._pending_n = None, 0
            if fid is not None and 0 < fid - prev_fid <= EMA_MAX_FRAME_GAP:
                a = GATE_EMA_ALPHA
                est["body_x_m"] = a * est["body_x_m"] + (1.0 - a) * pbx
                est["body_y_m"] = a * est["body_y_m"] + (1.0 - a) * pby
                est["body_z_m"] = a * est["body_z_m"] + (1.0 - a) * pbz
        if fid is not None:
            self._ema = (fid, est["body_x_m"], est["body_y_m"], est["body_z_m"])
        # _last_out / _ema track the TRUE new-source position; only the
        # published copy carries the decaying switch offset, so the ramp never
        # feeds back into the identity checks or the EMA history.
        self._src = est.get("source")
        self._last_out = est
        return self._present(est, cam_fid)

    def _on_target_break(self, est, vec, prev_vec, fid, cam_fid=None):
        """A different gate showed up: hold the incumbent until the challenger
        persists BREAK_CONFIRM_N frames, then snap to it (never blend across
        identities). Alternating incumbent/challenger frames (two detectors on
        two gates) reset the count, so flapping can never steal the lock."""
        if self._pending is not None and _same_target(vec, self._pending):
            self._pending_n += 1
        else:
            self._pending = vec
            self._pending_n = 1
        if self._pending_n < BREAK_CONFIRM_N:
            return self._present(self._last_out, cam_fid)
        self._pending, self._pending_n = None, 0
        if (
            prev_vec[0] < HANDOFF_NEAR_BX_M
            and vec[0] - prev_vec[0] > HANDOFF_DBX_M
        ):
            # Near gate replaced by a far one => we crossed its plane. Go
            # blind for the pass cooldown instead of instantly chasing the
            # next gate at the old gate's edge.
            self.yolo_tracker.force_cooldown(fid)
            self._clear_track()
            return None
        est["track_break"] = True  # guidance must drop derivative history
        if fid is not None:
            self._ema = (fid, est["body_x_m"], est["body_y_m"], est["body_z_m"])
        else:
            self._ema = None
        # A different GATE, not a different view of the same one: there is no
        # offset to bleed off, and carrying one over would drag the aim back
        # toward the gate we just stopped tracking.
        self._offset, self._offset_end = None, 0
        self._src = est.get("source")
        self._last_out = est
        return est


class VisionVelocityTracker:
    """Optical-flow body velocity from consecutive gate body positions."""

    def __init__(self):
        self._prev: tuple[int | None, float, float, float] | None = None
        self.last_velocity: dict | None = None

    def reset(self) -> None:
        self._prev = None
        self.last_velocity = None

    def update(self, estimate: dict | None, cam_hz: float = 30.0) -> dict | None:
        if estimate is None:
            self._prev = None
            self.last_velocity = None
            return None
        if estimate.get("track_break"):
            # New gate: differencing across identities makes phantom velocity
            # (a 1 m switch over one frame reads as ~30 m/s).
            self.reset()
        fid = estimate.get("frame_id")
        bx = estimate["body_x_m"]
        by = estimate["body_y_m"]
        bz = estimate["body_z_m"]
        if any(math.isnan(v) for v in (bx, by, bz)):
            self.last_velocity = None
            return None
        if self._prev is not None and fid is not None:
            prev_fid, pbx, pby, pbz = self._prev
            if prev_fid is not None and 0 < fid - prev_fid <= EMA_MAX_FRAME_GAP:
                dt = (fid - prev_fid) / cam_hz
                self.last_velocity = {
                    "vx_body_mps": -(bx - pbx) / dt,
                    "vy_body_mps": -(by - pby) / dt,
                    "vz_body_mps": -(bz - pbz) / dt,
                }
            else:
                self.last_velocity = None
        else:
            self.last_velocity = None
        if fid is not None:
            self._prev = (fid, bx, by, bz)
        return self.last_velocity
