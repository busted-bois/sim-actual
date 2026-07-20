"""Vision adapter → AndurilGP-style vision_gate_estimate fields.

GPPilot (make control-flight) prefers the YOLO-pose packet (8-keypoint PnP,
aims at the gate opening) whenever inference is keeping up with the camera.
Anduril HSV-red detection and the legacy HSV gate_target remain as fallbacks
for stale/absent YOLO frames and for tools that do not run GatePoseRunner.
"""

from __future__ import annotations

import math

import numpy as np

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
# Pass-through suppression (Anduril-tracker parity for the YOLO path).
YOLO_NEAR_BX_M = 2.0  # gate centre closer than this → threading it: go blind
YOLO_NEAR_BOX_FRAC = 0.85  # box filling this fraction of the frame = on top of gate
YOLO_PASS_COOLDOWN_FR = 10  # camera frames to stay blind after a near-gate pass


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
    best = None
    best_cost = float("inf")
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
        bearing = abs(math.atan2(gb[1], gb[0]))
        cost = rng + BEARING_COST_M_PER_RAD * bearing
        if cost < best_cost:
            best_cost = cost
            best = g
    return best


def _yolo_pose_estimate(data: dict) -> dict | None:
    """Fresh YOLO pose packet → estimate dict (no suppression logic)."""
    pose_pkt = data.get("pose") or {}
    fid = pose_pkt.get("frame_id")
    if fid is None:
        return None
    latest = (data.get("frame") or {}).get("frame_id")
    if latest is not None and latest - fid > YOLO_STALE_GAP_FR:
        return None  # inference fell behind — don't servo on an old world
    g = best_pose_gate(data)
    if g is None:
        return None
    p = g["pose"]
    gb = np.asarray(p["gate_pos_body"], dtype=np.float64).reshape(3)
    return {
        "frame_id": fid,
        "body_x_m": float(gb[0]),
        "body_y_m": float(gb[1]),
        "body_z_m": float(gb[2]),
        "pnp_ok": True,
        "pnp_rvec": None,
        "normal_body": np.asarray(p["normal_body"], dtype=np.float64).reshape(3),
        "u_px": None,
        "v_px": None,
        "reliable": True,
        "source": "yolo",
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

    def update(self, data: dict) -> tuple[dict | None, bool]:
        """(estimate, suppress). suppress=True means a gate is being threaded —
        the caller must return None without falling back to other sources."""
        pose_pkt = data.get("pose") or {}
        fid = pose_pkt.get("frame_id")
        if fid is None:
            return None, False
        latest = (data.get("frame") or {}).get("frame_id")
        ref_fid = latest if latest is not None else fid
        stale = latest is not None and latest - fid > YOLO_STALE_GAP_FR
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
        return _yolo_pose_estimate(data), False


def vision_gate_estimate(
    data: dict, yolo_tracker: YoloGateTracker | None = None
) -> dict | None:
    """Build Anduril-compatible vision estimate.

    Preference order for GPPilot:
      1. YOLO pose packet — 8-keypoint PnP aimed at the opening, when fresh
      2. data["anduril_gate"] — HSV-red detect_gate + PnP/pinhole
      3. HSV gate_target rays / pinhole

    With a yolo_tracker, near-gate pass-through suppression blanks ALL
    sources so the pilot flies through blind instead of chasing the near
    edge with a lower-grade detector.
    """
    if yolo_tracker is not None:
        est, suppress = yolo_tracker.update(data)
        if suppress:
            return None
    else:
        est = _yolo_pose_estimate(data)
    if est is not None:
        return est

    anduril = data.get("anduril_gate")
    if anduril is not None and anduril.get("body_x_m") is not None:
        bx = float(anduril["body_x_m"])
        by = float(anduril["body_y_m"])
        bz = float(anduril["body_z_m"])
        if not any(math.isnan(v) for v in (bx, by, bz)):
            out = dict(anduril)
            out.setdefault("source", "anduril")
            out.setdefault("reliable", True)
            return out

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

    def update(self, data: dict) -> dict | None:
        est = vision_gate_estimate(data, self.yolo_tracker)
        if est is None:
            self._clear_track()
            return None
        # Dedup on the source's NATIVE id before re-stamping — one output per
        # underlying measurement even when the camera clock advances.
        src_key = (est.get("source"), est.get("frame_id"))
        if src_key == self._last_src_key:
            return self._last_out
        self._last_src_key = src_key
        # Publish on the camera clock: yolo carries the (older) pose-packet
        # fid while anduril carries the camera fid, so raw fids REGRESS on
        # anduril->yolo flips — silently bypassing the EMA gap guard and the
        # guidance D-term guards (log-verified sawtooth).
        cam_fid = (data.get("frame") or {}).get("frame_id")
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
            return self._last_out
        vec = np.array(
            [est["body_x_m"], est["body_y_m"], est["body_z_m"]], dtype=np.float64
        )
        if self._ema is not None:
            prev_fid, pbx, pby, pbz = self._ema
            prev_vec = np.array([pbx, pby, pbz], dtype=np.float64)
            if not _same_target(vec, prev_vec):
                return self._on_target_break(est, vec, prev_vec, fid)
            self._pending, self._pending_n = None, 0
            if fid is not None and 0 < fid - prev_fid <= EMA_MAX_FRAME_GAP:
                a = GATE_EMA_ALPHA
                est["body_x_m"] = a * est["body_x_m"] + (1.0 - a) * pbx
                est["body_y_m"] = a * est["body_y_m"] + (1.0 - a) * pby
                est["body_z_m"] = a * est["body_z_m"] + (1.0 - a) * pbz
        if fid is not None:
            self._ema = (fid, est["body_x_m"], est["body_y_m"], est["body_z_m"])
        self._last_out = est
        return est

    def _on_target_break(self, est, vec, prev_vec, fid):
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
            return self._last_out
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
