"""Vision adapter → AndurilGP-style vision_gate_estimate fields.

Try 2 (classical hole): prefer HSV/CV inner-opening (u,v)+width over YOLO
PnP body pose. Anduril / YOLO / HSV gate_target remain fallbacks.
"""

from __future__ import annotations

import math

import numpy as np

GATE_OUTER_W_M = 2.7
GATE_INNER_W_M = 1.5  # hole width for CV pinhole range
FX = 320.0
CX = 320.0
CY = 180.0
CAM_TILT_DEG = 20.0
MIN_BOX_CONF = 0.5
MAX_REPROJ_PX = 10.0
KP_CONF = 0.7
AIM_V = CY + FX * math.tan(math.radians(CAM_TILT_DEG))  # IBVS aim line


def gate_body_from_pinhole(
    u_px: float, v_px: float, box_w_px: float, gate_w_m: float = GATE_OUTER_W_M
) -> tuple[float, float, float] | None:
    """Pinhole range from apparent width + cam-tilt → body FRD (bx, by, bz)."""
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


def cv_opening_pixels(gate: dict) -> tuple[float, float, float] | None:
    """(u, v, inner_spread_px) from CV hole corners, or None."""
    cv_c = gate.get("cv_corners")
    if cv_c is None:
        return None
    pts = np.asarray(cv_c, float).reshape(-1, 2)
    if pts.shape[0] < 4 or not np.isfinite(pts).all():
        return None
    ctr = pts.mean(axis=0)
    spread = float(pts[:, 0].max() - pts[:, 0].min())
    if spread < 1.0:
        return None
    return float(ctr[0]), float(ctr[1]), spread


def yolo_opening_pixels(gate: dict) -> tuple[float, float, float] | None:
    """(u, v, inner_spread_px) from CV corners, else INNER KPs, else box."""
    pix = cv_opening_pixels(gate)
    if pix is not None:
        return pix
    box = gate.get("box")
    kxy = gate.get("keypoints")
    kcf = gate.get("keypoint_conf")
    if kxy is not None and kcf is not None:
        kxy = np.asarray(kxy, float)
        kcf = np.asarray(kcf, float)
        if kxy.shape[0] >= 4 and kcf.shape[0] >= 4:
            inner = kxy[:4]
            ok = kcf[:4] > KP_CONF
            if ok.sum() >= 2 and np.isfinite(inner[ok]).all():
                pts = inner[ok]
                ctr = pts.mean(axis=0)
                spread = float(pts[:, 0].max() - pts[:, 0].min())
                if ok.sum() < 3 and box is not None:
                    b = np.asarray(box, float).reshape(-1)
                    if b.size >= 4 and np.isfinite(b).all():
                        spread = max(spread, float(b[2] - b[0]) * 0.55)
                return float(ctr[0]), float(ctr[1]), spread
    if box is None:
        return None
    b = np.asarray(box, float).reshape(-1)
    if b.size < 4 or not np.isfinite(b).all():
        return None
    return (
        float((b[0] + b[2]) / 2.0),
        float((b[1] + b[3]) / 2.0),
        float(b[2] - b[0]) * 0.55,
    )


def best_cv_hole_gate(data: dict) -> dict | None:
    """Best gate that carries CV inner-opening corners."""
    pose_pkt = data.get("pose") or {}
    best = None
    best_conf = -1.0
    for g in pose_pkt.get("gates") or []:
        if cv_opening_pixels(g) is None:
            continue
        conf = float(g.get("conf", 0.0))
        if conf > best_conf:
            best_conf = conf
            best = g
    return best


def best_pose_gate(data: dict) -> dict | None:
    """Best YOLO gate with a solved, quality-filtered pose from data['pose']."""
    pose_pkt = data.get("pose") or {}
    gates = pose_pkt.get("gates") or []
    best = None
    best_conf = -1.0
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
        if conf > best_conf:
            best_conf = conf
            best = g
    return best


def _cv_hole_estimate(data: dict) -> dict | None:
    """Classical hole: centroid + spread → pinhole body (inner 1.5 m)."""
    pose_pkt = data.get("pose") or {}
    frame_id = pose_pkt.get("frame_id")
    g = best_cv_hole_gate(data)
    if g is None:
        return None
    pix = cv_opening_pixels(g)
    if pix is None:
        return None
    u_px, v_px, spread = pix
    body = gate_body_from_pinhole(u_px, v_px, spread, gate_w_m=GATE_INNER_W_M)
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
        "u_px": u_px,
        "v_px": v_px,
        "inner_spread_px": spread,
        "reliable": True,
        "source": "cv",
        "method": "hsv-hole",
        "opening_ok": True,
    }


def _yolo_estimate(data: dict) -> dict | None:
    pose_pkt = data.get("pose") or {}
    frame_id = pose_pkt.get("frame_id")
    g = best_pose_gate(data)
    if g is None:
        return None
    p = g["pose"]
    gb = np.asarray(p["gate_pos_body"], dtype=np.float64).reshape(3)
    u_px = v_px = inner_spread_px = None
    pix = yolo_opening_pixels(g)
    if pix is not None:
        u_px, v_px, inner_spread_px = pix
    return {
        "frame_id": frame_id,
        "body_x_m": float(gb[0]),
        "body_y_m": float(gb[1]),
        "body_z_m": float(gb[2]),
        "pnp_ok": True,
        "pnp_rvec": None,
        "normal_body": np.asarray(p["normal_body"], dtype=np.float64).reshape(3),
        "u_px": u_px,
        "v_px": v_px,
        "inner_spread_px": inner_spread_px,
        "reliable": True,
        "source": "yolo",
        "method": p.get("method", "ippe-yolo8"),
        "opening_ok": True,
    }


def _anduril_estimate(data: dict) -> dict | None:
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
    out.setdefault("inner_spread_px", None)
    out["opening_ok"] = False
    return out


def _hsv_estimate(data: dict, frame_id=None) -> dict | None:
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
            "inner_spread_px": None,
            "reliable": False,
            "source": "hsv",
            "opening_ok": False,
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
        "inner_spread_px": None,
        "reliable": False,
        "source": "hsv",
        "opening_ok": False,
    }


def vision_gate_estimate(data: dict) -> dict | None:
    """Build Anduril-compatible vision estimate.

    Preference order (Try 2 — classical hole first):
      1. CV HSV hole (u,v)+spread → pinhole (ignore YOLO PnP body)
      2. data["anduril_gate"]
      3. YOLO pose packet
      4. HSV gate_target rays / pinhole
    """
    cv = _cv_hole_estimate(data)
    if cv is not None:
        return cv

    anduril = _anduril_estimate(data)
    if anduril is not None:
        return anduril

    yolo = _yolo_estimate(data)
    if yolo is not None:
        return yolo

    pose_pkt = data.get("pose") or {}
    return _hsv_estimate(data, frame_id=pose_pkt.get("frame_id"))


def gate_tilt_deg_from_normal(normal_body: np.ndarray) -> float:
    """Body-XY angle of gate normal (Anduril PnP tilt proxy)."""
    n = np.asarray(normal_body, dtype=np.float64).reshape(3)
    return float(np.clip(math.degrees(math.atan2(n[1], n[0])), -30.0, 30.0))


GATE_EMA_ALPHA = 0.35
PNP_STICKY_FRAMES = 5
EMA_MAX_FRAME_GAP = 3


class GateEstimateSmoother:
    """EMA over the gate body position + PnP-sticky source selection."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._ema: tuple[int, float, float, float] | None = None
        self._last_pnp_fid: int | None = None
        self._last_out: dict | None = None

    def update(self, data: dict) -> dict | None:
        est = vision_gate_estimate(data)
        if est is None:
            self.reset()
            return None
        fid = est.get("frame_id")
        if fid is not None and self._ema is not None and fid == self._ema[0]:
            return self._last_out
        if est.get("pnp_ok"):
            self._last_pnp_fid = fid
        elif (
            fid is not None
            and self._last_pnp_fid is not None
            and 0 < fid - self._last_pnp_fid <= PNP_STICKY_FRAMES
            and est.get("source") not in ("anduril", "cv")
        ):
            # Sticky YOLO only — Anduril/CV already have fresh opening estimates.
            return self._last_out
        if fid is not None and self._ema is not None:
            prev_fid, pbx, pby, pbz = self._ema
            if 0 < fid - prev_fid <= EMA_MAX_FRAME_GAP:
                a = GATE_EMA_ALPHA
                est["body_x_m"] = a * est["body_x_m"] + (1.0 - a) * pbx
                est["body_y_m"] = a * est["body_y_m"] + (1.0 - a) * pby
                est["body_z_m"] = a * est["body_z_m"] + (1.0 - a) * pbz
        if fid is not None:
            self._ema = (fid, est["body_x_m"], est["body_y_m"], est["body_z_m"])
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
        fid = estimate.get("frame_id")
        bx = estimate["body_x_m"]
        by = estimate["body_y_m"]
        bz = estimate["body_z_m"]
        if any(math.isnan(v) for v in (bx, by, bz)):
            self.last_velocity = None
            return None
        if self._prev is not None and fid is not None:
            prev_fid, pbx, pby, pbz = self._prev
            if prev_fid is not None and 0 < fid - prev_fid <= 3:
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
