"""YOLO / PnP adapter → AndurilGP-style vision_gate_estimate fields.

Consumes our shared_data["pose"] (GatePoseRunner) and gate_target, produces
body-frame gate position, optional PnP normal tilt, and optical-flow velocity
via centroid finite differences.
"""

from __future__ import annotations

import math

import numpy as np

GATE_OUTER_W_M = 2.7
FX = 320.0
CX = 320.0
CY = 180.0
CAM_TILT_DEG = 20.0


def gate_body_from_pinhole(
    u_px: float, v_px: float, box_w_px: float, gate_w_m: float = GATE_OUTER_W_M
) -> tuple[float, float, float] | None:
    """Pinhole range from outer width + cam-tilt → body FRD (bx, by, bz)."""
    if box_w_px < 1.0:
        return None
    z_cam = (gate_w_m * FX) / box_w_px
    x_cam = (u_px - CX) / FX * z_cam
    y_cam = (v_px - CY) / FX * z_cam
    # Same as gate_pnp: R_BODY_CAM = Ry(20) @ [[0,0,1],[1,0,0],[0,1,0]]
    # maps cam(x,y,z) → body via base=(z,x,y) then Ry(tilt).
    c = math.cos(math.radians(CAM_TILT_DEG))
    s = math.sin(math.radians(CAM_TILT_DEG))
    bx = c * z_cam + s * y_cam
    by = x_cam
    bz = -s * z_cam + c * y_cam
    return float(bx), float(by), float(bz)


def best_pose_gate(data: dict) -> dict | None:
    """Best YOLO gate with a solved pose from data['pose']."""
    pose_pkt = data.get("pose") or {}
    gates = pose_pkt.get("gates") or []
    best = None
    best_conf = -1.0
    for g in gates:
        if not g.get("pose"):
            continue
        conf = float(g.get("conf", 0.0))
        if conf > best_conf:
            best_conf = conf
            best = g
    return best


def vision_gate_estimate(data: dict) -> dict | None:
    """Build Anduril-compatible vision estimate from YOLO/PnP or gate_target.

    Returns dict with: frame_id, body_x_m, body_y_m, body_z_m,
    pnp_ok, pnp_rvec (or None), normal_body (or None), u_px, v_px.
    """
    pose_pkt = data.get("pose") or {}
    frame_id = pose_pkt.get("frame_id")
    g = best_pose_gate(data)
    if g is not None:
        p = g["pose"]
        gb = np.asarray(p["gate_pos_body"], dtype=np.float64).reshape(3)
        return {
            "frame_id": frame_id,
            "body_x_m": float(gb[0]),
            "body_y_m": float(gb[1]),
            "body_z_m": float(gb[2]),
            "pnp_ok": True,
            "pnp_rvec": None,  # unused; normal_body preferred
            "normal_body": np.asarray(p["normal_body"], dtype=np.float64).reshape(3),
            "u_px": None,
            "v_px": None,
        }

    gt = data.get("gate_target") or {}
    if not gt.get("detected"):
        return None
    frame_id = gt.get("frame_id", frame_id)
    u = gt.get("u_px")
    v = gt.get("v_px")
    range_m = gt.get("range_m")
    bearing = gt.get("bearing_rad")
    elev = gt.get("elevation_rad")

    # Vision rays: bearing / elev in body → R * unit ray
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
        }

    if u is None or v is None:
        return None
    # Fallback: box width from r_frac if present
    r_frac = float(gt.get("r_frac") or 0.0)
    # area fraction → crude side length: sqrt(area) ~ side of square
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
    }


def gate_tilt_deg_from_normal(normal_body: np.ndarray) -> float:
    """Body-XY angle of gate normal (Anduril PnP tilt proxy)."""
    n = np.asarray(normal_body, dtype=np.float64).reshape(3)
    return float(np.clip(math.degrees(math.atan2(n[1], n[0])), -30.0, 30.0))


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
                # Gate body position change ≈ -drone body motion (gate fixed world).
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
