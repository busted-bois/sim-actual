"""AndurilGP HSV-red gate detection (ported from vendor/AndurilGP/vision_rx.py).

Used by make control-flight / GPPilot as the preferred perception source.
Keeps detect_gate as a pure function so offline replay matches live flight.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

IMG_W, IMG_H = 640, 360
CX, CY = 320.0, 180.0
FX = FY = 320.0
CAM_TILT_DEG = 20.0
GATE_WIDTH_M = 2.7

_GATE_HALF = GATE_WIDTH_M / 2.0
_GATE_OBJ_PTS = np.array(
    [
        [-_GATE_HALF, _GATE_HALF, 0.0],
        [_GATE_HALF, _GATE_HALF, 0.0],
        [_GATE_HALF, -_GATE_HALF, 0.0],
        [-_GATE_HALF, -_GATE_HALF, 0.0],
    ],
    dtype=np.float32,
)
_GATE_INNER_HALF = 0.75
_GATE_INNER_OBJ = np.array(
    [
        [-_GATE_INNER_HALF, _GATE_INNER_HALF, 0.0],
        [_GATE_INNER_HALF, _GATE_INNER_HALF, 0.0],
        [_GATE_INNER_HALF, -_GATE_INNER_HALF, 0.0],
        [-_GATE_INNER_HALF, -_GATE_INNER_HALF, 0.0],
    ],
    dtype=np.float32,
)
_GATE_OBJ_8 = np.vstack([_GATE_OBJ_PTS, _GATE_INNER_OBJ])
_CAM_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]], dtype=np.float32)
_CAM_DIST = np.zeros((4, 1), dtype=np.float32)
_HAS_IPPE = hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
PNP_MAX_REPROJ_PX = 12.0

LOWER_RED_1 = np.array([0, 120, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 50])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 300
SERVO_AREA_FLOOR = 300
EDGE_MARGIN_PX = 4
CENTER_REJECT_FRAC = 0.25

EMA_ALPHA = 0.5
EMA_RESET_DCX = 120
EMA_RESET_WR = 1.6

_PASS_AREA_PX = 80_000
_PASS_COOLDOWN_FR = 10
PNP_EMA_ALPHA = 0.4


def _order_quad(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ],
        dtype=np.float32,
    )


def _approx_quad(contour, epsilons=(0.03, 0.05, 0.08, 0.12)):
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in epsilons:
        cand = cv2.approxPolyDP(hull, eps * peri, True)
        if len(cand) == 4:
            return cand.reshape(4, 2).astype(np.float32)
    return None


def _extract_gate_corners(img, contour, mask):
    gx, gy, gw, gh = cv2.boundingRect(contour)
    if gx <= 1 or gy <= 1 or (gx + gw) >= IMG_W - 2 or (gy + gh) >= IMG_H - 2:
        return None, None

    outer_pts = _approx_quad(contour)
    if outer_pts is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    try:
        cv2.cornerSubPix(gray, outer_pts, (7, 7), (-1, -1), crit)
    except Exception:
        pass
    outer_4 = _order_quad(outer_pts)

    inner_4 = None
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_fine = cv2.bitwise_or(
            cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
            cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2),
        )
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_fine = cv2.morphologyEx(mask_fine, cv2.MORPH_CLOSE, k3)
        mask_fine = cv2.morphologyEx(mask_fine, cv2.MORPH_OPEN, k3)
        all_cnts, hier = cv2.findContours(
            mask_fine, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hier is not None:
            hier = hier[0]
            gate_area = cv2.contourArea(contour)
            for i, h in enumerate(hier):
                if h[3] == -1:
                    continue
                ix, iy, iw, ih = cv2.boundingRect(all_cnts[i])
                if ix < gx or iy < gy or ix + iw > gx + gw or iy + ih > gy + gh:
                    continue
                iarea = cv2.contourArea(all_cnts[i])
                if not (0.10 * gate_area < iarea < 0.70 * gate_area):
                    continue
                inner_pts = _approx_quad(all_cnts[i])
                if inner_pts is None:
                    continue
                try:
                    cv2.cornerSubPix(gray, inner_pts, (5, 5), (-1, -1), crit)
                except Exception:
                    pass
                inner_4 = _order_quad(inner_pts)
                break
    except Exception:
        pass
    return outer_4, inner_4


def _solve_gate_pnp(corners):
    try:
        if _HAS_IPPE:
            n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                _GATE_OBJ_PTS,
                corners,
                _CAM_K,
                _CAM_DIST,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            best_i, best_err = None, float("inf")
            for i in range(n):
                depth = float(tvecs[i][2, 0])
                err = float(errors[i].flat[0])
                if depth > 0.3 and err < best_err:
                    best_err, best_i = err, i
            if best_i is None:
                return None, None, float("inf")
            return tvecs[best_i].flatten(), rvecs[best_i].flatten(), best_err
        ok, rvec, tvec = cv2.solvePnP(
            _GATE_OBJ_PTS, corners, _CAM_K, _CAM_DIST, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok or float(tvec[2]) <= 0.3:
            return None, None, float("inf")
        proj, _ = cv2.projectPoints(_GATE_OBJ_PTS, rvec, tvec, _CAM_K, _CAM_DIST)
        err = float(
            np.mean(
                np.linalg.norm(proj.reshape(4, 2) - corners.reshape(4, 2), axis=1)
            )
        )
        return tvec.flatten(), rvec.flatten(), err
    except Exception:
        return None, None, float("inf")


def _refine_pnp_8pt(outer_corners, inner_corners, init_rvec, init_tvec):
    try:
        all_img = np.vstack([outer_corners, inner_corners])
        ok, rvec_r, tvec_r = cv2.solvePnP(
            _GATE_OBJ_8,
            all_img,
            _CAM_K,
            _CAM_DIST,
            rvec=init_rvec,
            tvec=init_tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or float(tvec_r[2, 0]) <= 0.3:
            return None, None, float("inf")
        proj, _ = cv2.projectPoints(_GATE_OBJ_8, rvec_r, tvec_r, _CAM_K, _CAM_DIST)
        err = float(
            np.mean(np.linalg.norm(proj.reshape(-1, 2) - all_img.reshape(-1, 2), axis=1))
        )
        return tvec_r.flatten(), rvec_r.flatten(), err
    except Exception:
        return None, None, float("inf")


def detect_gate(img):
    """HSV red contour + optional PnP. Returns (estimate|None, mask, contours)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
        cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= SERVO_AREA_FLOOR]
    if not valid:
        return None, mask, contours

    best = max(valid, key=cv2.contourArea)
    moments = cv2.moments(best)
    if moments["m00"] == 0.0:
        return None, mask, contours

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    bx, by, bw, bh = cv2.boundingRect(best)
    area = cv2.contourArea(best)

    true_cx = bx + bw / 2.0
    true_cy = by + bh / 2.0
    touches_lr = (bx <= EDGE_MARGIN_PX) or (bx + bw >= IMG_W - EDGE_MARGIN_PX)
    touches_tb = (by <= EDGE_MARGIN_PX) or (by + bh >= IMG_H - EDGE_MARGIN_PX)
    off_centre_x = abs(true_cx - CX) > CENTER_REJECT_FRAC * IMG_W
    off_centre_y = abs(true_cy - CY) > CENTER_REJECT_FRAC * IMG_H
    if touches_lr and off_centre_x:
        return None, mask, contours

    reliable = (area >= MIN_CONTOUR_AREA) and not (
        touches_tb and off_centre_y and area < 5000
    )
    estimate = {
        "cx": cx,
        "cy": cy,
        "cx_offset": cx - CX,
        "cy_offset": cy - CY,
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_w": bw,
        "bbox_h": bh,
        "area": area,
        "bx": bx,
        "by": by,
        "bw": bw,
        "bh": bh,
        "reliable": reliable,
        "pnp_ok": False,
    }

    if reliable:
        outer_corners, inner_corners = _extract_gate_corners(img, best, mask)
        if outer_corners is not None:
            tvec, rvec, reproj = _solve_gate_pnp(outer_corners)
            if tvec is not None and reproj < PNP_MAX_REPROJ_PX:
                if inner_corners is not None:
                    tvec8, rvec8, reproj8 = _refine_pnp_8pt(
                        outer_corners,
                        inner_corners,
                        rvec.reshape(3, 1),
                        tvec.reshape(3, 1),
                    )
                    if tvec8 is not None and reproj8 < PNP_MAX_REPROJ_PX:
                        tvec, rvec, reproj = tvec8, rvec8, reproj8
                estimate["pnp_ok"] = True
                estimate["pnp_tvec"] = tvec
                estimate["pnp_rvec"] = rvec
                estimate["pnp_reproj"] = reproj

    return estimate, mask, contours


def ema_smooth(prev_bbox, estimate, alpha=EMA_ALPHA):
    if estimate is None:
        return None, None
    cur = (estimate["bx"], estimate["by"], estimate["bw"], estimate["bh"])
    if prev_bbox is None:
        sm = cur
    else:
        ccx = cur[0] + cur[2] / 2.0
        pcx = prev_bbox[0] + prev_bbox[2] / 2.0
        wr = (cur[2] / prev_bbox[2]) if prev_bbox[2] else 99.0
        if (
            abs(ccx - pcx) > EMA_RESET_DCX
            or wr > EMA_RESET_WR
            or wr < 1.0 / EMA_RESET_WR
        ):
            sm = cur
        else:
            sm = tuple(alpha * c + (1.0 - alpha) * p for c, p in zip(cur, prev_bbox))
    bx, by, bw, bh = sm
    out = dict(estimate)
    out["bx"], out["by"], out["bw"], out["bh"] = bx, by, bw, bh
    out["cx"] = bx + bw / 2.0
    out["cy"] = by + bh / 2.0
    out["cx_offset"] = out["cx"] - CX
    out["cy_offset"] = out["cy"] - CY
    return out, sm


def body_relative_pose(estimate):
    if estimate is None:
        return estimate
    if estimate.get("pnp_ok"):
        tvec = estimate["pnp_tvec"]
        cam_x_m = float(tvec[0])
        cam_y_m = float(tvec[1])
        cam_z_m = float(tvec[2])
    else:
        bw = estimate.get("bw", 0)
        cx = estimate.get("cx", CX)
        cy = estimate.get("cy", CY)
        if not bw:
            nan = float("nan")
            estimate.update(
                cam_x_m=nan,
                cam_y_m=nan,
                cam_z_m=nan,
                body_x_m=nan,
                body_y_m=nan,
                body_z_m=nan,
            )
            return estimate
        cam_z_m = (FX * GATE_WIDTH_M) / bw
        cam_x_m = (cx - CX) * cam_z_m / FX
        cam_y_m = (cy - CY) * cam_z_m / FY

    t = math.radians(CAM_TILT_DEG)
    ct, st = math.cos(t), math.sin(t)
    # Camera pitched DOWN by CAM_TILT vs body FRD: optical-axis gate → +body_z.
    body_x_m = ct * cam_z_m - st * cam_y_m
    body_y_m = cam_x_m
    body_z_m = st * cam_z_m + ct * cam_y_m
    estimate.update(
        cam_x_m=cam_x_m,
        cam_y_m=cam_y_m,
        cam_z_m=cam_z_m,
        body_x_m=body_x_m,
        body_y_m=body_y_m,
        body_z_m=body_z_m,
    )
    return estimate


def _normal_body_from_rvec(rvec) -> np.ndarray | None:
    if rvec is None:
        return None
    try:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        n_cam = R[:, 2].copy()
        # Point normal back toward camera (same sign-fix idea as gate_pnp).
        t = math.radians(CAM_TILT_DEG)
        # Map camera → body for the normal (same tilt as position).
        ct, st = math.cos(t), math.sin(t)
        # cam (x right, y down, z fwd) → body FRD via same down-tilt on (z,y):
        # body uses cam_x as Y; rotate (cam_z, cam_y) into (bx,bz).
        nx, ny, nz = float(n_cam[0]), float(n_cam[1]), float(n_cam[2])
        bx = ct * nz - st * ny
        by = nx
        bz = st * nz + ct * ny
        n = np.array([bx, by, bz], dtype=np.float64)
        # Prefer normal pointing back at drone (dot with +X body > 0 means
        # pointing forward — flip so fly-through is -normal).
        if n[0] > 0:
            n = -n
        n /= max(np.linalg.norm(n), 1e-9)
        return n
    except Exception:
        return None


def to_gp_estimate(raw: dict | None, frame_id: int | None) -> dict | None:
    """Convert Anduril detect+pose dict → GPPilot vision_gate_estimate shape."""
    if raw is None:
        return None
    bx = float(raw.get("body_x_m", float("nan")))
    by = float(raw.get("body_y_m", float("nan")))
    bz = float(raw.get("body_z_m", float("nan")))
    if any(math.isnan(v) for v in (bx, by, bz)):
        return None
    nb = None
    if raw.get("pnp_ok"):
        nb = _normal_body_from_rvec(raw.get("pnp_rvec"))
    return {
        "frame_id": frame_id,
        "body_x_m": bx,
        "body_y_m": by,
        "body_z_m": bz,
        "pnp_ok": bool(raw.get("pnp_ok")),
        "pnp_rvec": raw.get("pnp_rvec"),
        "normal_body": nb,
        "u_px": float(raw.get("cx", CX)),
        "v_px": float(raw.get("cy", CY)),
        "reliable": bool(raw.get("reliable", False)),
        "source": "anduril",
        "area": float(raw.get("area", 0.0)),
    }


class AndurilGateTracker:
    """Per-frame Anduril detection with EMA + pass-through suppress → shared_data."""

    def __init__(self):
        self._ema_prev = None
        self._pnp_ema = None
        self._in_gate = False
        self._pass_cooldown = 0

    def reset(self) -> None:
        self._ema_prev = None
        self._pnp_ema = None
        self._in_gate = False
        self._pass_cooldown = 0

    def process(self, frame_id: int, img) -> dict | None:
        estimate, _mask, _contours = detect_gate(img)
        raw_area = estimate["area"] if estimate is not None else 0

        if raw_area > _PASS_AREA_PX:
            self._in_gate = True
            self._ema_prev = None
            self._pnp_ema = None
            return None

        if self._in_gate:
            self._in_gate = False
            self._pass_cooldown = _PASS_COOLDOWN_FR
            self._ema_prev = None
            self._pnp_ema = None

        if self._pass_cooldown > 0:
            self._pass_cooldown -= 1
            return None

        estimate, self._ema_prev = ema_smooth(self._ema_prev, estimate)

        if estimate is not None and estimate.get("pnp_ok"):
            new_tvec = np.asarray(estimate["pnp_tvec"], dtype=np.float64).copy()
            if self._pnp_ema is not None:
                smoothed = PNP_EMA_ALPHA * new_tvec + (1.0 - PNP_EMA_ALPHA) * self._pnp_ema
                estimate["pnp_tvec"] = smoothed
            self._pnp_ema = np.asarray(estimate["pnp_tvec"], dtype=np.float64).copy()
        else:
            self._pnp_ema = None

        body_relative_pose(estimate)
        return to_gp_estimate(estimate, frame_id)
