"""PnP gate pose from YOLO corner keypoints -> gate position in BODY frame.

This is the path off hardcoded gate coordinates: instead of reading a gate map,
solve the gate's pose relative to the drone from the detected corners. No world
position, no gate map, no GATE_INFO needed.

Ported from the reference pose_estimator.py (proven with simulator/models/
gate_pose.pt). Self-contained: owns its intrinsics, gate geometry and the
camera->body rotation so simulator/ doesn't depend on rl/.

estimate_gate_pose(keypoints, confs) -> dict | None:
    gate_pos_cam   (3,)  gate centre in camera optical frame (m)
    gate_pos_body  (3,)  gate centre in body FRD frame (x fwd, y right, z down)
    range_m        float straight-line distance to gate centre
    yaw_bearing    float angle to gate in body XY plane (rad, +right)
    pitch_bearing  float elevation angle to gate (rad, +up)
    reproj_px      float RMS reprojection error (detection quality)
    method         str

Keypoints are the 8 YOLO slots in native order (TL,TR,BL,BR inner, then outer).
"""

import cv2
import numpy as np

# --- intrinsics (spec 3.8: fx=fy=320, cx=320, cy=180 @ 640x360) --------------
_K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]])
_DIST = np.zeros(4)
_IMG_W, _IMG_H = 640, 360

# --- gate corners in gate-local frame (m, z=0 plane) -------------------------
# Order MUST match YOLO slots. LEFT/RIGHT SWAP baked in: the training-data
# generator wrote left labels onto right corners, so YOLO's "TL" slot actually
# sits at the gate's top-RIGHT. Without this swap PnP solves a mirrored
# correspondence -> gate back-face toward camera. Inner 1.5m, outer 2.7m.
_hi, _ho = 1.5 / 2, 2.7 / 2
_GATE_PTS_3D = np.array(
    [
        [_hi, _hi, 0],
        [-_hi, _hi, 0],
        [_hi, -_hi, 0],
        [-_hi, -_hi, 0],  # inner
        [_ho, _ho, 0],
        [-_ho, _ho, 0],
        [_ho, -_ho, 0],
        [-_ho, -_ho, 0],  # outer
    ],
    dtype=np.float32,
)

KEYPOINT_CONF_THRESHOLD = 0.7
EDGE_MARGIN_PX = 1  # corners pinned to the image border are clip artifacts

# --- camera optical -> body FRD, with 20deg up-tilt (spec 3.8) ---------------
CAM_TILT_DEG = 20.0
_R_BODY_CAM_BASE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _R_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


R_BODY_CAM = _R_y(np.radians(CAM_TILT_DEG)) @ _R_BODY_CAM_BASE


def _on_edge(kp):
    x, y = kp[:, 0], kp[:, 1]
    return (
        (x < EDGE_MARGIN_PX)
        | (x > _IMG_W - 1 - EDGE_MARGIN_PX)
        | (y < EDGE_MARGIN_PX)
        | (y > _IMG_H - 1 - EDGE_MARGIN_PX)
    )


def _visible_mask(kp, confs):
    return (confs > KEYPOINT_CONF_THRESHOLD) & ~_on_edge(kp)


def _reproj_rms(obj, img, rvec, tvec):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, _K, _DIST)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def estimate_pose(keypoints, confs):
    """Planar PnP via IPPE (two-fold ambiguous -> pick lower reproj error),
    then one LM polish. Needs >=4 confident, non-edge corners. -> pose dict."""
    kp = np.asarray(keypoints, np.float64)
    confs = np.asarray(confs, np.float64)
    vis = _visible_mask(kp, confs)
    if vis.sum() < 4:
        return None

    obj, img = _GATE_PTS_3D[vis], kp[vis]
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, img, _K, _DIST, flags=cv2.SOLVEPNP_IPPE
    )
    if n == 0:
        return None
    best = min(range(n), key=lambda i: _reproj_rms(obj, img, rvecs[i], tvecs[i]))
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, _K, _DIST, rvecs[best], tvecs[best])
    return {
        "rvec": rvec,
        "tvec": tvec,
        "reproj_px": _reproj_rms(obj, img, rvec, tvec),
        "n_visible": int(vis.sum()),
        "method": "ippe-mincost",
    }


def estimate_gate_pose(keypoints, confs, box=None):
    """Full result: PnP + camera->body + bearings. -> dict | None.

    Depth (z, metric scale) comes from PnP against the known gate size, but the
    LATERAL and VERTICAL position are reconstructed from the gate-centre pixel +
    that depth via the pinhole model. The pixel centre is exactly where we SEE
    the gate, so its left/right + up/down sign is correct by construction --
    independent of any keypoint-label (L/R) ambiguity in the PnP correspondence,
    which otherwise mirrors the gate to the wrong side.

    Centre pixel = centroid of the 4 INNER keypoints (the opening corners) when
    >=2 are confident. That targets the HOLE we must fly through -- immune to the
    AI-GP box on TOP of the gate, which biases the bounding-box / all-keypoint
    centre upward and made the drone clip the top or duck under the opening.
    Falls back to the bbox centre, then any confident keypoints."""
    pose = estimate_pose(keypoints, confs)
    if pose is None:
        return None
    depth = float(pose["tvec"].reshape(3)[2])
    if depth <= 0:  # gate must be in front of the camera
        return None
    kp = np.asarray(keypoints, np.float64)
    cf = np.asarray(confs, np.float64)
    inner = kp[:4]
    inner_vis = cf[:4] > KEYPOINT_CONF_THRESHOLD
    if inner_vis.sum() >= 2:
        ctr = inner[inner_vis].mean(axis=0)  # opening centre (the hole)
    elif box is not None:
        b = np.asarray(box, np.float64).reshape(-1)
        ctr = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    else:
        vis = _visible_mask(kp, cf)
        ctr = kp[vis].mean(axis=0) if vis.any() else kp.mean(axis=0)
    x_cam = (ctr[0] - _K[0, 2]) / _K[0, 0] * depth
    y_cam = (ctr[1] - _K[1, 2]) / _K[1, 1] * depth
    gate_pos_cam = np.array([x_cam, y_cam, depth])
    gate_pos_body = R_BODY_CAM @ gate_pos_cam
    horiz = float(np.hypot(gate_pos_body[0], gate_pos_body[1]))
    return {
        "gate_pos_cam": gate_pos_cam,
        "gate_pos_body": gate_pos_body,
        "range_m": float(np.linalg.norm(gate_pos_cam)),
        "yaw_bearing": float(np.arctan2(gate_pos_body[1], gate_pos_body[0])),
        "pitch_bearing": float(np.arctan2(-gate_pos_body[2], horiz)),
        "reproj_px": pose["reproj_px"],
        "n_visible": pose["n_visible"],
        "method": pose["method"],
    }


def _selftest():
    # Project the gate from a known pose, recover it, check error.
    rng = np.random.default_rng(0)
    errs, yaws = [], []
    for _ in range(200):
        # gate ahead in camera frame: z (fwd) 3-9m, small x/y offset
        t_true = np.array(
            [rng.uniform(-1, 1), rng.uniform(-0.6, 0.6), rng.uniform(3, 9)]
        )
        r_true = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0.0])
        proj, _ = cv2.projectPoints(_GATE_PTS_3D, r_true, t_true, _K, _DIST)
        kp = proj.reshape(-1, 2)
        # only count corners actually in-frame as confident
        inb = (
            (kp[:, 0] > 1)
            & (kp[:, 0] < _IMG_W - 1)
            & (kp[:, 1] > 1)
            & (kp[:, 1] < _IMG_H - 1)
        )
        if inb.sum() < 4:
            continue
        confs = np.where(inb, 1.0, 0.0)
        est = estimate_gate_pose(kp, confs)
        assert est is not None
        errs.append(np.linalg.norm(est["gate_pos_cam"] - t_true))
        # body forward should dominate (gate ahead), yaw bearing small
        yaws.append(abs(est["yaw_bearing"]))
    errs = np.array(errs)
    print(
        f"[selftest] n={len(errs)} pos_err mean={errs.mean():.3f}m max={errs.max():.3f}m"
    )
    assert errs.mean() < 0.1, "PnP should recover camera-frame gate position"
    print("[selftest] OK -- PnP recovers gate pose in body frame")


if __name__ == "__main__":
    _selftest()
