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
    normal_body    (3,)  gate-plane unit normal in body FRD, pointing back at
                         the drone (fly-through direction is -normal_body)
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

# Inner-opening corners ordered [TL, TR, BR, BL] as they appear in the IMAGE
# for a front-facing gate (camera +x right, +y down): used by the CV-refined
# 4-corner path, no YOLO slot/label ambiguity involved.
_INNER_CORNERS_3D = np.array(
    [[-_hi, -_hi, 0], [_hi, -_hi, 0], [_hi, _hi, 0], [-_hi, _hi, 0]],
    dtype=np.float32,
)

KEYPOINT_CONF_THRESHOLD = 0.7
EDGE_MARGIN_PX = 1  # corners pinned to the image border are clip artifacts
MAX_REPROJ_PX = 10.0  # RMS reprojection above this = bad correspondence/depth

# Inner-opening edges as keypoint-slot pairs: (top, bottom, left, right) in
# slot space. The opening centre sits _hi metres from each edge's midpoint,
# perpendicular to it, toward the inside of the gate.
_INNER_EDGE_PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3))
_PAIR_MIN_PX = 30.0  # shorter edge = too far / degenerate for 2-point depth

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


def _solve_ippe(obj, img):
    """IPPE (two-fold ambiguous -> pick lower reproj error) + one LM polish.
    Rejects solutions with RMS reprojection above MAX_REPROJ_PX.
    -> (rvec, tvec, reproj_px) | None."""
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, img, _K, _DIST, flags=cv2.SOLVEPNP_IPPE
    )
    if n == 0:
        return None
    best = min(range(n), key=lambda i: _reproj_rms(obj, img, rvecs[i], tvecs[i]))
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, _K, _DIST, rvecs[best], tvecs[best])
    reproj = _reproj_rms(obj, img, rvec, tvec)
    if reproj > MAX_REPROJ_PX:
        return None
    return rvec, tvec, reproj


def estimate_pose(keypoints, confs):
    """Planar PnP on the YOLO keypoints. Needs >=4 confident, non-edge
    corners. -> pose dict | None."""
    kp = np.asarray(keypoints, np.float64)
    confs = np.asarray(confs, np.float64)
    vis = _visible_mask(kp, confs)
    if vis.sum() < 4:
        return None

    solved = _solve_ippe(_GATE_PTS_3D[vis], kp[vis])
    if solved is None:
        return None
    rvec, tvec, reproj = solved
    return {
        "rvec": rvec,
        "tvec": tvec,
        "reproj_px": reproj,
        "n_visible": int(vis.sum()),
        "method": "ippe-yolo8",
    }


def _diag_intersection(p4):
    """Opening centre from all 4 inner corners as the intersection of the
    diagonals TL-BR and TR-BL (slots 0-3 / 1-2 — the L/R label swap maps the
    diagonals onto each other, so the pairing is swap-proof). Falls back to
    the centroid only for a degenerate (near-parallel) quad."""
    a1, a2 = p4[0], p4[3]
    b1, b2 = p4[1], p4[2]
    d1 = a2 - a1
    d2 = b2 - b1
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-9:
        return p4.mean(axis=0)
    t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / denom
    return a1 + t * d1


def _edge_pair_centre(kp, vis, box, pose=None):
    """Opening-centre pixel from ONE fully visible inner edge.

    On the final approach the camera crops corners until the last thing still
    in frame is a single edge of the opening (usually the top bar, right under
    the gate's "AI GP" banner). The centre sits exactly _hi metres from that
    edge's midpoint, perpendicular to it, toward the inside of the gate —
    project that offset with the pinhole model.

    The inward sign is decided most-reliable-first:
      1. same-edge OUTER corners (0.6 m outward = exactly opposite the centre),
      2. any OTHER visible keypoint (opposite/side corners all lie across the
         edge on the centre side — raw-pixel geometric truth),
      3. the PnP-projected centre (direction only — its magnitude is what a
         one-edge trapezoid solve gets wrong),
      4. the YOLO box centre, only when it extends past the projected bar
         thickness (a cropped gate's box hugs the visible bar, whose centre
         sits ~0.2*length off the edge on the WRONG side — trusting it aimed
         the drone 1.45 m below a gate at 3 m, log/geometry-verified),
      5. slot identity: YOLO labels slots 0,1 as the image-TOP pair, so the
         centre is image-down (2,3: image-up). Side edges are skipped here.

    Depth comes from the PnP solve when available, else from the edge's pixel
    length against the known 1.5 m opening. -> (ctr_px, depth_m) | None.
    """
    for a, b in _INNER_EDGE_PAIRS:
        if not (vis[a] and vis[b]):
            continue
        pa, pb = kp[a], kp[b]
        edge = pb - pa
        length = float(np.hypot(edge[0], edge[1]))
        if length < _PAIR_MIN_PX:
            continue
        d = float(pose["tvec"].reshape(3)[2]) if pose is not None else 0.0
        if d <= 0.0:
            d = (2.0 * _hi) * _K[0, 0] / length
        perp = np.array([-edge[1], edge[0]]) / length
        mid = (pa + pb) / 2.0

        sign = 0.0
        outer = [kp[i + 4] for i in (a, b) if vis[i + 4]]
        if outer:
            dot = float(np.dot(perp, np.mean(outer, axis=0) - mid))
            if abs(dot) > 0.1 * length:
                sign = -1.0 if dot > 0.0 else 1.0  # away from the outer ring
        if sign == 0.0:
            skip = {a, b, a + 4, b + 4}
            others = [kp[i] for i in range(8) if vis[i] and i not in skip]
            if others:
                dot = float(np.dot(perp, np.mean(others, axis=0) - mid))
                if abs(dot) > 0.1 * length:
                    sign = 1.0 if dot > 0.0 else -1.0  # rest of gate = centre side
        if sign == 0.0 and pose is not None:
            proj, _ = cv2.projectPoints(
                np.zeros((1, 3), np.float32), pose["rvec"], pose["tvec"], _K, _DIST
            )
            dot = float(np.dot(perp, proj.reshape(2) - mid))
            if abs(dot) > 0.1 * length:
                sign = 1.0 if dot > 0.0 else -1.0
        if sign == 0.0 and box is not None:
            b_ = np.asarray(box, np.float64).reshape(-1)
            box_ctr = np.array([(b_[0] + b_[2]) / 2.0, (b_[1] + b_[3]) / 2.0])
            dot = float(np.dot(perp, box_ctr - mid))
            if abs(dot) > 0.4 * length:
                sign = 1.0 if dot > 0.0 else -1.0
        if sign == 0.0 and abs(perp[1]) > 1e-6:
            if (a, b) == (0, 1):  # image-top edge: centre is below (+v)
                sign = 1.0 if perp[1] > 0.0 else -1.0
            elif (a, b) == (2, 3):  # image-bottom edge: centre is above
                sign = -1.0 if perp[1] > 0.0 else 1.0
        if sign == 0.0:
            continue

        ctr = mid + sign * perp * (_hi * _K[0, 0] / d)
        return ctr, float(d)
    return None


def estimate_gate_pose(keypoints, confs, box=None):
    """Full result: PnP + camera->body + bearings. -> dict | None.

    Depth (z, metric scale) comes from PnP against the known gate size, but the
    LATERAL and VERTICAL position are reconstructed from the gate-centre pixel +
    that depth via the pinhole model. The pixel centre is exactly where we SEE
    the gate, so its left/right + up/down sign is correct by construction --
    independent of any keypoint-label (L/R) ambiguity in the PnP correspondence,
    which otherwise mirrors the gate to the wrong side.

    Centre pixel = centroid of the visible INNER keypoints (the opening) — but
    ONLY when that set is vertically+horizontally balanced (all 4, or a
    diagonal pair). The 20-deg-up camera pushes the BOTTOM corners out of
    frame inside ~4.5 m on a centred approach, and the centroid of what's
    left is the TOP EDGE of the opening, 0.75 m above centre — the drone flew
    to it and clipped the top bar. For one-sided visibility the centre is
    re-anchored off a fully visible inner edge (_edge_pair_centre) at PnP
    depth, falling back to projecting the PnP-solved centre.

    When PnP itself fails (< 4 confident corners — routine inside ~4.5 m as
    the crop eats the gate), a single visible inner edge still yields a full
    pose via the pinhole model (method "edge-pair"). Without it the gate
    vanished at 3-5 m out and the pilot retargeted a FAR gate mid-approach —
    log-verified as the dominant gate-2+ clip: banked toward the new target
    straight into the frame it was about to thread."""
    kp = np.asarray(keypoints, np.float64)
    cf = np.asarray(confs, np.float64)
    vis = _visible_mask(kp, cf)
    pose = estimate_pose(keypoints, confs)
    if pose is None:
        if int(vis.sum()) >= 4:
            # Enough corners but the solve was REJECTED (bad correspondence /
            # reproj) — don't resurrect inconsistent keypoints from one edge.
            return None
        pair = _edge_pair_centre(kp, vis, box)
        if pair is None:
            return None
        ctr, depth = pair
        return _pose_from_centre_depth(ctr, depth)
    vis4 = vis[:4]  # slots TL,TR,BL,BR (top pair, bottom pair)
    n4 = int(vis4.sum())
    balanced = n4 == 4 or (
        n4 == 2 and ((vis4[0] and vis4[3]) or (vis4[1] and vis4[2]))
    )
    if balanced and n4 == 4:
        # Diagonal intersection, not centroid: projective-exact centre of the
        # opening under any perspective. The centroid is pulled 6-10% toward
        # the nearer (larger-projected) edge, which damps vertical convergence
        # exactly when the drone is off-centre — a persistent LOW aim while
        # riding low into a gate.
        ctr = _diag_intersection(kp[:4])
    elif balanced:
        ctr = kp[:4][vis4].mean(axis=0)  # diagonal pair: midpoint = centre
    else:
        pair = _edge_pair_centre(kp, vis, box, pose=pose)
        if pair is not None:
            ctr = pair[0]
        else:
            proj, _ = cv2.projectPoints(
                np.zeros((1, 3), np.float32), pose["rvec"], pose["tvec"], _K, _DIST
            )
            ctr = proj.reshape(2)
    return _finalize_pose(pose, ctr)


def estimate_gate_pose_from_corners(corners):
    """Full result from 4 CV-refined inner-opening corners [TL,TR,BR,BL]
    (image order, see simulator/gate_corners_cv.py). Same dict as
    estimate_gate_pose; method 'ippe-cv4'. -> dict | None."""
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    if not np.all(np.isfinite(corners)):
        return None
    solved = _solve_ippe(_INNER_CORNERS_3D, corners)
    if solved is None:
        return None
    rvec, tvec, reproj = solved
    pose = {
        "rvec": rvec,
        "tvec": tvec,
        "reproj_px": reproj,
        "n_visible": 4,
        "method": "ippe-cv4",
    }
    return _finalize_pose(pose, corners.mean(axis=0))


def _finalize_pose(pose, ctr):
    """PnP depth + centre pixel -> body-frame position, plane normal and
    bearings (shared by the YOLO-keypoint and CV-corner paths)."""
    depth = float(pose["tvec"].reshape(3)[2])
    if depth <= 0:  # gate must be in front of the camera
        return None
    # Gate-plane normal: the corners live on the gate-local z=0 plane, so the
    # rotated local +z is the plane normal. Sign-fix it to point back at the
    # camera; the fly-through axis the guidance needs is then -normal.
    R_cg, _ = cv2.Rodrigues(pose["rvec"])
    n_cam = R_cg[:, 2]
    return _result_dict(
        ctr, depth, n_cam, pose["reproj_px"], pose["n_visible"], pose["method"]
    )


def _pose_from_centre_depth(ctr, depth):
    """Edge-pair fallback (no PnP solve): centre pixel + pinhole depth -> the
    same result dict. No solvable plane orientation from 2 points, so the
    normal is the straight-facing assumption — near-gate the guidance blends
    toward bearing anyway, and a zero tilt is the do-no-harm yaw input."""
    if depth <= 0:
        return None
    return _result_dict(ctr, depth, np.array([0.0, 0.0, -1.0]), 0.0, 2, "edge-pair")


def _result_dict(ctr, depth, n_cam, reproj_px, n_visible, method):
    x_cam = (ctr[0] - _K[0, 2]) / _K[0, 0] * depth
    y_cam = (ctr[1] - _K[1, 2]) / _K[1, 1] * depth
    gate_pos_cam = np.array([x_cam, y_cam, depth])
    gate_pos_body = R_BODY_CAM @ gate_pos_cam
    if float(np.dot(n_cam, gate_pos_cam)) > 0:
        n_cam = -n_cam
    normal_body = R_BODY_CAM @ n_cam
    horiz = float(np.hypot(gate_pos_body[0], gate_pos_body[1]))
    return {
        "gate_pos_cam": gate_pos_cam,
        "gate_pos_body": gate_pos_body,
        "normal_body": normal_body,
        "range_m": float(np.linalg.norm(gate_pos_cam)),
        "yaw_bearing": float(np.arctan2(gate_pos_body[1], gate_pos_body[0])),
        "pitch_bearing": float(np.arctan2(-gate_pos_body[2], horiz)),
        "reproj_px": reproj_px,
        "n_visible": n_visible,
        "method": method,
    }


def _selftest():
    # Project the gate from a known pose, recover it, check error.
    rng = np.random.default_rng(0)
    errs, yaws, nrm_degs = [], [], []
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
        # normal: compare against the true plane normal, sign-fixed toward cam
        R_true, _ = cv2.Rodrigues(r_true)
        n_true = R_true[:, 2]
        if float(np.dot(n_true, t_true)) > 0:
            n_true = -n_true
        n_est = R_BODY_CAM.T @ est["normal_body"]  # back to camera frame
        cosang = float(np.clip(np.dot(n_est, n_true), -1.0, 1.0))
        nrm_degs.append(np.degrees(np.arccos(cosang)))
    errs, nrm_degs = np.array(errs), np.array(nrm_degs)
    print(
        f"[selftest] n={len(errs)} pos_err mean={errs.mean():.3f}m max={errs.max():.3f}m"
        f" normal_err mean={nrm_degs.mean():.1f}deg max={nrm_degs.max():.1f}deg"
    )
    assert errs.mean() < 0.1, "PnP should recover camera-frame gate position"
    assert nrm_degs.mean() < 5.0, "PnP should recover the gate-plane normal"
    print("[selftest] OK -- PnP recovers gate pose in body frame")


if __name__ == "__main__":
    _selftest()
