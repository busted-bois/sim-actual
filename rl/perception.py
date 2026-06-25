"""Module 4 (vision) — YOLO-pose corners -> PnP -> gate pose in WORLD NED.

Glues the detector (Module 3, gatepose) to PnP (pnp.estimate_pose). For each
detected gate:

  1. keep only corners that beat the confidence threshold and aren't clipped to
     the image edge (gatepose.visible_mask);
  2. if all 4 survive, solve PnP (IPPE_SQUARE) for the gate pose in the camera /
     body frame;
  3. lift to a WORLD-NED gate-center position using the drone's filtered pose:
     gate_world = drone_pos + R_wb @ gate_pos_body.

The world position feeds the gate tracker (Module 4c). The same PnP result also
yields a drone world-position measurement (gate_world - R_wb @ gate_pos_body)
the EKF can fuse when a gate's map position is known.

    uv run -m rl.perception --selftest      # detector-free PnP roundtrip
"""

from __future__ import annotations

import argparse

import numpy as np

from rl import pnp, spec
from rl.gatepose import GateDet, visible_mask

# IPPE_SQUARE needs all four coplanar corners; below this we skip the gate.
MIN_VISIBLE_CORNERS = 4
MAX_REPROJ_ERR_PX = 8.0  # drop wildly inconsistent solves
MIN_GATE_SPAN_PX = 30.0  # corners closer than this = tiny/far gate, PnP range
# is unreliable (small pixel size -> huge range error -> phantom world tracks).


def _corner_span_px(corners: np.ndarray) -> float:
    """Largest pairwise pixel distance between the 4 corners (gate size in px)."""
    return max(
        float(np.linalg.norm(corners[i] - corners[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    )


def gate_world_pose(
    det: GateDet, drone_pos: np.ndarray, drone_quat: np.ndarray
) -> dict | None:
    """Detected gate -> {gate_pos_world, gate_pos_body, range_m, reproj_err_px}."""
    vis = visible_mask(det)
    if int(vis.sum()) < MIN_VISIBLE_CORNERS:
        return None
    corners = np.asarray(det.corners, float)
    if _corner_span_px(corners) < MIN_GATE_SPAN_PX:
        return None  # too small/far to trust the PnP range
    est = pnp.estimate_pose(corners)
    if est is None or est["reproj_err_px"] > MAX_REPROJ_ERR_PX:
        return None
    R_wb = spec.quat_to_R(np.asarray(drone_quat, float))
    gate_pos_world = np.asarray(drone_pos, float) + R_wb @ est["gate_pos_body"]
    return {
        "gate_pos_world": gate_pos_world,
        "gate_pos_body": est["gate_pos_body"],
        "range_m": est["range_m"],
        "reproj_err_px": est["reproj_err_px"],
        "det_conf": det.det_conf,
    }


def drone_pos_from_gate(
    gate_pos_body: np.ndarray, gate_world_pos: np.ndarray, drone_quat: np.ndarray
) -> np.ndarray:
    """Vision world-position measurement of the drone, given a known gate pose."""
    R_wb = spec.quat_to_R(np.asarray(drone_quat, float))
    return np.asarray(gate_world_pos, float) - R_wb @ np.asarray(gate_pos_body, float)


class GatePerception:
    """Owns the YOLO-pose model; turns a frame + drone pose into world gates."""

    def __init__(self, weights: str | None = None):
        from rl.gatepose import WEIGHTS_PATH, GatePoseInfer

        self.infer = GatePoseInfer(weights or WEIGHTS_PATH)

    def process(
        self, img_bgr: np.ndarray, drone_pos: np.ndarray, drone_quat: np.ndarray
    ) -> list[dict]:
        out = []
        for det in self.infer.detect(img_bgr):
            g = gate_world_pose(det, drone_pos, drone_quat)
            if g is not None:
                out.append(g)
        return out


# ----------------------------------------------------------------------------
# Self-test: synthesize a perfect detection by projecting a known gate, then
# verify PnP recovers its world position. Exercises the keypoint->PnP path that
# pnp.py's own (mask+order_quad) self-test doesn't cover.
# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(2)
    world_err, drone_err = [], []
    tilt = np.tan(np.radians(spec.CAM_TILT_DEG))
    n_seen = 0
    for _ in range(200):
        dpos = rng.uniform(-2, 2, 3)
        yaw = rng.uniform(-0.25, 0.25)
        dq = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        fwd = rng.uniform(5, 8)
        lat = rng.uniform(-1.0, 1.0)
        # Center the gate in the 20°-up-tilted view so all 4 corners frame.
        elev = -fwd * tilt + rng.uniform(-0.3, 0.3)
        gpos = dpos + np.array([fwd, lat, elev])
        gq = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # 90°-about-Z: faces drone
        corners_w = spec.gate_corners_world(gpos, gq)
        px, in_front = spec.project(corners_w, dpos, dq)
        # Require all corners on-screen (the detector only sees framed gates).
        on = in_front.all() and (px[:, 0] >= 0).all() and (px[:, 0] < spec.IMG_W).all()
        on = on and (px[:, 1] >= 0).all() and (px[:, 1] < spec.IMG_H).all()
        if not on:
            continue
        n_seen += 1
        # corners are already in TL,TR,BR,BL order (spec.GATE_CORNERS_LOCAL order)
        det = GateDet(
            corners=px,
            kpt_conf=np.ones(4),
            det_conf=1.0,
            bbox=np.array(
                [px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()]
            ),
        )
        g = gate_world_pose(det, dpos, dq)
        assert g is not None, "perfect head-on gate should solve"
        world_err.append(np.linalg.norm(g["gate_pos_world"] - gpos))
        dmeas = drone_pos_from_gate(g["gate_pos_body"], gpos, dq)
        drone_err.append(np.linalg.norm(dmeas - dpos))
    assert n_seen > 50, f"too few framed gates ({n_seen}) — fixture is degenerate"
    world_err, drone_err = np.array(world_err), np.array(drone_err)
    print(
        f"[selftest] n={len(world_err)} gate world-pos err "
        f"mean={world_err.mean():.3f}m max={world_err.max():.3f}m"
    )
    print(f"[selftest] drone world-pos err mean={drone_err.mean():.3f}m")
    assert world_err.mean() < 0.05, "gate world position should be near-exact"
    assert drone_err.mean() < 0.05, "drone world position should be near-exact"

    # A tiny/far gate (corner span < MIN_GATE_SPAN_PX) is rejected.
    tiny = GateDet(
        corners=np.array([[320, 180], [340, 180], [340, 200], [320, 200]], float),
        kpt_conf=np.ones(4),
        det_conf=1.0,
        bbox=np.array([320, 180, 340, 200], float),
    )
    assert gate_world_pose(tiny, np.zeros(3), np.array([1.0, 0, 0, 0])) is None, (
        "tiny/far gate should be rejected by the span gate"
    )
    print(
        "[selftest] OK — YOLO keypoints (TL,TR,BR,BL) -> PnP -> world pose; "
        "span-gate rejects tiny far gates"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
