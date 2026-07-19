"""YOLO-pose gate detection (corner keypoints).

Wraps the bundled YOLO11n-pose model (simulator/models/gate_pose.pt, trained
on Unity gate renders -- 8 corner keypoints per gate: TL,TR,BL,BR inner then
outer). Used to detect gates visually instead of relying on the broadcast gate
map -- the path off hardcoded coordinates for VQ2 (GATE_INFO blocked).

Inference is SLOW on CPU (~100-350 ms/frame), so it must never run inline in
the UDP receiver (would drop packets). GatePoseRunner runs it on its own
thread, newest-frame-wins: it always grabs the latest decoded frame and skips
any that piled up while the previous inference was running.

detect(img) -> (gates, annotated):
    gates     list of {box, conf, keypoints (N,2), keypoint_conf (N,)}
    annotated BGR frame with boxes + keypoints drawn (result.plot())
"""

import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from simulator.gate_corners_cv import find_gate_inner_corners
from simulator.gate_pnp import estimate_gate_pose, estimate_gate_pose_from_corners

_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "gate_pose.pt")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_IMGSZ = 640

# CV corner refine on the best YOLO hit (default ON): contour-accurate inner
# corners fix long-range PnP depth, where raw keypoint error dominates.
_CV_REFINE = os.environ.get("GATE_CV_REFINE", "1").lower() not in ("0", "false", "no")
_CV_BOX_MARGIN = 0.2  # CV quad centre must sit inside the YOLO box +20%

_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(_WEIGHTS):
            raise FileNotFoundError(f"gate-pose weights missing: {_WEIGHTS}")
        _model = YOLO(_WEIGHTS)
        if _DEVICE == "cuda":
            _model.to("cuda")
        print(
            f"[gate_pose] YOLO loaded on {_DEVICE} ({os.path.basename(_WEIGHTS)})",
            flush=True,
        )
    return _model


def detect(img):
    """Run gate detection on a BGR frame. Returns (gates, annotated)."""
    res = _get_model().predict(source=img, verbose=False, device=_DEVICE, imgsz=_IMGSZ)[
        0
    ]

    gates = []
    kp = res.keypoints
    for i in range(len(res.boxes)):
        kxy = kp.xy[i].cpu().numpy() if kp is not None else None
        kconf = (
            kp.conf[i].cpu().numpy()
            if (kp is not None and kp.conf is not None)
            else None
        )
        box = res.boxes.xyxy[i].cpu().numpy()
        # Solve gate pose (body frame) here in the detector thread so the
        # 150 Hz control loop only reads the result, never runs PnP.
        pose = (
            estimate_gate_pose(kxy, kconf, box=box)
            if (kxy is not None and kconf is not None)
            else None
        )
        gates.append(
            {
                "box": box,
                "conf": float(res.boxes.conf[i]),
                "keypoints": kxy,
                "keypoint_conf": kconf,
                "pose": pose,
                "cv_corners": None,
            }
        )
    annotated = res.plot(line_width=2)
    if _CV_REFINE:
        if gates:
            _cv_refine_best(img, gates, annotated)
        else:
            _cv_only_gate(img, gates, annotated)
    return gates, annotated


def _attach_cv_corners(gate, corners, annotated):
    """Publish hole quad + green draw; PnP refine is best-effort only."""
    gate["cv_corners"] = corners
    pose = estimate_gate_pose_from_corners(corners)
    if pose is not None:
        gate["pose"] = pose
    cv2.polylines(annotated, [corners.astype(np.int32)], True, (0, 255, 0), 2)


def _cv_refine_best(img, gates, annotated):
    """Attach CV inner-opening corners to the best YOLO gate when the hole
    centroid sits inside that box. Corners publish even if PnP fails."""
    best = max(gates, key=lambda g: g["conf"])
    corners = find_gate_inner_corners(img)
    if corners is None:
        return
    ctr = corners.mean(axis=0)
    b = np.asarray(best["box"], np.float64).reshape(-1)
    mx = (b[2] - b[0]) * _CV_BOX_MARGIN
    my = (b[3] - b[1]) * _CV_BOX_MARGIN
    if not (b[0] - mx <= ctr[0] <= b[2] + mx and b[1] - my <= ctr[1] <= b[3] + my):
        return  # CV found a different gate than YOLO's best -- don't mix
    _attach_cv_corners(best, corners, annotated)


def _cv_only_gate(img, gates, annotated):
    """YOLO-free path: HSV hole alone is enough for IBVS/GP pixel aim."""
    corners = find_gate_inner_corners(img)
    if corners is None:
        return
    xs, ys = corners[:, 0], corners[:, 1]
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    gate = {
        "box": box,
        "conf": 1.0,
        "keypoints": None,
        "keypoint_conf": None,
        "pose": None,
        "cv_corners": None,
    }
    _attach_cv_corners(gate, corners, annotated)
    gates.append(gate)


class GatePoseRunner:
    """Background thread: detect gates on the latest decoded frame and publish
    results to data["pose"] = {gates, annotated, frame_id, infer_ms}. Started
    by VisionRX so both `make fly` and `make sim` get it for free."""

    def __init__(self, data):
        self.data = data
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="GatePose")
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _loop(self):
        last_id = -1
        warned = False
        while self.is_running:
            frame = self.data.get("frame")
            if frame is None or frame["frame_id"] == last_id:
                time.sleep(0.005)
                continue
            last_id = frame["frame_id"]
            try:
                t0 = time.perf_counter()
                gates, annotated = detect(frame["img"])
                infer_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                if not warned:
                    print(f"[gate_pose] inference disabled: {e}", flush=True)
                    warned = True
                return  # weights missing / load failed -- stop trying
            self.data["pose"] = {
                "gates": gates,
                "annotated": annotated,
                "frame_id": last_id,
                "infer_ms": infer_ms,
            }
