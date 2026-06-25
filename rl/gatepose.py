"""Module 3 (replacement) — YOLO11-pose gate-corner detector.

Supersedes the GateNet U-Net (mask -> contour -> corners). A YOLO11-pose CNN
regresses the gate's 4 opening corners directly as keypoints, so we skip the
brittle segmentation->contour step and feed PnP (Module 4) ordered corners in
one forward pass.

Pretrained base: ``yolo11n-pose.pt`` (COCO-pretrained nano pose), fine-tuned on
the auto-labeled gate set from Module 2. Keypoint order is TL,TR,BR,BL — the
SAME order as spec.GATE_CORNERS_LOCAL and pnp._OBJ, so detector output drops
straight into pnp.estimate_pose with no reordering.

``ultralytics`` is imported lazily (only train()/GatePoseInfer touch it) so this
module — and its self-test — load without the dep installed.

    uv run -m rl.gatepose --train               # fine-tune on rl/data/gate_pose_ds
    uv run -m rl.gatepose --selftest            # post-processing smoke (no model)
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass

import numpy as np

from rl import spec

# COCO-pretrained YOLO11 SMALL-pose: ~3x the capacity of nano, much more robust
# to live-flight frames (motion, lighting). ~2x inference cost — fine on the GPU.
BASE_MODEL = "yolo11s-pose.pt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "gate_pose_ds")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "gate_pose.pt")

IMGSZ = 640  # matches the 640x360 frame (letterboxed); see SPEC.md Vision Stream
DET_CONF_THRESHOLD = 0.10  # LOW on purpose: the detector fires correct boxes at
# 0.1-0.2 conf on hard flight frames (motion/close/angle); 0.25 threw those away
# and caused the det=0 dropouts. The servo/tracker tolerate the odd false box.
KEYPOINT_CONF_THRESHOLD = 0.7  # a corner must beat this to be trusted by PnP
EDGE_MARGIN_PX = 2.0  # reject corners clipped to the image border (clip-to-edge bug)


@dataclass
class GateDet:
    """One detected gate. corners/kpt_conf are in label order TL,TR,BR,BL."""

    corners: np.ndarray  # (4,2) pixel coords
    kpt_conf: np.ndarray  # (4,) per-corner confidence
    det_conf: float  # gate-box confidence
    bbox: np.ndarray  # (4,) xyxy pixels


def _on_edge(pts: np.ndarray, w: int, h: int, margin: float) -> np.ndarray:
    """True for keypoints within `margin` px of an image border.

    YOLO tends to slam an off-screen corner onto the nearest edge with high
    confidence; treating those as invisible stops them from poisoning PnP.
    """
    x, y = pts[:, 0], pts[:, 1]
    return (x < margin) | (x > w - margin) | (y < margin) | (y > h - margin)


def visible_mask(
    det: GateDet,
    w: int = spec.IMG_W,
    h: int = spec.IMG_H,
    conf_th: float = KEYPOINT_CONF_THRESHOLD,
    margin: float = EDGE_MARGIN_PX,
) -> np.ndarray:
    """Boolean (4,): corners confident enough AND not clipped to the border."""
    return (np.asarray(det.kpt_conf, float) > conf_th) & ~_on_edge(
        np.asarray(det.corners, float), w, h, margin
    )


class GatePoseInfer:
    """Live YOLO11-pose inference wrapper. Lazy-imports ultralytics."""

    def __init__(
        self, weights: str = WEIGHTS_PATH, device: str | None = None, half: bool = True
    ):
        from ultralytics import YOLO  # lazy

        if not os.path.exists(weights) and not str(weights).endswith(".pt"):
            raise FileNotFoundError(weights)
        self.model = YOLO(weights)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(half) and device != "cpu"

    def detect(self, img_bgr: np.ndarray) -> list[GateDet]:
        res = self.model.predict(
            source=img_bgr,
            imgsz=IMGSZ,
            conf=DET_CONF_THRESHOLD,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        out: list[GateDet] = []
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return out
        kxy = res.keypoints.xy.cpu().numpy()  # (N,4,2)
        kcf = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones(kxy.shape[:2], np.float32)
        )
        bxy = res.boxes.xyxy.cpu().numpy()
        bcf = res.boxes.conf.cpu().numpy()
        for i in range(len(bxy)):
            if float(bcf[i]) < DET_CONF_THRESHOLD:
                continue
            out.append(
                GateDet(
                    corners=kxy[i].astype(np.float64),
                    kpt_conf=kcf[i].astype(np.float64),
                    det_conf=float(bcf[i]),
                    bbox=bxy[i].astype(np.float64),
                )
            )
        return out


def train(
    data_yaml: str = DATA_YAML,
    base: str = BASE_MODEL,
    epochs: int = 100,
    imgsz: int = IMGSZ,
    out: str = WEIGHTS_PATH,
):
    """Fine-tune YOLO11-pose on the gate set; copy best.pt -> rl/data/gate_pose.pt."""
    from ultralytics import YOLO

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"{data_yaml} missing — run `uv run -m rl.dataset --pose` first"
        )
    model = YOLO(base)
    # The synthetic generator (rl.synth_data) already randomizes pose, lighting,
    # colour, blur, noise and occlusion, so YOLO's own heavy aug (mosaic/mixup) is
    # redundant AND was starving the GPU (4-image composites bottlenecked the
    # dataloader). Dropping mosaic = 1 image read per sample instead of 4, which
    # is the real speedup. No RAM cache (it blew the Windows paging file), and
    # workers=0 (single-process data loading) — multiprocessing dataloader workers
    # kept conflicting with CUDA on this Windows box ("resource already mapped" /
    # paging-file errors). Loading is light enough now that single-process is fine.
    res = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=DATA_DIR,
        name="train",
        workers=0,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
    )
    best = os.path.join(str(res.save_dir), "weights", "best.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(best, out)
    print(f"[gatepose] {best} -> {out}", flush=True)


# ----------------------------------------------------------------------------
# Self-test: post-processing only (no model / no ultralytics needed).
# ----------------------------------------------------------------------------
def _selftest():
    # A clean head-on gate: 4 corners well inside the frame, all confident.
    corners = np.array(
        [[260, 120], [380, 120], [380, 240], [260, 240]], dtype=np.float64
    )
    det = GateDet(
        corners=corners,
        kpt_conf=np.ones(4),
        det_conf=0.9,
        bbox=np.array([260, 120, 380, 240], float),
    )
    vis = visible_mask(det)
    assert vis.all(), f"all 4 corners should be visible, got {vis}"

    # One corner clipped to the left edge + one low-confidence -> both rejected.
    det2 = GateDet(
        corners=np.array([[1, 120], [380, 120], [380, 240], [260, 240]], float),
        kpt_conf=np.array([0.9, 0.9, 0.4, 0.9]),
        det_conf=0.9,
        bbox=np.array([1, 120, 380, 240], float),
    )
    vis2 = visible_mask(det2)
    assert not vis2[0], "edge-clipped corner must be rejected"
    assert not vis2[2], "low-confidence corner must be rejected"
    assert vis2[1] and vis2[3], "good corners must survive"
    assert int(vis2.sum()) == 2
    print(
        f"[selftest] base={BASE_MODEL} kpts={spec.N_KEYPOINTS} "
        f"conf_th={KEYPOINT_CONF_THRESHOLD} edge={EDGE_MARGIN_PX}px"
    )
    print("[selftest] OK — keypoint confidence + edge gating correct")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.train:
        train(epochs=args.epochs)
    else:
        _selftest()
