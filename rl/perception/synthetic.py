"""Deterministic synthetic gate scene for offline perception QA.

GateNet weights are absent in this repo (the user trains them), so synthetic is
the only perception QA path here. ``make_synthetic_gate`` derives a normalized
image tensor and the four ordered gate corners from ``rl.core.spec`` intrinsics
and gate geometry — no real images, no training, no saved weights.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from rl.core import spec
from rl.perception.gatenet import IMAGENET_MEAN, IMAGENET_STD, TRAIN_H, TRAIN_W

_GATE_BGR = (10, 60, 240)  # matches gatenet._make_synthetic_ds
_BG_VALUE = 40


def _scene(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (drone_pos, drone_quat, gate_pos, gate_quat) for a seed.

    The gate sits in the drone body frame so a yaw offset diversifies the view
    without sliding it out of frame. A non-zero lateral offset is required: a
    perfectly centred, axis-aligned gate is fronto-parallel and IPPE_SQUARE PnP
    is undefined (no perspective).
    """
    rng = np.random.default_rng(seed)
    dist = float(rng.uniform(7.5, 9.5))
    yaw = float(rng.uniform(-0.2, 0.2))
    lat = float(rng.uniform(0.4, 0.7)) * float(rng.choice([-1.0, 1.0]))
    alt = float(rng.uniform(-0.4, -0.1))

    drone_pos = np.zeros(3, dtype=np.float64)
    drone_quat = np.array(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64
    )
    r_wb = spec.quat_to_R(drone_quat)
    gate_pos = drone_pos + r_wb @ np.array([dist, lat, alt], dtype=np.float64)
    gate_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return drone_pos, drone_quat, gate_pos, gate_quat


def make_synthetic_gate(seed: int = 0):
    """Return a deterministic synthetic gate scene ``(img, corners)``.

    Args:
        seed: integer RNG seed controlling the always-valid gate placement.

    Returns:
        img: ``torch.Tensor`` shape ``(1, 3, TRAIN_H, TRAIN_W)``, finite
            ``float32``, ImageNet-normalized (the preprocessing GateNet applies).
        corners: ``np.ndarray`` shape ``(4, 2)``, finite ``float32`` pixels in
            ``spec.K`` resolution, ordered TL/TR/BR/BL — the order
            ``pnp.estimate_pose`` expects, so it yields a valid pose.
    """
    drone_pos, drone_quat, gate_pos, gate_quat = _scene(seed)
    corners_world = spec.gate_corners_world(gate_pos, gate_quat)
    px, in_front = spec.project(corners_world, drone_pos, drone_quat)
    if not bool(in_front.all()):
        raise RuntimeError("synthetic gate corners fell behind the camera")
    corners = px.astype(np.float32, copy=False)

    img = np.full((spec.IMG_H, spec.IMG_W, 3), _BG_VALUE, dtype=np.uint8)
    poly = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [poly], _GATE_BGR)
    small = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor, corners
