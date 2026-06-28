"""Heuristic on-screen countdown detection (upper-center ROI)."""

from __future__ import annotations

import os

import cv2
import numpy as np

VISUAL_GO_MIN_CLEAR_FRAMES = int(os.environ.get("VISUAL_GO_MIN_CLEAR_FRAMES", "5"))


def countdown_roi_metrics(img) -> tuple[bool, float, float]:
    """Return (visible, bright_frac, edge_density) for debug logging."""
    if img is None or not hasattr(img, "shape") or img.ndim != 3:
        return False, 0.0, 0.0
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        return False, 0.0, 0.0
    y1, y2 = 0, max(1, int(h * 0.35))
    x1, x2 = int(w * 0.25), int(w * 0.75)
    if x2 <= x1:
        return False, 0.0, 0.0
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_frac = float(np.mean(gray > 200))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    visible = bright_frac > 0.08 and edge_density > 0.02
    return visible, bright_frac, edge_density


def countdown_visible(img) -> bool:
    """True when large bright countdown digits likely visible in upper-center ROI."""
    visible, _, _ = countdown_roi_metrics(img)
    return visible


def countdown_roi_rect(img_shape):
    """Return (x1, y1, x2, y2) for preview overlay."""
    h, w = img_shape[:2]
    return int(w * 0.25), 0, int(w * 0.75), max(1, int(h * 0.35))


def _countdown_gate_dict(data) -> dict:
    gate = data.get("_countdown_gate")
    if gate is None:
        gate = {"state": "idle", "clear_frames": 0}
        data["_countdown_gate"] = gate
    return gate


def reset_countdown_gate(data) -> None:
    data["_countdown_gate"] = {"state": "idle", "clear_frames": 0}


def update_countdown_gate(data, visible: bool) -> None:
    """idle -> saw_countdown -> cleared (after N clear frames)."""
    gate = _countdown_gate_dict(data)
    if visible:
        gate["state"] = "saw_countdown"
        gate["clear_frames"] = 0
        return
    if gate["state"] == "saw_countdown":
        gate["clear_frames"] = int(gate.get("clear_frames", 0)) + 1
        if gate["clear_frames"] >= VISUAL_GO_MIN_CLEAR_FRAMES:
            gate["state"] = "cleared"


def countdown_gate_cleared(data) -> bool:
    return _countdown_gate_dict(data).get("state") == "cleared"


def countdown_gate_saw(data) -> bool:
    return _countdown_gate_dict(data).get("state") in ("saw_countdown", "cleared")


def countdown_gate_state(data) -> str:
    return str(_countdown_gate_dict(data).get("state", "idle"))
