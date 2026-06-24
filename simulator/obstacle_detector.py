"""Obstacle detection pipeline — separate from gate detection.

Detects dark non-gate blobs for collision avoidance. Excludes gate-orange
regions and the current gate mask/bbox so gates are not classified as obstacles.
"""

from __future__ import annotations

import cv2
import numpy as np

from simulator.config import (
    GATE_HEX_COLOR,
    HSV_TOLERANCE,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    OBSTACLE_BRIGHT_EXCLUDE,
    OBSTACLE_GRAY_THRESH,
    OBSTACLE_MIN_AREA_PX,
    GateDetection,
    ObstacleDetection,
)
from simulator.transforms import hex_to_hsv_lower_upper

_GATE_HSV_LOWER, _GATE_HSV_UPPER = hex_to_hsv_lower_upper(GATE_HEX_COLOR, HSV_TOLERANCE)


def _gate_exclusion_mask(
    img_h: int,
    img_w: int,
    gate_detection: GateDetection | None,
    gate_mask: np.ndarray | None,
) -> np.ndarray:
    exclude = np.zeros((img_h, img_w), dtype=np.uint8)
    if gate_mask is not None:
        exclude = np.maximum(exclude, (gate_mask > 127).astype(np.uint8) * 255)
    if gate_detection is not None:
        cx = int(gate_detection.centroid_x_px)
        cy = int(gate_detection.centroid_y_px)
        r = int(max(gate_detection.width_px, gate_detection.height_px) * 0.6)
        cv2.circle(exclude, (cx, cy), max(r, 20), 255, -1)
    return exclude


def detect_obstacles(
    img: np.ndarray,
    frame_id: int,
    sim_time_ns: int,
    gate_detection: GateDetection | None = None,
    gate_mask: np.ndarray | None = None,
) -> list[ObstacleDetection]:
    """Find obstacle candidates; never reuse gate HSV blobs as obstacles."""
    if img is None or img.ndim != 3:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, obs_mask = cv2.threshold(gray, OBSTACLE_GRAY_THRESH, 255, cv2.THRESH_BINARY)
    obs_mask[gray > OBSTACLE_BRIGHT_EXCLUDE] = 0

    # Drop gate-orange pixels (same color family as gates).
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if _GATE_HSV_LOWER[0][0] > _GATE_HSV_UPPER[0][0]:
        gate_color = cv2.inRange(hsv, _GATE_HSV_LOWER, np.array([[[179, 255, 255]]]))
        gate_color |= cv2.inRange(hsv, np.array([[[0, 0, 0]]]), _GATE_HSV_UPPER)
    else:
        gate_color = cv2.inRange(hsv, _GATE_HSV_LOWER, _GATE_HSV_UPPER)
    obs_mask[gate_color > 0] = 0

    exclude = _gate_exclusion_mask(h, w, gate_detection, gate_mask)
    obs_mask[exclude > 0] = 0

    contours, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(h * w)
    results: list[ObstacleDetection] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < OBSTACLE_MIN_AREA_PX:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect > MAX_ASPECT_RATIO * 2 or aspect < MIN_ASPECT_RATIO * 0.5:
            continue

        m = cv2.moments(c)
        m00 = max(m["m00"], 1e-6)
        cx = m["m10"] / m00
        cy = m["m01"] / m00

        area_frac = area / max(frame_area, 1.0)
        centrality = 1.0 - (abs(cx - w / 2) / (w / 2) + abs(cy - h / 2) / (h / 2)) / 2.0
        confidence = min(1.0, area_frac / 0.02) * max(0.2, centrality)

        results.append(
            ObstacleDetection(
                frame_id=frame_id,
                sim_time_ns=sim_time_ns,
                centroid_x_px=float(cx),
                centroid_y_px=float(cy),
                area_px=float(area),
                width_px=float(bw),
                height_px=float(bh),
                confidence=float(confidence),
            )
        )

    results.sort(key=lambda o: o.confidence, reverse=True)
    return results
