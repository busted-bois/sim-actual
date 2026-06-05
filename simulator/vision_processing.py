"""FPV gate detection (HSV + Hough circles). Images are BGR from OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def find_blue_rings(image_bgr: np.ndarray) -> list[tuple[int, int, int]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 110, 60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=50,
        param2=22,
        minRadius=4,
        maxRadius=180,
    )
    if circles is None:
        return []
    rounded = np.round(circles[0, :]).astype(int)
    return [(int(x), int(y), int(r)) for x, y, r in rounded]


def blue_ring_info_normalized(
    image_bgr: np.ndarray,
) -> tuple[float, float, float] | None:
    """Largest blue ring as (nx, ny, r_frac) relative to image center."""
    circles = find_blue_rings(image_bgr)
    if not circles:
        return None

    height, width = image_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return None

    x, y, radius = max(circles, key=lambda item: item[2])
    nx = (x - 0.5 * width) / (0.5 * width)
    ny = (y - 0.5 * height) / (0.5 * height)
    r_frac = float(radius) / max(1.0, min(width, height))
    return (float(nx), float(ny), r_frac)


def detect_gate_target(image_bgr: np.ndarray) -> dict:
    info = blue_ring_info_normalized(image_bgr)
    if info is None:
        return {"detected": False, "nx": 0.0, "ny": 0.0, "r_frac": 0.0}
    nx, ny, r_frac = info
    return {"detected": True, "nx": nx, "ny": ny, "r_frac": r_frac}
