"""FPV gate detection (HSV + Hough circles). Images are BGR from OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from simulator.flight_config import PNP_R_FRAC_CAP
from simulator.gate_pnp import order_corners, solve_gate_pnp


def _blue_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 110, 60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def find_blue_rings(image_bgr: np.ndarray) -> list[tuple[int, int, int]]:
    mask = _blue_mask(image_bgr)
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


def find_gate_corners(image_bgr: np.ndarray) -> list[tuple[float, float]] | None:
    """Approximate inner gate square corners from blue mask contour."""
    mask = _blue_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 120.0:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    if box is None or len(box) < 4:
        return None
    return order_corners(box.tolist())


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
    corners = find_gate_corners(image_bgr)
    pnp = solve_gate_pnp(corners) if corners else None

    if info is None and pnp is None:
        return {
            "detected": False,
            "nx": 0.0,
            "ny": 0.0,
            "r_frac": 0.0,
            "corners": None,
            "pnp": None,
        }

    if info is None and pnp is not None:
        lateral = pnp["lateral_m"]
        range_m = pnp["range_m"]
        nx = lateral / max(range_m, 0.5)
        ny = 0.0
        r_frac = min(PNP_R_FRAC_CAP, 0.5 / max(range_m, 0.5))
    else:
        nx, ny, r_frac = info

    return {
        "detected": True,
        "nx": float(nx),
        "ny": float(ny),
        "r_frac": float(r_frac),
        "corners": corners,
        "pnp": pnp,
    }
