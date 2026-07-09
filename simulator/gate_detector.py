import numpy as np
import cv2

from simulator.config import (
    GATE_HEX_COLOR,
    HSV_TOLERANCE,
    MORPH_KERNEL_SIZE,
    MORPH_ITERS,
    MIN_CONTOUR_AREA_PX,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    GateDetection,
)
from simulator.transforms import hex_to_hsv_lower_upper

# Compute HSV bounds and morphology kernel once at import time.
_HSV_LOWER, _HSV_UPPER = hex_to_hsv_lower_upper(GATE_HEX_COLOR, HSV_TOLERANCE)
_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)

# Hue range endpoints for wraparound case (red-ish colors straddle 0/179).
_HUE_MAX_UPPER = np.array([[[179, 255, 255]]])
_HUE_MIN_LOWER = np.array([[[0, 0, 0]]])


def _color_mask(hsv: np.ndarray) -> np.ndarray:
    if _HSV_LOWER[0][0] > _HSV_UPPER[0][0]:
        mask1 = cv2.inRange(hsv, _HSV_LOWER, _HUE_MAX_UPPER)
        mask2 = cv2.inRange(hsv, _HUE_MIN_LOWER, _HSV_UPPER)
        return cv2.bitwise_or(mask1, mask2)
    return cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)


def _desaturated_orange_mask(hsv: np.ndarray) -> np.ndarray:
    """VQ2 R2 scanned gates are often less saturated than VQ1."""
    h_lo = max(0, int(_HSV_LOWER[0][0]) - 12)
    h_hi = min(179, int(_HSV_UPPER[0][0]) + 12)
    return cv2.inRange(
        hsv,
        np.array([[[h_lo, 25, 35]]], dtype=np.uint8),
        np.array([[[h_hi, 140, 255]]], dtype=np.uint8),
    )


def _best_gate_contour(mask: np.ndarray) -> np.ndarray | None:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL, iterations=MORPH_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=MORPH_ITERS)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = mask.shape[:2]
    half_w, half_h = img_w / 2.0, img_h / 2.0
    best_contour = None
    best_score = -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA_PX:
            continue
        _, _, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            continue
        moments = cv2.moments(c)
        m00 = max(moments["m00"], 1e-6)
        cx = moments["m10"] / m00
        cy = moments["m01"] / m00
        dist = abs(cx - half_w) + abs(cy - half_h)
        score = area / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_contour = c
    return best_contour


def detect_gate(
    img: np.ndarray, frame_id: int, sim_time_ns: int
) -> GateDetection | None:
    """Detect a hex-colored gate in a BGR image frame.

    Pipeline: HSV threshold (with hue wraparound) → morphology open+close →
    contour filter (area + aspect) → largest surviving contour reported.

    Args:
        img: BGR input image (H, W, 3).
        frame_id: Sequential frame identifier.
        sim_time_ns: Simulator timestamp in nanoseconds.

    Returns:
        GateDetection for the largest valid contour, or None when no gate
        contour passes the filters / input is empty.
    """
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None

    # 1. Convert to HSV colorspace.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    best_contour = _best_gate_contour(_color_mask(hsv))
    if best_contour is None:
        best_contour = _best_gate_contour(_desaturated_orange_mask(hsv))

    if best_contour is None:
        return None

    moments = cv2.moments(best_contour)
    m00 = max(moments["m00"], 1e-6)
    cx = moments["m10"] / m00
    cy = moments["m01"] / m00
    _, _, w, h = cv2.boundingRect(best_contour)
    best_area = cv2.contourArea(best_contour)

    return GateDetection(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        centroid_x_px=cx,
        centroid_y_px=cy,
        area_px=best_area,
        width_px=float(w),
        height_px=float(h),
        contour_valid=True,
    )
