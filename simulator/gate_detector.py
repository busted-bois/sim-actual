import cv2
import numpy as np

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

_HSV_LOWER, _HSV_UPPER = hex_to_hsv_lower_upper(GATE_HEX_COLOR, HSV_TOLERANCE)
_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)
_HUE_MAX_UPPER = np.array([[[179, 255, 255]]])
_HUE_MIN_LOWER = np.array([[[0, 0, 0]]])


def detect_gate(
    img: np.ndarray, frame_id: int, sim_time_ns: int
) -> GateDetection | None:
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if _HSV_LOWER[0][0] > _HSV_UPPER[0][0]:
        mask1 = cv2.inRange(hsv, _HSV_LOWER, _HUE_MAX_UPPER)
        mask2 = cv2.inRange(hsv, _HUE_MIN_LOWER, _HSV_UPPER)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL, iterations=MORPH_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=MORPH_ITERS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA_PX:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            continue
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    if largest_contour is None:
        return None

    moments = cv2.moments(largest_contour)
    m00 = max(moments["m00"], 1e-6)
    cx = moments["m10"] / m00
    cy = moments["m01"] / m00
    _, _, w, h = cv2.boundingRect(largest_contour)

    return GateDetection(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        centroid_x_px=cx,
        centroid_y_px=cy,
        area_px=largest_area,
        width_px=float(w),
        height_px=float(h),
        contour_valid=True,
    )
