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
from rl.pnp import detect_corners, estimate_pose

MAX_REPROJ_ERR_PX = 15.0

# Compute HSV bounds and morphology kernel once at import time.
_HSV_LOWER, _HSV_UPPER = hex_to_hsv_lower_upper(GATE_HEX_COLOR, HSV_TOLERANCE)
_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)

# Hue range endpoints for wraparound case (red-ish colors straddle 0/179).
_HUE_MAX_UPPER = np.array([[[179, 255, 255]]])
_HUE_MIN_LOWER = np.array([[[0, 0, 0]]])


def build_gate_mask(img: np.ndarray) -> np.ndarray | None:
    """HSV gate mask for PnP / VIO (morphology-cleaned, 0/255)."""
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
    return mask


def detect_gate(
    img: np.ndarray,
    frame_id: int,
    sim_time_ns: int,
    corner_roi: tuple[int, int, int, int] | None = None,
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
    mask = build_gate_mask(img)
    if mask is None:
        return None

    # Extract external contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter by area and aspect ratio; score by area / distance-to-center.
    best_contour = None
    best_score = -1.0
    img_h, img_w = mask.shape[:2]
    half_w, half_h = img_w / 2.0, img_h / 2.0
    contours_area_pass = 0
    contours_aspect_pass = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA_PX:
            continue
        contours_area_pass += 1
        _, _, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            continue
        contours_aspect_pass += 1
        moments = cv2.moments(c)
        m00 = max(moments["m00"], 1e-6)
        cx = moments["m10"] / m00
        cy = moments["m01"] / m00
        dist = abs(cx - half_w) + abs(cy - half_h)
        score = area / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_contour = c

    # 6. Nothing passed the filters.
    if best_contour is None:
        return None

    # 7. Centroid via image moments (guarded against m00 == 0).
    moments = cv2.moments(best_contour)
    m00 = max(moments["m00"], 1e-6)
    cx = moments["m10"] / m00
    cy = moments["m01"] / m00
    _, _, w, h = cv2.boundingRect(best_contour)
    best_area = cv2.contourArea(best_contour)

    corners_px = None
    reproj_err_px = None
    quality = 0.0

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, -1)
    corners = detect_corners(contour_mask, roi=corner_roi)
    if corners is not None:
        corners_px = tuple((float(x), float(y)) for x, y in corners.reshape(4, 2))
        pose = estimate_pose(corners)
        if pose is not None:
            reproj_err_px = float(pose["reproj_err_px"])
            quality = max(0.0, 1.0 - reproj_err_px / MAX_REPROJ_ERR_PX)
            aspect = w / max(h, 1)
            area_frac = best_area / max(img_h * img_w, 1)
            if MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO and area_frac < 0.15:
                quality = min(1.0, quality * (0.5 + 0.5 * min(area_frac / 0.08, 1.0)))

    return GateDetection(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        centroid_x_px=cx,
        centroid_y_px=cy,
        area_px=best_area,
        width_px=float(w),
        height_px=float(h),
        contour_valid=True,
        corners_px=corners_px,
        reproj_err_px=reproj_err_px,
        quality=quality,
    )
