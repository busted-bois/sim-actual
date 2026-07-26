"""Classical HSV dual-cyan corridor detection (no YOLO).

Two parallel blue ribbons define left/right track edges. Near-field ROI
centroids give lateral/vertical corridor error; far ROI gives heading.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV HSV: H 0-179. Bright cyan/electric-blue glow from the AI-GP ribbon.
_CYAN_LOWER = np.array([85, 70, 50], dtype=np.uint8)
_CYAN_UPPER = np.array([140, 255, 255], dtype=np.uint8)

_MORPH_KSIZE = 5
_NEAR_FRAC = 0.40  # lower fraction of frame = near field
_FAR_FRAC = 0.55  # upper fraction used for heading (from top)
_MIN_SIDE_AREA = 80.0
_MIN_TOTAL_AREA = 120.0


@dataclass(frozen=True)
class BlueLineEstimate:
    found: bool
    cx_norm: float = 0.0  # corridor mid, -1 left .. +1 right
    cy_norm: float = 0.0  # avg near height, -1 top .. +1 bottom
    heading_err: float = 0.0  # rad, + = corridor vanishes right of center
    width_norm: float = 0.0  # left-right separation / image width
    left_found: bool = False
    right_found: bool = False
    frame_id: int = 0


def cyan_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CYAN_LOWER, _CYAN_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KSIZE, _MORPH_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _centroid(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, area) or None if empty."""
    area = float(cv2.countNonZero(mask))
    if area < _MIN_SIDE_AREA:
        return None
    m = cv2.moments(mask)
    m00 = m["m00"]
    if m00 < 1e-6:
        return None
    return m["m10"] / m00, m["m01"] / m00, area


def _side_centroids(
    mask: np.ndarray, y0: int, y1: int
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Left/right centroids in horizontal band [y0, y1)."""
    h, w = mask.shape[:2]
    y0 = max(0, min(h, y0))
    y1 = max(y0, min(h, y1))
    band = np.zeros_like(mask)
    band[y0:y1, :] = mask[y0:y1, :]
    mid = w // 2
    left = _centroid(band[:, :mid])
    right = _centroid(band[:, mid:])
    if left is not None:
        left = (left[0], left[1], left[2])
    if right is not None:
        right = (right[0] + mid, right[1], right[2])
    return left, right


def detect_blue_lines(
    bgr: np.ndarray, frame_id: int = 0, *, return_mask: bool = False
) -> tuple[BlueLineEstimate, np.ndarray | None]:
    """Detect dual cyan lines; return estimate and optional mask."""
    h, w = bgr.shape[:2]
    mask = cyan_mask(bgr)
    total = float(cv2.countNonZero(mask))
    if total < _MIN_TOTAL_AREA:
        est = BlueLineEstimate(found=False, frame_id=frame_id)
        return est, (mask if return_mask else None)

    near_y0 = int(h * (1.0 - _NEAR_FRAC))
    left_n, right_n = _side_centroids(mask, near_y0, h)

    far_y1 = int(h * _FAR_FRAC)
    left_f, right_f = _side_centroids(mask, 0, far_y1)

    left_found = left_n is not None
    right_found = right_n is not None
    if not left_found and not right_found:
        # Fall back to full-frame sides if near ROI empty (looking at ribbon).
        left_n, right_n = _side_centroids(mask, 0, h)
        left_found = left_n is not None
        right_found = right_n is not None

    if not left_found and not right_found:
        est = BlueLineEstimate(found=False, frame_id=frame_id)
        return est, (mask if return_mask else None)

    half_w = w / 2.0
    half_h = h / 2.0

    if left_found and right_found:
        mid_x = 0.5 * (left_n[0] + right_n[0])
        mid_y = 0.5 * (left_n[1] + right_n[1])
        width_norm = abs(right_n[0] - left_n[0]) / float(w)
    elif left_found:
        # Only left: bias corridor right of the line by ~0.25 of half-width.
        mid_x = left_n[0] + 0.25 * w
        mid_y = left_n[1]
        width_norm = 0.0
    else:
        mid_x = right_n[0] - 0.25 * w
        mid_y = right_n[1]
        width_norm = 0.0

    # Altitude cue: mid/far ribbon height (elevated path), not near floor paint.
    mid_band_y0 = int(h * 0.25)
    mid_band_y1 = int(h * 0.65)
    left_m, right_m = _side_centroids(mask, mid_band_y0, mid_band_y1)
    if left_m is not None and right_m is not None:
        alt_y = 0.5 * (left_m[1] + right_m[1])
    elif left_f is not None and right_f is not None:
        alt_y = 0.5 * (left_f[1] + right_f[1])
    else:
        alt_y = mid_y

    cx_norm = float(np.clip((mid_x - half_w) / half_w, -1.5, 1.5))
    cy_norm = float(np.clip((alt_y - half_h) / half_h, -1.5, 1.5))

    # Heading: vanishing of far mid relative to near mid (image x).
    heading_err = 0.0
    if left_f is not None and right_f is not None and left_found and right_found:
        far_mid_x = 0.5 * (left_f[0] + right_f[0])
        near_mid_x = 0.5 * (left_n[0] + right_n[0])
        # Positive = corridor bends/vanishes to the right → yaw right.
        heading_err = float(np.arctan2(far_mid_x - near_mid_x, max(h * 0.3, 1.0)))
    elif left_found and right_found:
        # Fit from near line tilt: average of left/right column slopes via far.
        if left_f is not None:
            heading_err = float(
                np.arctan2(left_f[0] - left_n[0], max(left_n[1] - left_f[1], 1.0))
            )
        elif right_f is not None:
            heading_err = float(
                np.arctan2(right_f[0] - right_n[0], max(right_n[1] - right_f[1], 1.0))
            )

    est = BlueLineEstimate(
        found=True,
        cx_norm=cx_norm,
        cy_norm=cy_norm,
        heading_err=heading_err,
        width_norm=float(width_norm),
        left_found=left_found,
        right_found=right_found,
        frame_id=frame_id,
    )
    return est, (mask if return_mask else None)


def estimate_to_dict(est: BlueLineEstimate) -> dict:
    return {
        "found": est.found,
        "cx_norm": est.cx_norm,
        "cy_norm": est.cy_norm,
        "heading_err": est.heading_err,
        "width_norm": est.width_norm,
        "left_found": est.left_found,
        "right_found": est.right_found,
        "frame_id": est.frame_id,
    }


def annotate_blue_lines(
    bgr: np.ndarray, est: BlueLineEstimate, mask: np.ndarray | None
) -> np.ndarray:
    """Overlay cyan mask tint + corridor mid for the live vision window."""
    out = bgr.copy()
    if mask is not None:
        tint = np.zeros_like(out)
        tint[:, :] = (255, 200, 0)  # cyan-ish BGR highlight
        m = mask > 0
        out[m] = (0.45 * out[m] + 0.55 * tint[m]).astype(np.uint8)

    h, w = out.shape[:2]
    hud = "no blue line"
    if est.found:
        cx = int((est.cx_norm * 0.5 + 0.5) * w)
        cy = int((est.cy_norm * 0.5 + 0.5) * h)
        cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)
        cv2.line(out, (w // 2, h), (cx, cy), (0, 255, 255), 2)
        hud = (
            f"BLUE cx={est.cx_norm:+.2f} cy={est.cy_norm:+.2f} "
            f"hdg={est.heading_err:+.2f} L={int(est.left_found)} R={int(est.right_found)}"
        )
    cv2.putText(
        out,
        hud,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
