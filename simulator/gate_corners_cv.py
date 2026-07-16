"""Classical-CV inner-corner extraction for the gate opening.

Refines a YOLO gate hit with contour-accurate corners (adapted from
yahya3867/ai-grand-prix-sim solver/gate_detector_cv.py): HSV gate mask ->
CCOMP contour hierarchy -> the hole contour of the largest gate frame ->
4 ordered inner corners. PnP on these beats the raw YOLO keypoints at range,
where a few px of keypoint error blows up monocular depth.

Color comes from our shared HSV pipeline (simulator/gate_detector.py, keyed to
GATE_HEX_COLOR with the desaturated-orange fallback for VQ2 R2 scans), not a
hard-coded red range.

find_gate_inner_corners(bgr) -> (4,2) float array [TL,TR,BR,BL] in source px,
or None.
"""

import cv2
import numpy as np

from simulator.gate_detector import _color_mask, _desaturated_orange_mask

# Inner opening is 1.5 m of a 2.7 m outer frame: shrink factor when the hole
# contour itself isn't resolvable (e.g. mask bled into the opening).
INNER_OVER_OUTER = 1.5 / 2.7

# Reference min outer-contour area (px) at 640x360, scaled to the frame.
MIN_AREA_REF_PX = 120.0
_REF_PIXELS = 640.0 * 360.0

# Corners this close to the border are clip artifacts (partial gate in frame);
# PnP on a clipped quad solves a mirrored/garbage pose.
EDGE_MARGIN_PX = 2.0

# 3x3 kernel (not gate_detector's 5x5): a distant gate's hole is only a few px
# wide and a larger closing kernel fills it, destroying the child contour.
_KERNEL3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL3, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL3, iterations=1)


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float64)
    rect = cv2.minAreaRect(cnt)
    return cv2.boxPoints(rect).astype(np.float64)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 image points to [TL, TR, BR, BL] (matches gate_pnp's inner-4
    object points for a front-facing gate)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    by_v = pts[np.argsort(pts[:, 1])]
    top = by_v[:2][np.argsort(by_v[:2, 0])]
    bottom = by_v[2:][np.argsort(by_v[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


def _near_edge(quad: np.ndarray, w: int, h: int) -> bool:
    x, y = quad[:, 0], quad[:, 1]
    return bool(
        (x < EDGE_MARGIN_PX).any()
        or (x > w - 1 - EDGE_MARGIN_PX).any()
        or (y < EDGE_MARGIN_PX).any()
        or (y > h - 1 - EDGE_MARGIN_PX).any()
    )


def _inner_quad(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    h, w = mask.shape[:2]
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    hier = hier[0]
    best_i, best_area = -1, 0.0
    for i, c in enumerate(contours):
        if hier[i][3] != -1:  # not a top-level contour
            continue
        area = cv2.contourArea(c)
        if area >= min_area and area > best_area:
            best_area, best_i = area, i
    if best_i < 0:
        return None
    # The gate opening is the frame contour's hole (child in CCOMP hierarchy).
    child = hier[best_i][2]
    if child != -1 and cv2.contourArea(contours[child]) > 0.15 * best_area:
        quad = _quad_from_contour(contours[child])
        if quad is None or _near_edge(quad, w, h):
            return None  # clipped opening -> PnP would solve a mirrored pose
        return quad
    outer = _quad_from_contour(contours[best_i])
    # A border-touching frame with no resolvable hole is a clipped gate; the
    # shrink fallback would fabricate interior corners from partial geometry.
    if outer is None or _near_edge(outer, w, h):
        return None
    c = outer.mean(axis=0)
    return c + (outer - c) * INNER_OVER_OUTER


def find_gate_inner_corners(
    bgr: np.ndarray, min_area: float | None = None
) -> np.ndarray | None:
    """Inner-opening corners [TL,TR,BR,BL] (source px) of the largest gate."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return None
    h, w = bgr.shape[:2]
    if min_area is None:
        min_area = MIN_AREA_REF_PX * (w * h) / _REF_PIXELS

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    quad = _inner_quad(_clean_mask(_color_mask(hsv)), min_area)
    if quad is None:
        quad = _inner_quad(_clean_mask(_desaturated_orange_mask(hsv)), min_area)
    if quad is None or not np.all(np.isfinite(quad)):
        return None
    return order_corners(quad)
