"""
Color-based detection for the autonomous pilot.

Two things are detected per frame:
  * the gate   - an orange/red-orange blob (#f3390f-ish) we steer toward and fly through
  * the path   - a blue trail used as a fallback heading when no gate is visible,
                 and (together with the gate) as the "is there still course ahead?"
                 signal that drives end-of-course safety.

Everything here is a pure function of (image, config): no sockets, no threads,
no global state. That keeps it trivially unit-testable on synthetic frames
(see tests/test_vision_processing.py).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """One detected color blob, in normalized image coordinates."""

    found: bool
    cx_norm: float = 0.0  # horizontal offset of blob center from frame center, -1 (left) .. +1 (right)
    cy_norm: float = 0.0  # vertical offset from frame center, -1 (top) .. +1 (bottom)
    area_frac: float = (
        0.0  # blob area as a fraction of the whole frame (proxy for closeness)
    )
    pixel_count: int = 0


@dataclass
class FrameAnalysis:
    """Result of analyzing a single camera frame."""

    frame_id: int
    width: int
    height: int
    gate: Detection
    path: Detection
    timestamp: float

    @property
    def any_detection(self):
        return self.gate.found or self.path.found


def _to_np_ranges(hsv_ranges):
    """Convert JSON [[lo],[hi]] pairs into (lo_array, hi_array) uint8 tuples."""
    out = []
    for lo, hi in hsv_ranges:
        out.append((np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    return out


def color_mask(hsv_img, hsv_ranges, morph_ksize=5):
    """
    Build a binary mask of every pixel falling inside any of ``hsv_ranges``.

    Ranges are OR-ed together so a color straddling the hue wrap (e.g. red-orange)
    can be captured with a low band and a high band. A morphological open+close
    removes speckle and fills small holes.
    """
    mask = None
    for lo, hi in _to_np_ranges(hsv_ranges):
        band = cv2.inRange(hsv_img, lo, hi)
        mask = band if mask is None else cv2.bitwise_or(mask, band)

    if mask is None:  # empty range list -> nothing matches
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    if morph_ksize and morph_ksize > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_from_mask(mask, width, height, min_area_frac):
    """
    Reduce a binary mask to a single :class:`Detection` (the largest blob).

    Using the largest contour rather than the raw pixel centroid makes the
    detection robust to scattered same-color noise elsewhere in the frame.
    """
    frame_area = float(width * height)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Detection(found=False)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area_frac * frame_area:
        return Detection(found=False)

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return Detection(found=False)

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return Detection(
        found=True,
        # map pixel [0,w] -> [-1,+1] about the center
        cx_norm=(cx - width / 2.0) / (width / 2.0),
        cy_norm=(cy - height / 2.0) / (height / 2.0),
        area_frac=area / frame_area,
        pixel_count=int(area),
    )


def analyze_frame(img_bgr, vision_cfg, frame_id=0, timestamp=0.0):
    """
    Detect the gate and the path in a BGR frame.

    Returns a :class:`FrameAnalysis`. ``timestamp`` should be a monotonic-ish
    seconds value supplied by the caller (kept out of here so the function stays
    pure and deterministic for tests).
    """
    height, width = img_bgr.shape[:2]

    blur = vision_cfg.get("blur_ksize", 0)
    if blur and blur > 1:
        img_bgr = cv2.GaussianBlur(img_bgr, (blur, blur), 0)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    morph = vision_cfg.get("morph_ksize", 5)
    min_area = vision_cfg.get("min_blob_area_frac", 0.0008)

    gate_mask = color_mask(hsv, vision_cfg["gate_hsv_ranges"], morph)
    path_mask = color_mask(hsv, vision_cfg["path_hsv_ranges"], morph)

    gate = detect_from_mask(gate_mask, width, height, min_area)
    path = detect_from_mask(path_mask, width, height, min_area)

    return FrameAnalysis(
        frame_id=frame_id,
        width=width,
        height=height,
        gate=gate,
        path=path,
        timestamp=timestamp,
    )


def annotate(img_bgr, analysis):
    """Draw detection markers onto a copy of the frame (for the debug view / tool)."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

    for det, color, label in (
        (analysis.gate, (0, 140, 255), "GATE"),
        (analysis.path, (255, 100, 0), "PATH"),
    ):
        if not det.found:
            continue
        px = int((det.cx_norm * w / 2.0) + w / 2.0)
        py = int((det.cy_norm * h / 2.0) + h / 2.0)
        cv2.circle(out, (px, py), 12, color, 2)
        cv2.line(out, (w // 2, h // 2), (px, py), color, 1)
        cv2.putText(
            out,
            f"{label} a={det.area_frac:.3f} x={det.cx_norm:+.2f}",
            (10, 25 if label == "GATE" else 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out
