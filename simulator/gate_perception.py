"""Gate perception - lateral offset of the current gate in metres.

VQ1 approach: color threshold + contour detection of the gate opening.
Pixel offset from frame centre is scaled to metres using the active gate's
real width from track data:

    d = (px_offset / gate_bbox_width_px) * track.gates[active_index].width

Publishes into shared_data["perception"]:
    gate_lateral_m - lateral offset in metres (inf when not visible)
    gate_visible   - bool
    confidence     - 0..1 crude contour-area confidence
"""

import math

import cv2
import numpy as np

# HSV threshold for the gate frame color (VQ1 gates render orange).
# Two ranges to cover hue wrap-around near red.
GATE_HSV_LOWER_1 = np.array([0, 80, 80])
GATE_HSV_UPPER_1 = np.array([25, 255, 255])
GATE_HSV_LOWER_2 = np.array([160, 80, 80])
GATE_HSV_UPPER_2 = np.array([179, 255, 255])

MIN_CONTOUR_AREA_FRAC = 0.001  # of frame area - reject specks
FULL_CONFIDENCE_AREA_FRAC = 0.05  # contour area for confidence 1.0


class GatePerception:
    def __init__(self, data):
        self.data = data
        self._warned_no_width = False
        data["perception"] = {
            "gate_lateral_m": math.inf,
            "gate_visible": False,
            "confidence": 0.0,
        }

    def process_frame(self, img):
        visible, px_offset, bbox_w_px, confidence = self._detect_gate(img)

        gate_width_m = self._active_gate_width()
        if visible and gate_width_m is None:
            # competition config may null the widths - never commit blind
            if not self._warned_no_width:
                print(
                    "WARNING: no track gate width available - skipping commit scale",
                    flush=True,
                )
                self._warned_no_width = True
            visible = False

        if visible and bbox_w_px > 0:
            # signed: positive = gate is right of frame centre
            d = (px_offset / bbox_w_px) * gate_width_m
        else:
            d = math.inf
            confidence = 0.0

        self.data["perception"] = {
            "gate_lateral_m": d,
            "gate_visible": visible,
            "confidence": confidence,
        }

    def _active_gate_width(self):
        track = self.data.get("track")
        race = self.data.get("race")
        if not track or not track.get("gates"):
            return None
        gates = track["gates"]
        index = race["active_gate_index"] if race else 0
        if index < 0 or index >= len(gates):
            return None
        width = gates[index]["width"]
        if not width or width <= 0 or math.isnan(width):
            return None
        return width

    def _detect_gate(self, img):
        """Returns (visible, px_offset_from_center, bbox_width_px, confidence)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GATE_HSV_LOWER_1, GATE_HSV_UPPER_1) | cv2.inRange(
            hsv, GATE_HSV_LOWER_2, GATE_HSV_UPPER_2
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0.0, 0, 0.0

        frame_area = img.shape[0] * img.shape[1]
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_CONTOUR_AREA_FRAC * frame_area:
            return False, 0.0, 0, 0.0

        x, y, w, h = cv2.boundingRect(largest)
        gate_center_x = x + w / 2.0
        px_offset = gate_center_x - img.shape[1] / 2.0
        confidence = min(1.0, area / (FULL_CONFIDENCE_AREA_FRAC * frame_area))
        return True, px_offset, w, confidence
