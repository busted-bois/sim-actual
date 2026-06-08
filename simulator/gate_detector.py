from dataclasses import dataclass

import cv2
import numpy as np

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CX = 320.0
CY = 180.0
FX = 320.0
OUTER_GATE_WIDTH_M = 2.7
INNER_GATE_WIDTH_M = 1.5

MIN_AREA_RATIO = 0.001
OUTER_ASPECT_MIN = 0.45
OUTER_ASPECT_MAX = 2.2
FILL_RATIO_MIN = 0.04
FILL_RATIO_MAX = 0.95
INNER_ASPECT_MIN = 0.7
INNER_ASPECT_MAX = 1.4
RANGE_MISMATCH_MAX = 0.40
LOCK_CONFIDENCE = 0.55
STRONG_LOCK_CONFIDENCE = 0.75


@dataclass(frozen=True)
class GateDetection:
    frame_id: int
    target_x: float
    target_y: float
    bbox: tuple[int, int, int, int]
    inner_bbox: tuple[int, int, int, int] | None
    range_m: float
    confidence: float
    area_ratio: float
    clipped: bool
    candidate_count: int

    @property
    def ex(self):
        return (self.target_x - CX) / CX

    @property
    def ey(self):
        return (self.target_y - CY) / CY

    @property
    def strong_lock(self):
        return self.confidence >= STRONG_LOCK_CONFIDENCE


class GateDetector:
    def __init__(self):
        self.previous_target = None
        self.last_stats = {}

    def detect(self, frame_id, frame):
        mask = self._color_mask(frame)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_pixels = cv2.countNonZero(mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            self.previous_target = None
            self.last_stats = {"mask_px": mask_pixels, "contours": 0, "candidates": 0}
            return None

        candidates = []
        hierarchy = hierarchy[0]
        for index, contour in enumerate(contours):
            if hierarchy[index][3] != -1:
                continue
            candidate = self._candidate(frame_id, mask, contour, contours, hierarchy, index)
            if candidate is not None:
                candidates.append(candidate)
        self.last_stats = {"mask_px": mask_pixels, "contours": len(contours), "candidates": len(candidates)}

        if not candidates:
            self.previous_target = None
            return None

        candidates.sort(key=self._selection_score, reverse=True)
        detection = candidates[0]
        detection = GateDetection(
            frame_id=detection.frame_id,
            target_x=detection.target_x,
            target_y=detection.target_y,
            bbox=detection.bbox,
            inner_bbox=detection.inner_bbox,
            range_m=detection.range_m,
            confidence=detection.confidence,
            area_ratio=detection.area_ratio,
            clipped=detection.clipped,
            candidate_count=len(candidates),
        )
        self.previous_target = (detection.target_x, detection.target_y)
        return detection

    def _color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([25, 255, 255]))
        upper_red = cv2.inRange(hsv, np.array([160, 60, 40]), np.array([179, 255, 255]))
        hsv_mask = cv2.bitwise_or(lower_red, upper_red)

        bgr_mask = cv2.inRange(frame, np.array([0, 20, 100]), np.array([120, 180, 255]))

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        target = np.uint8([[[15, 57, 243]]])
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)
        diff = lab.astype(np.int16) - target_lab
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        lab_mask = (dist < 95).astype(np.uint8) * 255

        mask = cv2.bitwise_or(hsv_mask, bgr_mask)
        return cv2.bitwise_or(mask, lab_mask)

    def _candidate(self, frame_id, mask, contour, contours, hierarchy, index):
        frame_area = FRAME_WIDTH * FRAME_HEIGHT
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h else 0.0
        bbox_area = w * h
        orange_area = cv2.countNonZero(mask[y : y + h, x : x + w])
        area_ratio = orange_area / frame_area
        fill_ratio = orange_area / bbox_area if bbox_area else 0.0

        if area_ratio < MIN_AREA_RATIO:
            return None
        if not OUTER_ASPECT_MIN <= aspect <= OUTER_ASPECT_MAX:
            return None
        if not FILL_RATIO_MIN <= fill_ratio <= FILL_RATIO_MAX:
            return None

        inner_bbox = self._inner_bbox(contours, hierarchy, index)
        target_x = x + w / 2
        target_y = y + h / 2
        hole_score = 0.0
        range_m = OUTER_GATE_WIDTH_M * FX / max(w, 1)

        if inner_bbox is not None:
            ix, iy, iw, ih = inner_bbox
            target_x = ix + iw / 2
            target_y = iy + ih / 2
            inner_range_m = INNER_GATE_WIDTH_M * FX / max(iw, 1)
            mismatch = abs(range_m - inner_range_m) / max(range_m, inner_range_m)
            if mismatch > RANGE_MISMATCH_MAX:
                hole_score = 0.05
            else:
                hole_score = 1.0

        area_score = min(area_ratio / 0.20, 1.0)
        aspect_score = 1.0 - min(abs(1.0 - aspect) / 0.33, 1.0)
        fill_score = 1.0 - min(abs(0.69 - fill_ratio) / 0.54, 1.0)
        stability_score = self._stability_score(target_x, target_y)
        confidence = (
            0.25 * area_score
            + 0.20 * aspect_score
            + 0.20 * fill_score
            + 0.20 * hole_score
            + 0.15 * stability_score
        )

        clipped = x <= 0 or y <= 0 or x + w >= FRAME_WIDTH - 1 or y + h >= FRAME_HEIGHT - 1
        if clipped and self.previous_target is None:
            confidence *= 0.7

        return GateDetection(
            frame_id=frame_id,
            target_x=target_x,
            target_y=target_y,
            bbox=(x, y, w, h),
            inner_bbox=inner_bbox,
            range_m=range_m,
            confidence=max(0.0, min(confidence, 1.0)),
            area_ratio=area_ratio,
            clipped=clipped,
            candidate_count=1,
        )

    def _inner_bbox(self, contours, hierarchy, parent_index):
        child_index = hierarchy[parent_index][2]
        best = None
        best_area = 0.0
        while child_index != -1:
            contour = contours[child_index]
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h else 0.0
            area = cv2.contourArea(contour)
            if INNER_ASPECT_MIN <= aspect <= INNER_ASPECT_MAX and area > best_area:
                best = (x, y, w, h)
                best_area = area
            child_index = hierarchy[child_index][0]
        return best

    def _stability_score(self, target_x, target_y):
        if self.previous_target is None:
            return 0.5
        px, py = self.previous_target
        dist = ((target_x - px) ** 2 + (target_y - py) ** 2) ** 0.5
        return max(0.0, 1.0 - dist / 120.0)

    def _selection_score(self, detection):
        _, _, w, h = detection.bbox
        bbox_ratio = (w * h) / (FRAME_WIDTH * FRAME_HEIGHT)
        area_bonus = min(bbox_ratio / 0.010, 1.0) * 0.55
        stability_bonus = self._stability_score(detection.target_x, detection.target_y) * 0.15
        return detection.confidence + area_bonus + stability_bonus
