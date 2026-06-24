"""Gate classification: geometric + temporal validation before navigation/KF."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from simulator.config import (
    GATE_CONFIDENCE_AMBIGUOUS,
    GATE_CONFIDENCE_MIN_NAV,
    GATE_MAX_AREA_FRAC,
    GATE_TEMPORAL_MIN_STREAK,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    MIN_CONTOUR_AREA_PX,
    GateDetection,
)


@dataclass
class ClassifiedGate:
    detection: GateDetection | None
    gate_confidence: float
    temporal_streak: int
    geometric_valid: bool
    ambiguous: bool
    validated: bool


class GateClassifier:
    """Temporal + geometric gate validation separate from raw HSV detection."""

    def __init__(self, history_len: int = 8) -> None:
        self._streak = 0
        self._areas: deque[float] = deque(maxlen=history_len)
        self._nx: deque[float] = deque(maxlen=history_len)

    def classify(
        self,
        detection: GateDetection | None,
        img_w: int,
        img_h: int,
    ) -> ClassifiedGate:
        if detection is None or not detection.contour_valid:
            self._streak = max(0, self._streak - 1)
            return ClassifiedGate(
                detection=None,
                gate_confidence=0.0,
                temporal_streak=self._streak,
                geometric_valid=False,
                ambiguous=True,
                validated=False,
            )

        geometric = self._geometric_valid(detection, img_w, img_h)
        area_frac = detection.area_px / max(img_w * img_h, 1)

        if geometric:
            base = detection.quality
            # Close approach: PnP quality drops but HSV streak is reliable.
            if self._streak >= 1 and area_frac > 0.08:
                base = max(base, 0.38)
            if area_frac > 0.20:
                base = max(base, 0.42)
        else:
            base = detection.quality * 0.25

        if geometric and (detection.quality >= 0.10 or area_frac > 0.06):
            self._streak += 1
        else:
            self._streak = max(0, self._streak - 1)

        nx = (detection.centroid_x_px - img_w / 2.0) / (img_w / 2.0)
        self._areas.append(detection.area_px)
        self._nx.append(nx)

        temporal_score = min(1.0, self._streak / max(GATE_TEMPORAL_MIN_STREAK, 1))
        area_score = self._area_consistency()
        motion_score = self._position_consistency()

        gate_confidence = base * (
            0.4 + 0.3 * temporal_score + 0.15 * area_score + 0.15 * motion_score
        )
        gate_confidence = min(1.0, max(0.0, gate_confidence))

        approaching = self._streak >= 3 and area_frac > 0.04
        if approaching:
            gate_confidence = max(
                gate_confidence,
                min(0.65, 0.40 + 0.04 * min(self._streak, 8)),
            )

        ambiguous = (
            not approaching
            and (
                not geometric
                or gate_confidence < GATE_CONFIDENCE_AMBIGUOUS
                or self._streak < GATE_TEMPORAL_MIN_STREAK
            )
        )
        validated = (
            geometric
            and self._streak >= GATE_TEMPORAL_MIN_STREAK
            and gate_confidence >= GATE_CONFIDENCE_MIN_NAV
        )

        return ClassifiedGate(
            detection=detection,
            gate_confidence=gate_confidence,
            temporal_streak=self._streak,
            geometric_valid=geometric,
            ambiguous=ambiguous,
            validated=validated,
        )

    @staticmethod
    def _geometric_valid(det: GateDetection, img_w: int, img_h: int) -> bool:
        if det.area_px < MIN_CONTOUR_AREA_PX:
            return False
        aspect = det.width_px / max(det.height_px, 1.0)
        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            return False
        area_frac = det.area_px / max(img_w * img_h, 1)
        if area_frac > GATE_MAX_AREA_FRAC:
            return False
        if det.reproj_err_px is not None and det.reproj_err_px > 15.0:
            return False
        return True

    def _area_consistency(self) -> float:
        if len(self._areas) < 2:
            return 0.5
        areas = list(self._areas)
        mean_a = sum(areas) / len(areas)
        if mean_a < 1.0:
            return 0.0
        spread = max(abs(a - mean_a) / mean_a for a in areas)
        return max(0.0, 1.0 - spread)

    def _position_consistency(self) -> float:
        if len(self._nx) < 2:
            return 0.5
        nx_vals = list(self._nx)
        spread = max(nx_vals) - min(nx_vals)
        return max(0.0, 1.0 - spread / 0.5)
