import math

import numpy as np
from dataclasses import dataclass

from simulator.config import (
    FOCAL_LENGTH_PX_INIT,
    SELF_CAL_SAMPLES,
    REFERENCE_GATE_WIDTH_M,
    GateDetection,
    DroneState,
    TrackGate,
)
from simulator.transforms import estimate_focal_from_track, pixel_offset_to_bearing

_IMG_W = 640.0


@dataclass
class GateEstimate:
    bearing_rad: float
    range_m: float | None
    lateral_offset_m: float | None
    confidence: float
    source: str


class GateEstimator:
    def __init__(self):
        self.focal_px: float | None = FOCAL_LENGTH_PX_INIT
        self._cal_samples: list[float] = []

    def update(
        self,
        detection: GateDetection | None,
        drone_state: DroneState,
        gates: list[TrackGate],
        active_gate_index: int,
    ) -> GateEstimate:
        if detection is None:
            return GateEstimate(0.0, None, None, 0.0, "none")

        gate = gates[active_gate_index] if active_gate_index < len(gates) else None

        if gate is not None and drone_state.has_position and detection.width_px > 0:
            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(drone_state.pos_ned, gate.pos_ned))
            )
            if distance > 0 and gate.width_m > 0:
                focal = estimate_focal_from_track(
                    detection.width_px, gate.width_m, distance
                )
                self._cal_samples.append(focal)
                if len(self._cal_samples) >= SELF_CAL_SAMPLES:
                    self.focal_px = float(np.median(self._cal_samples))

        if self.focal_px is not None:
            real_width = gate.width_m if gate is not None else REFERENCE_GATE_WIDTH_M
            range_m = real_width * self.focal_px / max(detection.width_px, 1.0)
        else:
            range_m = None

        offset_px = detection.centroid_x_px - _IMG_W / 2.0
        if self.focal_px is not None:
            bearing = pixel_offset_to_bearing(offset_px, self.focal_px)
        else:
            bearing = offset_px / _IMG_W

        if range_m is not None:
            lateral_offset_m = math.tan(bearing) * range_m
        else:
            lateral_offset_m = None

        if self.focal_px is not None:
            return GateEstimate(bearing, range_m, lateral_offset_m, 0.8, "track")
        return GateEstimate(bearing, range_m, lateral_offset_m, 0.3, "pixel-ratio")
