"""Gate bearing/range estimator with self-calibrating focal length.

Estimates the bearing, range, and lateral offset to the active gate from a
gate detection. Self-calibrates the focal length (in pixels) from track
ground-truth during the first few samples, then falls back to a coarse
pixel-ratio estimate when no focal estimate is available.
"""

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

# Default image dimensions (pixels). The estimator does not receive the actual
# frame size, so it assumes a standard 640x480 image to convert centroid pixel
# position into an offset from the image center.
_IMG_W = 640.0
_IMG_H = 480.0


@dataclass
class GateEstimate:
    bearing_rad: float
    range_m: float | None
    lateral_offset_m: float | None
    confidence: float  # 0.0 to 1.0
    source: str  # "track", "pixel-ratio", "none"


class GateEstimator:
    """Self-calibrating focal-length + pixel-ratio gate estimator.

    Accumulates focal-length samples from track ground-truth, then locks in the
    median once enough samples are collected. Once locked, produces geometric
    bearing/range/lateral-offset estimates. Before locking, produces a coarse
    pixel-ratio bearing only (no range).
    """

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
        # 1. No detection -> nothing to estimate.
        if detection is None:
            return GateEstimate(
                bearing_rad=0.0,
                range_m=None,
                lateral_offset_m=None,
                confidence=0.0,
                source="none",
            )

        # Resolve the active track gate (may be None if index is out of range).
        gate = gates[active_gate_index] if active_gate_index < len(gates) else None

        # 2. Self-calibration: collect focal samples from track ground-truth.
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

        # 3. Range estimation (only when a focal estimate is locked).
        if self.focal_px is not None:
            real_width = gate.width_m if gate is not None else REFERENCE_GATE_WIDTH_M
            range_m = real_width * self.focal_px / max(detection.width_px, 1.0)
        else:
            range_m = None

        # 4. Bearing from pixel offset.
        offset_px = detection.centroid_x_px - _IMG_W / 2.0
        if self.focal_px is not None:
            bearing = pixel_offset_to_bearing(offset_px, self.focal_px)
        else:
            # Pixel-ratio fallback: normalized offset, NOT radians.
            bearing = offset_px / _IMG_W

        # 5. Lateral offset (geometric; requires a real range).
        if range_m is not None:
            lateral_offset_m = math.tan(bearing) * range_m
        else:
            lateral_offset_m = None

        # 6. Source and confidence.
        if self.focal_px is not None:
            source = "track"
            confidence = 0.8
        else:
            source = "pixel-ratio"
            confidence = 0.3

        return GateEstimate(
            bearing_rad=bearing,
            range_m=range_m,
            lateral_offset_m=lateral_offset_m,
            confidence=confidence,
            source=source,
        )
