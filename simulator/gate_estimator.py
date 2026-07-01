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

# Default image dimensions (pixels).
_IMG_W = 640.0
_IMG_H = 360.0
_FX_DEFAULT = 320.0


def _gate_has_valid_pose(gate: TrackGate | None) -> bool:
    if gate is None:
        return False
    px, py, pz = gate.pos_ned
    return abs(px) + abs(py) + abs(pz) > 0.01


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
        img_w: float = _IMG_W,
        img_h: float = _IMG_H,
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

        # 2. Self-calibration: collect focal samples from track ground-truth (VQ1 only).
        if (
            gate is not None
            and _gate_has_valid_pose(gate)
            and drone_state.has_position
            and detection.width_px > 0
        ):
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

        # 3. Range estimation (focal locked, or VQ2 pixel-width heuristic).
        if self.focal_px is not None:
            real_width = gate.width_m if gate is not None and gate.width_m > 0 else REFERENCE_GATE_WIDTH_M
            range_m = real_width * self.focal_px / max(detection.width_px, 1.0)
        elif detection.width_px > 0:
            range_m = REFERENCE_GATE_WIDTH_M * _FX_DEFAULT / max(detection.width_px, 1.0)
        else:
            range_m = None

        # 4. Bearing from pixel offset.
        offset_px = detection.centroid_x_px - img_w / 2.0
        focal = self.focal_px if self.focal_px is not None else _FX_DEFAULT
        bearing = pixel_offset_to_bearing(offset_px, focal)

        # 5. Lateral offset (geometric; requires a real range).
        if range_m is not None:
            lateral_offset_m = math.tan(bearing) * range_m
        else:
            lateral_offset_m = None

        # 6. Source and confidence.
        if self.focal_px is not None and _gate_has_valid_pose(gate):
            source = "track"
            confidence = 0.8
        elif range_m is not None:
            source = "pixel-ratio"
            confidence = 0.5
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
