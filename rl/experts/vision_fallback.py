"""Camera-native expert fallback for deploy.

Uses GateEstimateSmoother, VisionVelocityTracker, and compute_guidance
to produce rate+thrust commands when the position EKF is unhealthy.
Does NOT use GPExpert or synthetic pose-derived vision.
Does NOT fabricate translational velocity from gyro.
"""

from __future__ import annotations

import math

import numpy as np

from rl.experts.fly2_course import HOVER_T as LIVE_HOVER_THRUST
from rl.experts.gp_expert import ATT_RATE_GAIN
from simulator.gp_pilot import (
    _fresh_hold_state,
    compute_guidance,
)
from simulator.gp_vision import GateEstimateSmoother, VisionVelocityTracker

# Rate conversion mirrors GPExpert exactly: -roll, +pitch, -yaw with gain.
LIVE_RATE_CLIP = 0.60  # rad/s — above fly2's 0.30 so policy can bank
LIVE_THRUST_MIN = 0.12
LIVE_THRUST_MAX = 0.55


def guidance_to_rate_cmds(
    roll_cmd_deg: float,
    pitch_cmd_deg: float,
    yaw_cmd_deg: float,
    thrust: float,
) -> tuple[float, float, float, float]:
    """Convert compute_guidance degree commands to the deploy rate convention.

    Mirrors GPExpert exactly: compute_guidance emits live sim's inverted
    roll/yaw signs (gp_pilot KR=KY=-1). The deploy loop sends rad/s rate
    commands, so flip roll/yaw back and apply ATT_RATE_GAIN, then clip.
    """
    deg2rad = math.pi / 180.0
    g = ATT_RATE_GAIN
    roll_rate = float(
        np.clip(-roll_cmd_deg * deg2rad * g, -LIVE_RATE_CLIP, LIVE_RATE_CLIP)
    )
    pitch_rate = float(
        np.clip(pitch_cmd_deg * deg2rad * g, -LIVE_RATE_CLIP, LIVE_RATE_CLIP)
    )
    yaw_rate = float(
        np.clip(-yaw_cmd_deg * deg2rad * g, -LIVE_RATE_CLIP, LIVE_RATE_CLIP)
    )
    return (
        roll_rate,
        pitch_rate,
        yaw_rate,
        float(np.clip(thrust, LIVE_THRUST_MIN, LIVE_THRUST_MAX)),
    )


def euler_deg(q: np.ndarray) -> tuple[float, float]:
    """Roll/pitch (deg) from quaternion (w,x,y,z), aerospace convention."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


class FallbackBrain:
    """Camera-native expert fallback: socket-free decision logic.

    Zero degraded translational priors (vY=0, vD=0, vX=NaN).
    VisionVelocityTracker output passed through so compute_guidance
    can supply optical damping. Gyro must not represent velocity.
    """

    def __init__(self) -> None:
        self.smoother = GateEstimateSmoother()
        self.vel_tracker = VisionVelocityTracker()
        self.hold = _fresh_hold_state()
        self._vision: dict | None = None
        self._vision_vel: dict | None = None
        self._gate_quat: np.ndarray | None = None

    def reset(self) -> None:
        self.smoother.reset()
        self.vel_tracker.reset()
        self.hold = _fresh_hold_state()
        self._vision = None
        self._vision_vel = None
        self._gate_quat = None

    def set_gate_context(self, gate_quat: np.ndarray | None) -> None:
        """Set the active gate quaternion for doffset/CTE (from deploy gate_map)."""
        self._gate_quat = gate_quat

    def update(
        self,
        data: dict,
        quat: np.ndarray,
        dt: float,
    ) -> tuple[float, float, float, float]:
        try:
            vision = self.smoother.update(data)
        except (KeyError, TypeError, ValueError):
            vision = None
        self._vision = vision
        vision_vel = self.vel_tracker.update(vision)
        self._vision_vel = vision_vel

        if vision is None:
            return (0.0, 0.0, 0.0, LIVE_HOVER_THRUST)

        roll_deg, pitch_deg = euler_deg(quat)
        # Zero degraded translational priors — gyro must not represent velocity.
        vY = 0.0
        vD = 0.0

        roll_cmd_deg, pitch_cmd_deg, yaw_cmd_deg, thrust, _dbg = compute_guidance(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            quat=quat,
            vY=vY,
            vD=vD,
            vision=vision,
            vision_vel=vision_vel,
            state=self.hold,
            hover_thrust=LIVE_HOVER_THRUST,
            vX=float("nan"),  # no forward speed estimate
            dt=dt,
            gate_quat=self._gate_quat,
        )
        return guidance_to_rate_cmds(roll_cmd_deg, pitch_cmd_deg, yaw_cmd_deg, thrust)
