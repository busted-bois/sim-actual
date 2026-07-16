"""Barometric / Kalman altitude hold.

Law (UP-positive altitude):
    Vz_cmd = kp * (z_up_tgt - zhat_up) - kd * (vzhat_up - vz_des_up)
    T_vert = hover + kt * Vz_cmd
    thrust  = T_vert / (cos(phi)*cos(theta))   # tilt compensate

zhat/vzhat come from the VIO/Kalman stack (StateEstimator). When sim baro
(pressure_alt) is finite it is fused into the EKF; on many TRAINING builds
baro is NaN and z rides the thrust model.
"""

from __future__ import annotations

from flightlab.bus import HOVER_THRUST, THRUST_MAX, THRUST_MIN
from flightlab.controllers.leveler import level_rates, tilt_comp_factor
from flightlab.protocol import Cmd, Target
from flightlab.state import State

KP_Z = 1.0
KD_Z = 0.4
KT = 0.05  # thrust units per (m/s) Vz_cmd


class BaroHoldController:
    name = "baro_hold"

    def __init__(
        self,
        hover: float = HOVER_THRUST,
        kp: float = KP_Z,
        kd: float = KD_Z,
        kt: float = KT,
    ) -> None:
        self.hover = hover
        self.kp = kp
        self.kd = kd
        self.kt = kt
        self.last_vz_cmd = 0.0
        self.last_tilt_comp = 1.0

    def reset(self, s: State) -> None:
        self.last_vz_cmd = 0.0
        self.last_tilt_comp = 1.0

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        z_tgt = target.z if target.z is not None else s.zhat
        zhat_up = -s.zhat
        vzhat_up = -s.vzhat
        zt_up = -z_tgt
        vz_des_up = -float(target.vz) if target.vz is not None else 0.0
        # Hold: Vz = kp(ztarget - zhat) - kd(vzhat); with vz_des for rate segs.
        vz_cmd = self.kp * (zt_up - zhat_up) - self.kd * (vzhat_up - vz_des_up)
        self.last_vz_cmd = float(vz_cmd)
        t_vert = self.hover + self.kt * vz_cmd
        self.last_tilt_comp = float(tilt_comp_factor(s.roll, s.pitch))
        thrust = float(max(THRUST_MIN, min(THRUST_MAX, t_vert * self.last_tilt_comp)))
        rr, pr = level_rates(
            s.roll,
            s.pitch,
            target.lean_roll,
            target.lean_pitch,
            soft=(s.pose_source == "estimator"),
        )
        return Cmd(rr, pr, 0.0, thrust)
