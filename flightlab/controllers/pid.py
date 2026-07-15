"""PID altitude controller with anti-windup integrator."""

from __future__ import annotations

from flightlab.bus import HOVER_THRUST, THRUST_MAX, THRUST_MIN
from flightlab.controllers.leveler import level_rates
from flightlab.protocol import Cmd, Target
from flightlab.state import State

KP_Z = 0.025
KD_Z = 0.10
KI_Z = 0.008
I_LIMIT = 0.08


class PidController:
    name = "pid"

    def __init__(
        self,
        hover: float = HOVER_THRUST,
        kp: float = KP_Z,
        kd: float = KD_Z,
        ki: float = KI_Z,
    ) -> None:
        self.hover = hover
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self._i = 0.0

    def reset(self, s: State) -> None:
        self._i = 0.0

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        z = s.pos[2]
        vz = s.vel[2]
        z_tgt = target.z if target.z is not None else z
        vz_des = target.vz if target.vz is not None else 0.0
        err = z - z_tgt
        # Provisional thrust without I for anti-windup check
        u_pd = self.hover + self.kp * err + self.kd * (vz - vz_des)
        saturated = u_pd + self._i > THRUST_MAX or u_pd + self._i < THRUST_MIN
        # Anti-windup: only integrate when not pushing further into sat,
        # or when error drives us out of saturation.
        if (
            not saturated
            or (err > 0 and u_pd + self._i < THRUST_MIN)
            or (err < 0 and u_pd + self._i > THRUST_MAX)
        ):
            self._i += self.ki * err * max(dt, 0.0)
            self._i = float(max(-I_LIMIT, min(I_LIMIT, self._i)))
        thrust = float(max(THRUST_MIN, min(THRUST_MAX, u_pd + self._i)))
        rr, pr = level_rates(
            s.roll,
            s.pitch,
            target.lean_roll,
            target.lean_pitch,
            soft=(s.pose_source == "estimator"),
        )
        return Cmd(rr, pr, 0.0, thrust)
