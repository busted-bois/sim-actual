"""PD altitude controller (baseline)."""

from __future__ import annotations

from flightlab.bus import HOVER_THRUST, THRUST_MAX, THRUST_MIN
from flightlab.controllers.leveler import level_rates
from flightlab.protocol import Cmd, Target
from flightlab.state import State

KP_Z = 0.025
KD_Z = 0.10


class PdController:
    name = "pd"

    def __init__(
        self,
        hover: float = HOVER_THRUST,
        kp: float = KP_Z,
        kd: float = KD_Z,
    ) -> None:
        self.hover = hover
        self.kp = kp
        self.kd = kd

    def reset(self, s: State) -> None:
        pass

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        z = s.pos[2]
        vz = s.vel[2]
        z_tgt = target.z if target.z is not None else z
        vz_des = target.vz if target.vz is not None else 0.0
        # NED: z down. Error z - z_tgt > 0 means too low → need more thrust?
        # If z=3 (below), z_tgt=0 (higher up), z-z_tgt=3 → climb needs thrust UP.
        # Climb = negative vz in NED. More thrust → accelerates upward → vz decreases.
        # So thrust = hover + kp*(z - z_tgt) + kd*(vz - vz_des)
        # When too low (z > z_tgt), positive → more thrust. Correct.
        thrust = self.hover + self.kp * (z - z_tgt) + self.kd * (vz - vz_des)
        thrust = float(max(THRUST_MIN, min(THRUST_MAX, thrust)))
        rr, pr = level_rates(
            s.roll,
            s.pitch,
            target.lean_roll,
            target.lean_pitch,
            soft=(s.pose_source == "estimator"),
        )
        return Cmd(rr, pr, 0.0, thrust)
