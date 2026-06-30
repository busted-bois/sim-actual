from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RaceConfig:
    name: str = "measured"
    kph_ct: float = 0.6
    kph_vel: float = 1.6
    vh_max: float = 4.0
    ah_max: float = 10.0
    kpz_pos: float = 1.0
    kpz_vel: float = 3.0
    kp_att: float = 1.0
    max_rate: float = 1.5
    look_ahead: float = 4.0
    alt_bias: float = 0.6
    v_max: float = 3.0
    v_min: float = 1.0
    v_start: float = 0.5
    a_lat: float = 5.5
    a_lon: float = 6.0
    a_brk: float = 5.0
    climb_max: float = 2.3
    curv_ff: float = 0.0
    tilt_deg: float = 35.0

    @property
    def tilt_rad(self) -> float:
        return math.radians(self.tilt_deg)


MEASURED = RaceConfig()
