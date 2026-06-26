from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RaceConfig:
    # === Qualifier baseline (the proven 6/6 @ ~24s config). Reset to this after an
    # incremental-tuning drift made our sim hard-reset at gate 0. Tune from here in
    # SMALL, single-knob steps, reading the PLANE-crossing diagnostic each time. ===
    name: str = "measured"
    kph_ct: float = 1.4
    kph_vel: float = 2.2
    vh_max: float = 13.0
    ah_max: float = 16.0
    kpz_pos: float = 1.0
    kpz_vel: float = 3.0
    kp_att: float = 1.3
    max_rate: float = 2.0
    look_ahead: float = 4.0
    alt_bias: float = 0.6
    v_max: float = 9.0
    v_min: float = 2.5
    v_start: float = 1.0
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
