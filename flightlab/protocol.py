"""Shared controller protocol — keep exact shape so methods swap cleanly."""

from __future__ import annotations

from typing import NamedTuple, Protocol

from flightlab.state import State


class Cmd(NamedTuple):
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float


class Target(NamedTuple):
    """Vertical setpoint. NED: z down-positive; vz_des negative = climb."""

    z: float | None = None
    vz: float | None = None
    lean_roll: float = 0.0
    lean_pitch: float = 0.0


class Controller(Protocol):
    name: str

    def reset(self, s: State) -> None: ...

    def update(self, s: State, target: Target, dt: float) -> Cmd: ...
