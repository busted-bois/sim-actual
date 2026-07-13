"""Shared types for the attitude harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol


class Cmd(NamedTuple):
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float


@dataclass
class State:
    t_mono: float
    armed: bool
    roll: float
    pitch: float
    yaw: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    pos_ned: tuple[float, float, float]
    vel_ned: tuple[float, float, float]
    gyro: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    pose_age_s: float
    gravity_body: tuple[float, float, float] | None = None
    pose_source: str = "unknown"
    alt_trusted: bool = False


@dataclass
class Target:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    z: float = -3.0


class Controller(Protocol):
    name: str

    def reset(self, s: State) -> None: ...

    def update(self, s: State, target: Target, dt: float) -> Cmd: ...
