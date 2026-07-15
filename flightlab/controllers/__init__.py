"""Controller registry."""

from __future__ import annotations

from flightlab.controllers.baro_hold import BaroHoldController
from flightlab.controllers.pd import PdController
from flightlab.controllers.pid import PidController
from flightlab.controllers.pid_tilt import PidTiltController
from flightlab.controllers.pid_tilt_filt import PidTiltFiltController
from flightlab.protocol import Controller

CONTROLLERS: dict[str, type] = {
    "baro_hold": BaroHoldController,
    "pd": PdController,
    "pid": PidController,
    "pid_tilt": PidTiltController,
    "pid_tilt_filt": PidTiltFiltController,
}


def make_controller(name: str, **kwargs) -> Controller:
    if name not in CONTROLLERS:
        known = ", ".join(sorted(CONTROLLERS))
        raise KeyError(f"unknown method {name!r}; choose one of: {known}")
    return CONTROLLERS[name](**kwargs)


def list_methods() -> list[str]:
    return sorted(CONTROLLERS)
