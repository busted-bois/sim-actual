"""Throttled debug logging for auto-flight preflight and control."""

from __future__ import annotations

import os
import time

_AUTO_TRUE = frozenset({"1", "true", "yes"})
_last_log: dict[str, float] = {}


def flight_debug_enabled() -> bool:
    env = os.environ
    return (
        env.get("AUTO_FLIGHT", "").strip().lower() in _AUTO_TRUE
        or env.get("AUTO_FLIGHT_DEBUG", "").strip().lower() in _AUTO_TRUE
    )


def dbg(tag: str, msg: str, throttle_s: float = 0.5) -> None:
    if not flight_debug_enabled():
        return
    now = time.monotonic()
    key = tag
    last = _last_log.get(key, 0.0)
    if now - last < throttle_s:
        return
    _last_log[key] = now
    print(f"[FLIGHT_DBG] {tag} {msg}", flush=True)


def dbg_now(tag: str, msg: str) -> None:
    """Unthrottled one-shot debug line."""
    if not flight_debug_enabled():
        return
    print(f"[FLIGHT_DBG] {tag} {msg}", flush=True)


def motion_snapshot(data: dict) -> str:
    from simulator.preflight import latch_race_go_boot_ms

    race = data.get("race_status") or {}
    sim_boot = race.get("sim_boot_time_ms", 0)
    race_start = race.get("race_start_boot_time_ms", -1)
    go_boot, branch = latch_race_go_boot_ms(sim_boot, race_start)
    vel = data.get("vel_ned")
    pos = data.get("pos_ned")
    vel_s = (
        f"({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f})"
        if vel is not None
        else "?"
    )
    pos_s = (
        f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
        if pos is not None
        else "?"
    )
    return (
        f"armed={data.get('armed', False)} "
        f"sim_boot={sim_boot} race_start={race_start} "
        f"go_boot={go_boot} branch={branch} "
        f"vel_ned={vel_s} pos_ned={pos_s}"
    )
