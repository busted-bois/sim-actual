"""Log race_status transitions during arm -> countdown -> GO for timing calibration."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymavlink import mavutil

from simulator.preflight import is_restart_arm_context, latch_race_go_boot_ms
from simulator.mavlink_tx import GCS_HEARTBEAT_INTERVAL_S, send_gcs_heartbeat
from simulator.setup import HEARTBEAT_TIMEOUT_S

LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 14550
PROBE_TIMEOUT_S = 120.0


def _race_snapshot(data: dict) -> tuple:
    race = data.get("race_status") or {}
    return (
        race.get("sim_boot_time_ms"),
        race.get("race_start_boot_time_ms"),
        race.get("active_gate_index"),
        race.get("last_gate_race_time"),
    )


def main() -> int:
    print(f"Race timing probe on udpin:{LISTEN_IP}:{LISTEN_PORT}", flush=True)
    print(
        "Arm the drone in sim (or run make sim). Logs race_status changes.",
        flush=True,
    )

    conn = mavutil.mavlink_connection(f"udpin:{LISTEN_IP}:{LISTEN_PORT}")
    shared_data: dict = {}

    from simulator.mavlink_rx import MAVLinkRX

    rx = MAVLinkRX.create_mavlink_rx(conn, shared_data)

    deadline = time.monotonic() + HEARTBEAT_TIMEOUT_S
    next_tx = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_tx:
            send_gcs_heartbeat(conn)
            next_tx = now + GCS_HEARTBEAT_INTERVAL_S
        if shared_data.get("race_status"):
            break
        time.sleep(0.02)

    if not shared_data.get("race_status"):
        print("No race_status received. Start a flight session first.", flush=True)
        rx.get_thread_for_join().join(timeout=1.0)
        return 1

    last = None
    armed_at_mono = None
    armed_sim_boot = None
    go_boot_ms = None
    latch_branch = None
    last_race_start = -1
    start_mono = time.monotonic()

    print("time_s sim_boot race_start gate_idx gate_race go_boot branch", flush=True)

    while time.monotonic() - start_mono < PROBE_TIMEOUT_S:
        snap = _race_snapshot(shared_data)
        if snap != last:
            elapsed = time.monotonic() - start_mono
            sim_boot, race_start, gate_idx, gate_race = snap

            if shared_data.get("armed") and armed_at_mono is None:
                armed_at_mono = time.monotonic()
                armed_sim_boot = sim_boot

            is_restart = (
                armed_sim_boot is not None
                and is_restart_arm_context(armed_sim_boot)
                and last_race_start >= 0
            )
            if (
                go_boot_ms is None
                and last_race_start < 0
                and race_start is not None
                and race_start >= 0
                and sim_boot is not None
                and (armed_sim_boot is None or sim_boot > armed_sim_boot)
            ):
                go_boot_ms, latch_branch = latch_race_go_boot_ms(
                    sim_boot, race_start, is_restart=is_restart
                )

            if race_start is not None and race_start >= 0:
                last_race_start = race_start

            since_arm = (
                f"{(time.monotonic() - armed_at_mono) * 1000:.0f}ms"
                if armed_at_mono is not None
                else "-"
            )
            print(
                f"{elapsed:7.2f} {sim_boot} {race_start} {gate_idx} {gate_race} "
                f"{go_boot_ms} {latch_branch} since_arm={since_arm}",
                flush=True,
            )
            last = snap

        time.sleep(0.004)

    rx.get_thread_for_join().join(timeout=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
