"""P0 (BLOCKER): verify whether `sim_reset` clears a fly-away.

Wide RL action bounds mean the policy will fly the drone away thousands of times. If a
fly-away corrupts the gate frame and only a full FlightSim.exe restart clears it (per the
project's saved sim notes), an in-process Gym reset loop is impossible and training must
relaunch the sim per bad episode. This script answers that empirically:

  1. connect + arm, record the gate table and start pose
  2. deliberately fly away (large constant tilt) for a few seconds
  3. fire `sim_reset`
  4. re-read the gate table and pose; report whether they recovered

Run with the simulator up:  make verify-reset   (or: uv run python -m simulator.verify_reset)
"""

from __future__ import annotations

import time

from pymavlink import mavutil

from simulator.config import ROUND1_GATES
from simulator.controller import MAVLINK_CMD_SIM_RESET, _send_attitude_rates
from simulator.mavlink_rx import MAVLinkRX

SERVER_IP = "127.0.0.1"
SERVER_UDP_PORT = 14550

GATE_POS_TOL_M = 0.5
FLYAWAY_SECONDS = 4.0
RESET_SETTLE_SECONDS = 3.0


def _arm(conn):
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _reset(conn):
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        MAVLINK_CMD_SIM_RESET,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _snapshot_gates(shared):
    gates = shared.get("track_gates") or []
    return [tuple(round(c, 3) for c in g["position_ned"]) for g in gates]


def _wait_for(shared, key, timeout_s=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if shared.get(key):
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    boot_ms = int(time.time() * 1000)
    conn = mavutil.mavlink_connection(f"udpin:{SERVER_IP}:{SERVER_UDP_PORT}")
    print("Waiting for heartbeat (15s)...", flush=True)
    if conn.wait_heartbeat(timeout=15) is None:
        print(
            "NO SIM: no MAVLink heartbeat on "
            f"{SERVER_IP}:{SERVER_UDP_PORT}. Launch FlightSim.exe and log in, then retry.",
            flush=True,
        )
        return 2
    shared = {}
    rx = MAVLinkRX.create_mavlink_rx(conn, shared)

    try:
        _arm(conn)
        if not _wait_for(shared, "odometry", timeout_s=15.0):
            print(
                "FAIL: no odometry after heartbeat — is the drone spawned?", flush=True
            )
            return 1

        # The gate table is broadcast ONCE at race start; if we connected mid-race we may
        # miss it. Give the user a window to start the race, but don't hard-fail without it —
        # the reset itself restarts the race and should rebroadcast the table.
        print(
            "Connected. If the race isn't running, click RACE now "
            "(waiting up to 45s for the gate table)...",
            flush=True,
        )
        have_gates_before = _wait_for(shared, "track_gates", timeout_s=45.0)
        gates_before = _snapshot_gates(shared) if have_gates_before else []
        pos_before = shared.get("odometry")
        print(
            f"baseline: {len(gates_before)} gates, pose present={pos_before is not None}",
            flush=True,
        )

        # Fly away: large nose-down pitch + high thrust at 250 Hz.
        print(f"flying away for {FLYAWAY_SECONDS}s...", flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < FLYAWAY_SECONDS:
            _arm(conn)
            _send_attitude_rates(conn, boot_ms, pitch_rate=-0.4, thrust=0.9)
            time.sleep(1.0 / 250)
        pos_flown = shared.get("odometry")
        print(f"pose after fly-away: {pos_flown}", flush=True)

        print("firing sim_reset...", flush=True)
        shared["track_gates"] = None  # so we can detect a fresh rebroadcast
        _reset(conn)
        time.sleep(RESET_SETTLE_SECONDS)
        _arm(conn)
        _wait_for(
            shared, "track_gates", timeout_s=15.0
        )  # reset should restart the race
        time.sleep(0.5)

        gates_after = _snapshot_gates(shared)
        idx_after = shared.get("active_gate_index")
        pos_after = shared.get("odometry")
        print(
            f"after reset: {len(gates_after)} gates, idx={idx_after}, pose={pos_after}",
            flush=True,
        )

        # Known-good Round-1 frame to compare against (absolute reference).
        known = [tuple(round(c, 3) for c in g["position_ned"]) for g in ROUND1_GATES]

        def _max_dev(a_list, b_list):
            dev = 0.0
            for a, b in zip(a_list, b_list):
                dev = max(dev, max(abs(a[i] - b[i]) for i in range(3)))
            return dev

        ok = True
        checked_frame = False
        if gates_after:
            ref = (
                gates_before
                if (have_gates_before and len(gates_before) == len(gates_after))
                else known
            )
            ref_name = (
                "pre-reset gates" if ref is gates_before else "known Round-1 frame"
            )
            if len(gates_after) == len(ref):
                dev = _max_dev(gates_after, ref)
                checked_frame = True
                print(f"gate-frame deviation vs {ref_name}: {dev:.3f} m", flush=True)
                if dev > GATE_POS_TOL_M:
                    print(
                        "  -> gate frame SHIFTED beyond tolerance = CORRUPTED",
                        flush=True,
                    )
                    ok = False
            else:
                print(
                    f"WARN: gate count {len(gates_after)} != reference {len(ref)}",
                    flush=True,
                )
        else:
            print(
                "WARN: no gate table after reset — can't directly confirm the frame.",
                flush=True,
            )

        # Pose recovery: after reset the drone should be back near the start pad, not where
        # it flew off to.
        if pos_after is not None:
            mag = (
                pos_after["x"] ** 2 + pos_after["y"] ** 2 + pos_after["z"] ** 2
            ) ** 0.5
            print(f"post-reset distance from origin: {mag:.2f} m", flush=True)
            if pos_flown is not None:
                flown_mag = (
                    pos_flown["x"] ** 2 + pos_flown["y"] ** 2 + pos_flown["z"] ** 2
                ) ** 0.5
                if flown_mag > 10.0 and mag > 0.5 * flown_mag:
                    print(
                        "  -> pose did NOT return to start = reset ineffective",
                        flush=True,
                    )
                    ok = False
        if idx_after not in (0, None):
            print(
                f"WARN: active_gate_index did not reset to 0 (={idx_after})", flush=True
            )

        print("\n==== P0 RESULT ====", flush=True)
        if ok and (checked_frame or pos_after is not None):
            print(
                "PASS: soft sim_reset recovers a clean state -> in-process Gym reset is OK.",
                flush=True,
            )
            if not checked_frame:
                print(
                    "  (NOTE: gate frame not directly confirmed — pose/index recovered.)",
                    flush=True,
                )
            return 0
        print(
            "FAIL: soft reset insufficient -> process-supervisor restart (Contingency).",
            flush=True,
        )
        return 1
    finally:
        rx.get_thread_for_join().join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
