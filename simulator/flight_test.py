"""Bounded, instrumented bare-pilot flight test (P0.5 + P3 baseline).

Resets the sim to a clean start, flies the real pilot, logs every gate transition with a
timestamp, and stops at RACE COMPLETE or timeout. Prints how many gates cleared and the lap
time — the baseline the RL deploy gate must beat by >=10%.

Run with the sim up:  make flight-test   (or: uv run python -m simulator.flight_test)
"""

from __future__ import annotations

import time

from simulator.rl_core import frame_looks_corrupted
from simulator.setup import setup_components

SERVER_IP = "127.0.0.1"
SERVER_UDP_PORT = 14550
MAX_LAP_SECONDS = 240.0
STALL_TIMEOUT_S = 60.0  # no gate progress for this long -> give up


def main() -> int:
    boot_ms = int(time.time() * 1000)
    shared: dict = {}
    comps = setup_components(shared, boot_ms, SERVER_IP, SERVER_UDP_PORT)
    controller = comps["controller"]

    print("resetting sim for a clean start...", flush=True)
    controller.send_sim_reset_command()
    time.sleep(3.0)

    # Fast corruption check: after a reset the drone must be near the pad. If it's km away,
    # the sim's coordinate frame is corrupted (a prior crash/collision) and only a full
    # FlightSim.exe restart recovers it — bail immediately instead of flailing for 60s.
    od0 = shared.get("odometry")
    pos0 = (od0["x"], od0["y"], od0["z"]) if od0 else None
    if frame_looks_corrupted(pos0):
        p = tuple(round(c, 1) for c in pos0) if pos0 else None
        print(
            f"FRAME CORRUPTED: drone at {p} after reset (should be ~origin). "
            "Fully restart FlightSim.exe, log in, start a race, then retry.",
            flush=True,
        )
        return 2

    num_gates = None
    last_idx = -1
    t_start = None
    t_last_progress = time.monotonic()
    transitions: list[tuple[int, float]] = []
    last_log = 0.0
    last_collision = None

    try:
        while True:
            controller.update()  # arms + pilot.tick + send + 1/250 sleep
            now = time.monotonic()

            gates = shared.get("track_gates")
            if num_gates is None and gates:
                num_gates = len(gates)

            # Collision events (sim may auto-respawn the drone on a crash).
            col = shared.get("collision")
            if col is not None and col != last_collision:
                last_collision = col
                print(f"  !! COLLISION {col}", flush=True)

            # Periodic trajectory trace (~2 Hz) to see where it goes before a stall/respawn.
            if now - last_log > 0.5:
                last_log = now
                pos = shared.get("odometry")
                tgt = shared.get("active_gate_index")
                if pos is not None:
                    print(
                        f"    [trace] t={(now - t_start) if t_start else 0:5.1f}s idx={tgt} "
                        f"pos=({pos['x']:+.1f},{pos['y']:+.1f},{pos['z']:+.1f})",
                        flush=True,
                    )

            idx = shared.get("active_gate_index")
            if idx is None:
                if now - t_last_progress > STALL_TIMEOUT_S:
                    print(
                        "FAIL: race never started (no active_gate_index).", flush=True
                    )
                    return 1
                continue

            if t_start is None:
                t_start = now  # first frame the race is live

            if idx != last_idx:
                last_idx = idx
                t_last_progress = now
                elapsed = now - t_start
                transitions.append((idx, elapsed))
                pos = shared.get("odometry")
                xyz = (
                    (round(pos["x"], 1), round(pos["y"], 1), round(pos["z"], 1))
                    if pos
                    else None
                )
                print(f"  gate idx -> {idx}  t={elapsed:5.1f}s  pos={xyz}", flush=True)

            ng = num_gates if num_gates is not None else 6
            if idx >= ng:
                lap = now - t_start
                print(f"\nRACE COMPLETE: {ng}/{ng} gates in {lap:.1f}s", flush=True)
                print(f"BASELINE LAP TIME: {lap:.1f}s", flush=True)
                return 0

            if now - t_last_progress > STALL_TIMEOUT_S:
                reached = last_idx
                print(
                    f"\nSTALLED: no progress for {STALL_TIMEOUT_S:.0f}s; reached gate {reached}"
                    f" of {ng}.",
                    flush=True,
                )
                return 1

            if t_start is not None and now - t_start > MAX_LAP_SECONDS:
                print(
                    f"\nTIMEOUT after {MAX_LAP_SECONDS:.0f}s; reached gate {last_idx} of {ng}.",
                    flush=True,
                )
                return 1
    finally:
        comps["mavlink_rx"].get_thread_for_join().join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
