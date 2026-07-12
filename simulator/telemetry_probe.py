"""Passive telemetry probe — what does the sim ACTUALLY send right now?

Listens on the MAVLink port for N seconds (no arming, no commands, safe during
a live race) and reports per-message-type rates plus a sanity dump of one
HIGHRES_IMU sample. Run it in Training and then in Qualification (VQ2) to see
exactly which messages the event block removes and what survives — especially
whether the race-status half of ENCAPSULATED_DATA is still there, the IMU rate
(sets the estimator dt), and the IMU conventions (gravity sign, mag magnitude,
pressure_alt plausibility).

    uv run -m simulator.telemetry_probe            (or: make probe)
"""

import argparse
import math
import time
from collections import Counter

from pymavlink import mavutil

from simulator.mavlink_rx import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
)

# Messages the VQ2 block is expected to remove / keep.
BLOCKED_EXPECTED = ("ODOMETRY", "ATTITUDE", "LOCAL_POSITION_NED")
KEPT_EXPECTED = ("HIGHRES_IMU", "HEARTBEAT", "TIMESYNC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    args = ap.parse_args()

    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    print(
        f"[probe] listening udpin:{args.ip}:{args.port} for {args.seconds:.0f}s "
        "(passive, no commands)...",
        flush=True,
    )

    counts: Counter = Counter()
    encap: Counter = Counter()
    imu_sample = None
    imu_t_first = imu_t_last = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        msg = conn.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            continue
        counts[mtype] += 1
        if mtype == "HIGHRES_IMU":
            if imu_sample is None:
                imu_sample = msg
                imu_t_first = msg.time_usec
            imu_t_last = msg.time_usec
        elif mtype == "ENCAPSULATED_DATA":
            encap[bytes(msg.data)[0]] += 1

    dur = time.monotonic() - t0
    if not counts:
        print("[probe] NOTHING received — is the sim running / port right?")
        return

    print(f"\n[probe] message rates over {dur:.1f}s:")
    for mtype, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {mtype:<28} {n:>6}  ({n / dur:6.1f} Hz)")
    if encap:
        race = encap.get(ENCAPSULATED_RACE_STATUS_MSG_ID, 0)
        track = encap.get(ENCAPSULATED_TRACK_INFO_MSG_ID, 0)
        print(f"  ENCAPSULATED breakdown: race_status={race} track_info={track}")

    print("\n[probe] block profile:")
    for m in BLOCKED_EXPECTED + KEPT_EXPECTED:
        state = "PRESENT" if counts.get(m) else "absent"
        print(f"  {m:<28} {state}")

    if imu_sample is not None:
        s = imu_sample
        anorm = math.sqrt(s.xacc**2 + s.yacc**2 + s.zacc**2)
        mnorm = math.sqrt(s.xmag**2 + s.ymag**2 + s.zmag**2)
        n_imu = counts["HIGHRES_IMU"]
        rate = (
            (n_imu - 1) / ((imu_t_last - imu_t_first) * 1e-6)
            if n_imu > 1 and imu_t_last > imu_t_first
            else 0.0
        )
        print("\n[probe] HIGHRES_IMU sample (first received):")
        print(f"  accel=({s.xacc:+.3f},{s.yacc:+.3f},{s.zacc:+.3f}) |a|={anorm:.3f}")
        print("  -> on ground expect |a|~9.81; az sign sets estimator accel_sign")
        print(f"  gyro =({s.xgyro:+.4f},{s.ygyro:+.4f},{s.zgyro:+.4f})")
        print(f"  mag  =({s.xmag:+.4f},{s.ymag:+.4f},{s.zmag:+.4f}) |m|={mnorm:.4f}")
        print(
            f"  abs_pressure={s.abs_pressure:.2f} hPa  "
            f"pressure_alt={s.pressure_alt:.2f}  temp={s.temperature:.1f}C"
        )
        print(f"  sensor-timestamp rate: {rate:.1f} Hz")


if __name__ == "__main__":
    main()
