"""Telemetry probe — what does the sim ACTUALLY send right now?

Reports per-message-type rates plus a sanity dump of one HIGHRES_IMU sample.
Run it in Training and then in Qualification (VQ2) to see exactly which
messages the event block removes and what survives — especially whether the
race-status half of ENCAPSULATED_DATA is still there, the IMU rate (sets the
estimator dt), and the IMU conventions (gravity sign, mag magnitude,
pressure_alt plausibility).

ACTIVE by default. The sim only streams telemetry to a client it has seen a
ground-station heartbeat from, and HIGHRES_IMU additionally needs an explicit
SET_MESSAGE_INTERVAL (simulator/setup.py:27-31,125-127). A purely passive
listener therefore reports HIGHRES_IMU *and* TIMESYNC as "absent" even in
Training, which says nothing about the VQ2 block. So this probe performs the
same handshake the flight client does, then measures. It never arms and never
sends a setpoint — safe to run during a live race.

Use --passive to reproduce the old listen-only behaviour for comparison.

Deliberately depends on pymavlink only (not simulator.setup), so it stays
light enough to run mid-race: importing the flight path would pull in VisionRX
-> GatePoseRunner -> torch/ultralytics and take a GPU slice.

    uv run -m simulator.telemetry_probe            (or: make probe)
"""

import argparse
import math
import threading
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

# Streams we explicitly request so that "absent" means blocked, not unasked-for.
# ODOMETRY = 331 is MAVLink-2-only and absent from the v1 dialect's constants.
_MSG_ID_ATTITUDE = 30
_MSG_ID_LOCAL_POSITION_NED = 32
_MSG_ID_HIGHRES_IMU = 105
_MSG_ID_ODOMETRY = 331
_MAV_CMD_SET_MESSAGE_INTERVAL = 511

_REQUESTED = (
    _MSG_ID_HIGHRES_IMU,
    _MSG_ID_ATTITUDE,
    _MSG_ID_LOCAL_POSITION_NED,
    _MSG_ID_ODOMETRY,
)

GCS_HEARTBEAT_INTERVAL_S = 1.0
HEARTBEAT_TIMEOUT_S = 10.0
REQUEST_INTERVAL_US = 10_000  # 100 Hz


def _send_gcs_heartbeat(conn):
    """Announce ourselves as a GCS so the sim starts streaming to us."""
    conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )


def _wait_for_sim_heartbeat(conn, timeout_s=HEARTBEAT_TIMEOUT_S):
    """Send GCS heartbeats until the sim answers. Returns its HEARTBEAT or None."""
    deadline = time.monotonic() + timeout_s
    next_tx = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_tx:
            _send_gcs_heartbeat(conn)
            next_tx = now + GCS_HEARTBEAT_INTERVAL_S
        msg = conn.recv_match(type="HEARTBEAT", blocking=False)
        if msg is not None:
            return msg
        time.sleep(0.02)
    return None


def _start_gcs_heartbeat_thread(conn):
    """Keep announcing ourselves so the sim keeps streaming for the whole window."""

    def loop():
        while True:
            _send_gcs_heartbeat(conn)
            time.sleep(GCS_HEARTBEAT_INTERVAL_S)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def _request_message_interval(conn, message_id, interval_us):
    """Best-effort MAVLink request to stream `message_id` at a fixed rate."""
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        _MAV_CMD_SET_MESSAGE_INTERVAL,
        0,  # confirmation
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def _handshake(conn):
    """Do what the flight client does: heartbeat, then request the streams.

    Returns True if the sim answered, False on timeout (in which case the
    block profile below is meaningless and we say so).
    """
    print("[probe] sending GCS heartbeat, waiting for the sim...", flush=True)
    if _wait_for_sim_heartbeat(conn) is None:
        print(
            f"[probe] no sim HEARTBEAT within {HEARTBEAT_TIMEOUT_S:.0f}s — "
            "is the sim running with an active flight session on this port?",
            flush=True,
        )
        return False
    print(f"[probe] connected to system {conn.target_system}", flush=True)
    _start_gcs_heartbeat_thread(conn)
    for msg_id in _REQUESTED:
        _request_message_interval(conn, msg_id, REQUEST_INTERVAL_US)
    # TIMESYNC is request/response, so SET_MESSAGE_INTERVAL cannot produce it —
    # it only appears if we actually ping. Without this, "TIMESYNC absent" says
    # nothing about the block.
    conn.mav.timesync_send(0, time.time_ns())
    print(
        f"[probe] requested {len(_REQUESTED)} streams at "
        f"{1e6 / REQUEST_INTERVAL_US:.0f} Hz: {', '.join(str(m) for m in _REQUESTED)}",
        flush=True,
    )
    return True


def _report_actuators(sample):
    """Dump ACTUATOR_OUTPUT_STATUS — the ACHIEVED motor state.

    This is not in the competition SPEC.md but the sim streams it. It is the
    only observation of what the motors actually did (vs what we commanded),
    so it can replace the commanded-thrust term the estimator currently
    dead-reckons on, and it accounts for saturation and command lag.
    """
    n = int(getattr(sample, "active", 0) or 0)
    vals = list(getattr(sample, "actuator", []) or [])
    live = [v for v in vals if v != 0.0]
    print("\n[probe] ACTUATOR_OUTPUT_STATUS sample:")
    print(f"  active bitmask={n} ({bin(n)}), array len={len(vals)}")
    print(f"  non-zero outputs ({len(live)}): {[round(v, 4) for v in live[:12]]}")
    print(f"  first 8 raw: {[round(v, 4) for v in vals[:8]]}")
    if live:
        print(f"  sum={sum(live):.4f}  mean={sum(live) / len(live):.4f}")
    print(f"  time_usec={getattr(sample, 'time_usec', None)}")


def _report_imu(sample, stamps):
    s = sample
    anorm = math.sqrt(s.xacc**2 + s.yacc**2 + s.zacc**2)
    mnorm = math.sqrt(s.xmag**2 + s.ymag**2 + s.zmag**2)
    n_imu = len(stamps)
    span_us = (max(stamps) - min(stamps)) if stamps else 0
    rate = (n_imu - 1) / (span_us * 1e-6) if n_imu > 1 and span_us > 0 else 0.0
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

    # A frozen time_usec is catastrophic and silent: every consumer that
    # dedupes or differentiates on it (gp_estimation, rl/deploy) throws away
    # almost every sample, and camera<->IMU sync becomes impossible.
    uniq = len(set(stamps))
    print(
        f"  time_usec: n={n_imu} unique={uniq} span={span_us} us "
        f"first={stamps[0] if stamps else None} last={stamps[-1] if stamps else None}"
    )
    if n_imu > 1 and span_us <= 0:
        print(
            "  -> *** time_usec IS NOT ADVANCING *** derive dt from arrival time, "
            "and do NOT dedupe on time_usec (it would drop every sample but one)"
        )
    elif uniq < n_imu:
        print(
            f"  -> {n_imu - uniq}/{n_imu} samples share a timestamp "
            f"({100.0 * (n_imu - uniq) / n_imu:.0f}% would be dropped by a dedupe)"
        )

    # The estimator disables mag/baro on NaN; say so explicitly rather than
    # leaving it to be re-derived from a printed 'nan'.
    dead = [
        name
        for name, val in (("mag", mnorm), ("baro", s.abs_pressure))
        if math.isnan(val)
    ]
    if dead:
        print(
            f"  -> NaN sensors (estimator will self-disable these): {', '.join(dead)}"
        )
    else:
        print("  -> mag and baro carry finite values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument(
        "--passive",
        action="store_true",
        help="listen only, send nothing (old behaviour; absence is uninformative)",
    )
    args = ap.parse_args()

    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    mode = "PASSIVE" if args.passive else "ACTIVE"
    print(f"[probe] {mode} on udpin:{args.ip}:{args.port} for {args.seconds:.0f}s")

    handshook = False
    if not args.passive:
        handshook = _handshake(conn)

    counts: Counter = Counter()
    encap: Counter = Counter()
    imu_sample = None
    imu_stamps: list = []
    act_sample = None
    print(f"[probe] measuring for {args.seconds:.0f}s...", flush=True)
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
            imu_stamps.append(msg.time_usec)
        elif mtype == "ACTUATOR_OUTPUT_STATUS":
            act_sample = msg
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

    # An "absent" only means "blocked" if we actually asked for it.
    if args.passive:
        print(
            "  NOTE: passive run — 'absent' may just mean the stream was never "
            "requested. Re-run without --passive before concluding anything."
        )
    elif not handshook:
        print("  NOTE: handshake failed — 'absent' is uninformative for every row.")
    else:
        print(
            "  Streams were requested after a GCS heartbeat, so 'absent' here "
            "means the sim is withholding them."
        )

    unexpected = sorted(set(counts) - set(BLOCKED_EXPECTED) - set(KEPT_EXPECTED))
    if unexpected:
        print("\n[probe] present but NOT in either expected list:")
        for m in unexpected:
            print(f"  {m:<28} {counts[m]:>6}  ({counts[m] / dur:6.1f} Hz)")

    if imu_sample is not None:
        _report_imu(imu_sample, imu_stamps)
    if act_sample is not None:
        _report_actuators(act_sample)


if __name__ == "__main__":
    main()
