"""Raw MAVLink diagnostic — dump EVERY message the sim sends, unfiltered.

Bypasses simulator.mavlink_rx entirely so a decoder bug can't hide gate data.
Logs first occurrence of every message type in full, every ENCAPSULATED_DATA /
DATA_TRANSMISSION_HANDSHAKE chunk, anything gate/mission/track-ish, and a final
histogram of message-type counts.

    uv run -m rl.diag_mav            # 25s on the live race
    uv run -m rl.diag_mav 40         # custom duration
"""

import collections
import os
import sys
import time

from pymavlink import mavutil

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
INTERESTING = (
    "MISSION",
    "WAYPOINT",
    "TRACK",
    "GATE",
    "POSITION_TARGET",
    "NAMED_VALUE",
    "STATUSTEXT",
    "DEBUG",
    "PARAM",
)


def main():
    print(f"connecting udpin:127.0.0.1:14550 (logging {DURATION:.0f}s) ...", flush=True)
    conn = mavutil.mavlink_connection("udpin:127.0.0.1:14550")
    conn.wait_heartbeat()
    print(f"heartbeat from system {conn.target_system}; LOGGING NOW", flush=True)

    counts = collections.Counter()
    encap_types = collections.Counter()
    t0 = time.time()
    next_tick = 5.0
    while time.time() - t0 < DURATION:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            continue
        t = msg.get_type()
        counts[t] += 1

        if t == "BAD_DATA":
            continue

        # First time we see a type, dump it whole.
        if counts[t] == 1:
            try:
                print(f"[FIRST] {t}: {msg.to_dict()}", flush=True)
            except Exception as e:
                print(f"[FIRST] {t}: <{e}>", flush=True)

        if t == "DATA_TRANSMISSION_HANDSHAKE":
            print(f"[HANDSHAKE] {msg.to_dict()}", flush=True)

        if t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            dt = raw[0] if raw else -1
            encap_types[dt] += 1
            if encap_types[dt] <= 3:  # first few of each subtype
                print(
                    f"[ENCAP] seqnr={getattr(msg, 'seqnr', '?')} data_type={dt} "
                    f"len={len(raw)} head={raw[:16].hex()}",
                    flush=True,
                )

        if any(k in t for k in INTERESTING):
            print(f"[{t}] {msg.to_dict()}", flush=True)

        el = time.time() - t0
        if el >= next_tick:
            print(
                f"  ...{el:.0f}s, {sum(counts.values())} msgs, types={sorted(counts)}",
                flush=True,
            )
            next_tick += 5.0

    print("=== MESSAGE TYPE COUNTS ===", flush=True)
    for k, v in counts.most_common():
        print(f"  {k:32s} {v}", flush=True)
    if encap_types:
        print("=== ENCAPSULATED_DATA subtypes (first byte) ===", flush=True)
        for k, v in encap_types.most_common():
            print(f"  data_type={k}: {v}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
