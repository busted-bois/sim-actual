"""Capture the race-start gate-map burst and save it.

The sim transmits the track/gate map ONLY as a short burst at race start
(DATA_TRANSMISSION_HANDSHAKE + ENCAPSULATED_DATA data_type=2). We confirmed
this with raw logging. This tool uses a BLOCKING MAVLink receive (reliable for
the burst) and the same chunk-reassembly + struct layout as
simulator.mavlink_rx, then writes rl/data/gate_map.json.

Usage: start this FIRST, then (re)start the race.
    uv run -m rl.capture_gates           # 150s window
    uv run -m rl.capture_gates 90
"""

import json
import os
import struct
import sys
import time

from pymavlink import mavutil

from rl.sim_interface import GATE_MAP_PATH

TRACK_INFO_DATA_TYPE = 2


def decode_gates(full: bytes) -> list:
    (num_gates,) = struct.unpack_from("<H", full, 0)
    off = 2
    gates = []
    for _ in range(num_gates):
        (gid, px, py, pz, ow, ox, oy, oz, w, h) = struct.unpack_from(
            "<Hfffffffff", full, off
        )
        off += 38
        gates.append(
            {
                "id": int(gid),
                "pos": [px, py, pz],
                "quat": [ow, ox, oy, oz],
                "w": w,
                "h": h,
            }
        )
    return gates


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    print(f"connecting udpin:127.0.0.1:14550 ({duration:.0f}s window) ...", flush=True)
    conn = mavutil.mavlink_connection("udpin:127.0.0.1:14550")
    conn.wait_heartbeat()
    print(
        "heartbeat OK. === (RE)START THE RACE NOW to trigger the gate burst ===",
        flush=True,
    )

    expected: dict[int, int] = {}
    chunks: dict[int, dict[int, bytes]] = {}
    gates = None
    t0 = time.time()
    while time.time() - t0 < duration and gates is None:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "DATA_TRANSMISSION_HANDSHAKE":
            tid = msg.width
            expected[tid] = msg.packets
            chunks[tid] = {}
            print(
                f"[handshake] transfer_id={tid} packets={msg.packets} size={msg.size}",
                flush=True,
            )
        elif t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if not raw or raw[0] != TRACK_INFO_DATA_TYPE:
                continue
            _dt, tid = struct.unpack_from("<BH", raw)
            if tid not in expected:
                continue
            chunks[tid][msg.seqnr] = raw[3:]
            if len(chunks[tid]) == expected[tid]:
                full = b"".join(chunks[tid][i] for i in range(expected[tid]))
                gates = decode_gates(full)

    if gates:
        os.makedirs(os.path.dirname(GATE_MAP_PATH), exist_ok=True)
        with open(GATE_MAP_PATH, "w") as f:
            json.dump({"gates": gates}, f, indent=2)
        print(f"CAPTURED {len(gates)} gates -> {GATE_MAP_PATH}", flush=True)
        for g in gates:
            p, q = g["pos"], g["quat"]
            print(
                f"  gate {g['id']}: pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                f"quat=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}) "
                f"w={g['w']:.2f} h={g['h']:.2f}",
                flush=True,
            )
    else:
        print("NO gate burst captured in window.", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
