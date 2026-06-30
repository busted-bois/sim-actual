from __future__ import annotations

import argparse
import select
import socket


def bound(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    return sock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ip", required=True)
    parser.add_argument("--mavlink-port", type=int, default=14550)
    args = parser.parse_args()

    sim = bound(args.mavlink_port)
    pilot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sim_addr = None

    print(f"relay :{args.mavlink_port} <-> {args.target_ip}:{args.mavlink_port}", flush=True)
    while True:
        readable, _, _ = select.select([sim, pilot], [], [], 1.0)
        for sock in readable:
            data, addr = sock.recvfrom(65536)
            if sock is sim:
                sim_addr = addr
                pilot.sendto(data, (args.target_ip, args.mavlink_port))
            elif sim_addr is not None:
                sim.sendto(data, sim_addr)


if __name__ == "__main__":
    main()
