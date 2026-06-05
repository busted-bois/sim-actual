import socket

import subprocess

import sys

import time


from pymavlink import mavutil


from simulator.controller import Controller

from simulator.mavlink_rx import MAVLinkRX

from simulator.timesync import TimeSync

from simulator.vision_rx import VisionRX


HEARTBEAT_TIMEOUT_S = 60

HEARTBEAT_POLL_S = 0.02

GCS_HEARTBEAT_INTERVAL_S = 1.0


SIM_INSTALL_HINT = (
    "C:\\Users\\trung\\Downloads\\AI-GP Simulator v1.0.3364\\AIGP_3364\\FlightSim.exe"
)


def _udp_port_available(host, port):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.bind((host, port))

        return True

    except OSError:
        return False

    finally:
        sock.close()


def _pid_on_udp_port(port):

    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    except (OSError, subprocess.TimeoutExpired):
        return None

    needle = f":{port}"

    for line in result.stdout.splitlines():
        if "UDP" not in line or needle not in line:
            continue

        parts = line.split()

        if parts and parts[-1].isdigit():
            return int(parts[-1])

    return None


def _send_gcs_heartbeat(sim_conn):

    sim_conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )


def wait_for_sim_heartbeat(sim_conn, timeout_s=HEARTBEAT_TIMEOUT_S):

    deadline = time.monotonic() + timeout_s

    next_tx = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()

        if now >= next_tx:
            _send_gcs_heartbeat(sim_conn)

            next_tx = now + GCS_HEARTBEAT_INTERVAL_S

        msg = sim_conn.recv_match(type="HEARTBEAT", blocking=False)

        if msg is not None:
            return msg

        time.sleep(HEARTBEAT_POLL_S)

    return None


def _exit_port_in_use(host, port):

    pid = _pid_on_udp_port(port)

    print(
        f"\nUDP port {port} is already in use on {host}.",
        flush=True,
    )

    if pid is not None:
        print(
            f"  Process holding the port: PID {pid} "
            f"(stop any other `make sim` / python pilot, then retry).",
            flush=True,
        )

        print(f"  PowerShell: Stop-Process -Id {pid} -Force", flush=True)

    sys.exit(1)


def _exit_no_heartbeat(host, port):

    print(
        f"\nNo MAVLink heartbeat received on udpin:{host}:{port} "
        f"within {HEARTBEAT_TIMEOUT_S}s.",
        flush=True,
    )

    print(
        "Startup order:\n"
        f"  1. Run FlightSim.exe (e.g. {SIM_INSTALL_HINT})\n"
        "  2. Log in and START a qualifier / flight session (drone in session, not main menu)\n"
        "  3. Then run: make sim\n"
        "Also check:\n"
        "  - FlightSim is still running and the session did not end\n"
        "  - No other pilot is using UDP port 14550\n"
        "  - Windows Firewall allows FlightSim.exe and python.exe on private networks",
        flush=True,
    )

    print(
        "Run `make mavlink-probe` while a session is active for more detail.",
        flush=True,
    )

    sys.exit(1)


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):

    if not _udp_port_available(server_ip, server_udp_port):
        _exit_port_in_use(server_ip, server_udp_port)

    sim_conn = mavutil.mavlink_connection("udpin:%s:%s" % (server_ip, server_udp_port))

    endpoint = "udpin:%s:%s" % (server_ip, server_udp_port)

    print(f"Listening for MAVLink on {endpoint}", flush=True)

    print(
        "Ensure FlightSim is running with an active flight session, then wait...",
        flush=True,
    )

    print(
        f"Waiting for heartbeat (timeout {HEARTBEAT_TIMEOUT_S}s, sending GCS heartbeat)...",
        flush=True,
    )

    heartbeat = wait_for_sim_heartbeat(sim_conn, timeout_s=HEARTBEAT_TIMEOUT_S)

    if heartbeat is None:
        _exit_no_heartbeat(server_ip, server_udp_port)

    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    print("Setting up MAVLink rx...", flush=True)

    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    print("Setting up Timesync loop...", flush=True)

    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    vision_rx = VisionRX(shared_data)

    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        "vision_rx": vision_rx,
        "mavlink_rx": mavlink_rx,
        "ts_loop": ts_loop,
        "sim_conn": sim_conn,
        "controller": controller,
    }
