from pymavlink import mavutil

from simulator.controller import Controller
from simulator.gate_tracker import GateTracker
from simulator.mavlink_rx import MAVLinkRX
from simulator.timesync import TimeSync
from simulator.vio import VisualInertialOdometry
from simulator.vision_rx import VisionRX


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection(
        "udpin:%s:%s"
        % (
            server_ip,
            server_udp_port,
        )
    )
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    vio = VisualInertialOdometry(shared_data)
    shared_data["vio_processor"] = vio
    gate_tracker = GateTracker(shared_data)
    shared_data["gate_tracker"] = gate_tracker
    print("[setup] VIO enabled (drone: IMU + camera)", flush=True)
    print("[setup] Gate tracker enabled (gate KF + detector)", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data, vio=vio)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data, vio=vio, gate_tracker=gate_tracker)

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        "vision_rx": vision_rx,
        "mavlink_rx": mavlink_rx,
        "ts_loop": ts_loop,
        "sim_conn": sim_conn,
        "controller": controller,
        "vio": vio,
        "gate_tracker": gate_tracker,
    }
