from pymavlink import mavutil

from simulator.controller import Controller
from simulator.dead_reckon import DeadReckoner
from simulator.gate_perception import GatePerception
from simulator.gate_transition import GateTransitionConfig, GateTransitionTracker
from simulator.mavlink_rx import MAVLinkRX
from simulator.timesync import TimeSync
from simulator.velocity_estimate import VelocityEstimator
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

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync(sim_conn, shared_data)

    # -------------------------------
    # Gate transition modules
    # -------------------------------
    perception = GatePerception(shared_data)
    velocity_estimator = VelocityEstimator(shared_data)
    tracker = GateTransitionTracker(
        GateTransitionConfig(r_acceptance=0.5, t_min=0.5, v_max=1.0)
    )
    dead_reckoner = DeadReckoner(shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data, perception=perception)

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(
        sim_conn,
        shared_data,
        system_boot_ms,
        velocity_estimator=velocity_estimator,
        tracker=tracker,
        dead_reckoner=dead_reckoner,
    )

    return {
        "vision_rx": vision_rx,
        "mavlink_rx": mavlink_rx,
        "ts_loop": ts_loop,
        "sim_conn": sim_conn,
        "controller": controller,
    }
