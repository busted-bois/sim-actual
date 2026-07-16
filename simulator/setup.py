from pymavlink import mavutil

from simulator.controller import Controller
from simulator.mavlink_rx import MAVLinkRX
from simulator.preflight import udp_port_in_use
from simulator.timesync import TimeSync
from simulator.vision_rx import VisionRX

# HIGHRES_IMU is the "primary sensor" for VQ2 (ODOMETRY/ATTITUDE disabled
# there); request it explicitly since nothing streams it by default.
IMU_MESSAGE_INTERVAL_US = 10000  # 100 Hz


def _request_message_interval(sim_conn, message_id: int, interval_us: int) -> None:
    """Best-effort MAVLink request to stream `message_id` at a fixed rate."""
    sim_conn.mav.command_long_send(
        sim_conn.target_system,
        sim_conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,  # confirmation
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    if udp_port_in_use(server_ip, server_udp_port):
        raise TimeoutError(
            f"UDP {server_udp_port} already in use by another process "
            "(likely a stale `make auto`/`make sim`). Free it with `make free-port`, "
            "then retry."
        )
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection(
        "udpin:%s:%s"
        % (
            server_ip,
            server_udp_port,
        )
    )
    print("Waiting for heartbeat...", flush=True)
    if not sim_conn.wait_heartbeat(timeout=30):
        raise TimeoutError(
            "No MAVLink heartbeat on UDP 14550 — open FlightSim and enter a flight session first"
        )
    print(f"Connected to system: {sim_conn.target_system}", flush=True)
    _request_message_interval(
        sim_conn, mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, IMU_MESSAGE_INTERVAL_US
    )
    # NOTE: no GCS heartbeat here on purpose. The flight path (auto/auto-gp/
    # sim) historically flies without one, and adding it changes the wire
    # traffic mid-flight; flightlab/bus.py manages its own for the harness.

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
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data)

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
    }
