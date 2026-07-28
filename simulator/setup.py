import threading
import time

from pymavlink import mavutil

from simulator.controller import Controller
from simulator.mavlink_rx import MAVLinkRX
from simulator.preflight import udp_port_in_use
from simulator.state_estimator import StateEstimator
from simulator.timesync import TimeSync
from simulator.vision_rx import VisionRX

HEARTBEAT_TIMEOUT_S = 60
GCS_HEARTBEAT_INTERVAL_S = 1.0

# HIGHRES_IMU is the "primary sensor" for VQ2 (ODOMETRY/ATTITUDE disabled there);
# request it explicitly since nothing streams it by default.
IMU_MESSAGE_INTERVAL_US = 10000  # 100 Hz

# MAVLink message IDs (numeric so this works under both MAVLink 1 and 2 dialects;
# ODOMETRY = 331 is MAVLink-2-only and absent from the v1 dialect's constants).
_MSG_ID_ATTITUDE = 30
_MSG_ID_LOCAL_POSITION_NED = 32
_MSG_ID_ODOMETRY = 331
_MAV_CMD_SET_MESSAGE_INTERVAL = 511


def _send_gcs_heartbeat(sim_conn):
    """Announce ourselves to the sim as a GCS.

    The sim streams telemetry and accepts offboard setpoints once it sees a
    ground-station heartbeat, so this is sent during connect and kept up.
    """
    sim_conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )


def _wait_for_sim_heartbeat(sim_conn, timeout_s=HEARTBEAT_TIMEOUT_S):
    """Send GCS heartbeats and wait for the sim's heartbeat. Returns it or None."""
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
        time.sleep(0.02)
    return None


def _start_gcs_heartbeat_thread(sim_conn):
    """Keep sending GCS heartbeats so the sim keeps streaming and accepting control."""

    def loop():
        while True:
            _send_gcs_heartbeat(sim_conn)
            time.sleep(GCS_HEARTBEAT_INTERVAL_S)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def _request_message_interval(sim_conn, message_id, interval_us):
    """Best-effort MAVLink request to stream `message_id` at a fixed rate."""
    sim_conn.mav.command_long_send(
        sim_conn.target_system,
        sim_conn.target_component,
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


def _request_data_streams(sim_conn, rate_hz=50):
    """Request the pose streams manual flight uses (HIGHRES_IMU is requested
    separately at 100 Hz). Harmless if already streaming — we keep the latest."""
    interval_us = int(1_000_000 / rate_hz)
    for msg_id in (_MSG_ID_ATTITUDE, _MSG_ID_LOCAL_POSITION_NED, _MSG_ID_ODOMETRY):
        _request_message_interval(sim_conn, msg_id, interval_us)


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
    print("Waiting for heartbeat (sending GCS heartbeat)...", flush=True)
    if _wait_for_sim_heartbeat(sim_conn) is None:
        raise TimeoutError(
            f"No MAVLink heartbeat within {HEARTBEAT_TIMEOUT_S}s — is the sim running "
            f"with an active flight session on udp {server_ip}:{server_udp_port}?"
        )
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # Keep announcing ourselves so the sim keeps streaming telemetry and accepting
    # our setpoints, and request the streams we consume.
    _start_gcs_heartbeat_thread(sim_conn)
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
    # IMU-driven ESKF. Under the VQ2 block the sim never sends ODOMETRY/
    # ATTITUDE/LOCAL_POSITION_NED, so velocity and attitude feedback read
    # nan/0 and every speed/attitude loop in the pilots is inert. MAVLinkRX
    # publishes this only as a fallback (see _publish_estimated_state), so a
    # link that streams pose normally is unaffected.
    estimator = StateEstimator()
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data, estimator=estimator)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data)

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms)
    # The ESKF predicts velocity from COMMANDED thrust, not the accelerometer
    # (this sim's accel is only clean on the ground) — so it needs every send.
    controller.estimator = estimator

    return {
        "vision_rx": vision_rx,
        "mavlink_rx": mavlink_rx,
        "ts_loop": ts_loop,
        "sim_conn": sim_conn,
        "controller": controller,
        "estimator": estimator,
    }
