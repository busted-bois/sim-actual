import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simulator.setup import setup_components, wait_for_sim_heartbeat


def test_wait_for_sim_heartbeat_returns_first_heartbeat():
    conn = MagicMock()
    conn.recv_match.side_effect = [None, SimpleNamespace(get_type=lambda: "HEARTBEAT")]
    with patch("simulator.setup._send_gcs_heartbeat"):
        with patch("simulator.setup.time.sleep"):
            hb = wait_for_sim_heartbeat(conn, timeout_s=1.0)
    assert hb is not None
    conn.mav.heartbeat_send.assert_not_called()


def test_wait_for_sim_heartbeat_times_out():
    conn = MagicMock()
    conn.recv_match.return_value = None
    start = time.monotonic()
    with patch("simulator.setup._send_gcs_heartbeat"):
        hb = wait_for_sim_heartbeat(conn, timeout_s=0.05)
    assert hb is None
    assert time.monotonic() - start >= 0.05


def test_setup_exits_when_vision_preflight_fails():
    shared = {}
    conn = MagicMock()
    conn.target_system = 1
    conn.recv_match.return_value = SimpleNamespace(get_type=lambda: "HEARTBEAT")

    with patch("simulator.setup._udp_port_available", return_value=True):
        with patch("simulator.setup.mavutil.mavlink_connection", return_value=conn):
            with patch("simulator.setup.wait_for_sim_heartbeat", return_value=conn):
                with patch("simulator.setup.run_preflight_checks", return_value=False):
                    with patch("simulator.setup.sys.exit") as exit_mock:
                        setup_components(shared, 1000, "127.0.0.1", 14550)
                        exit_mock.assert_called_once_with(1)
