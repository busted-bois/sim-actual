import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simulator.setup import wait_for_sim_heartbeat


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
