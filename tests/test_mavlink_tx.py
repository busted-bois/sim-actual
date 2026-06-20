from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simulator.controller import Controller
from simulator.mavlink_tx import GCSHeartbeat, send_gcs_heartbeat


def test_send_gcs_heartbeat():
    mav = MagicMock()
    conn = SimpleNamespace(mav=mav)
    send_gcs_heartbeat(conn)
    mav.heartbeat_send.assert_called_once()


def test_gcs_heartbeat_respects_interval():
    mav = MagicMock()
    conn = SimpleNamespace(mav=mav)
    hb = GCSHeartbeat(conn, interval_s=1.0)

    with patch("simulator.mavlink_tx.time.monotonic", side_effect=[0.0, 0.1, 1.1]):
        hb.tick()
        hb.tick()
        hb.tick()

    assert mav.heartbeat_send.call_count == 2


def test_controller_update_sends_gcs_heartbeat():
    mav = MagicMock()
    conn = SimpleNamespace(
        target_system=1,
        target_component=1,
        mav=mav,
    )
    ctrl = Controller(conn, {}, system_boot_ms=1000)
    ctrl.pilot = MagicMock()

    with patch.object(ctrl._gcs_heartbeat, "tick") as tick:
        ctrl.update()
        tick.assert_called_once()
