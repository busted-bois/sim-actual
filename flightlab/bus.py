"""MAVLink bus: connect, GCS heartbeat, 90 Hz send, arm/reset/disarm.

MAVLINK20 MUST be set before the first pymavlink import (ODOMETRY is MAV2-only).
"""

from __future__ import annotations

import os
import threading
import time

# ODOMETRY (msg 331) requires MAVLink 2 — set before any pymavlink import.
os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from simulator.controller import _send_attitude_rates  # noqa: E402

from flightlab.state import State, StateTracker  # noqa: E402

CONTROL_HZ = 90.0
HOVER_THRUST = 0.27
THRUST_MIN = 0.12
THRUST_MAX = 0.60
MAVLINK_CMD_SIM_RESET = 31000
DEFAULT_CONN = "udpin:0.0.0.0:14550"
# Estimator needs ~100 IMU samples (~1 s at 100+ Hz); allow boot margin.
POSE_WAIT_S = 20.0


class Bus:
    """Thin MAVLink client for the vertical harness."""

    def __init__(self, conn_str: str = DEFAULT_CONN) -> None:
        self.system_boot_ms = int(time.time() * 1000)
        self.tracker = StateTracker()
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._last_arm_t = 0.0
        self._want_armed = False

        print(f"[bus] connecting {conn_str} ...", flush=True)
        self.conn = mavutil.mavlink_connection(conn_str)
        self._send_gcs_heartbeat()
        hb = self.conn.wait_heartbeat(timeout=15)
        if hb is None or self.conn.target_system == 0:
            raise TimeoutError(
                "no vehicle heartbeat within 15 s - is the sim running "
                "and a TRAINING session started?"
            )
        print(
            f"[bus] heartbeat from system {self.conn.target_system}",
            flush=True,
        )
        self._start_gcs_heartbeat()

    def _send_gcs_heartbeat(self) -> None:
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )

    def _start_gcs_heartbeat(self) -> None:
        def loop() -> None:
            while not self._hb_stop.wait(1.0):
                try:
                    self._send_gcs_heartbeat()
                except Exception:
                    break

        self._hb_thread = threading.Thread(target=loop, daemon=True)
        self._hb_thread.start()

    def drain(self) -> State:
        """Drain UDP buffer; return freshest State."""
        while True:
            try:
                msg = self.conn.recv_match(blocking=False)
            except ConnectionResetError:
                break
            if msg is None:
                break
            if msg.get_type() == "BAD_DATA":
                continue
            self.tracker.ingest(msg)
        # Re-arm every ~1 s while we want armed and aren't yet.
        if self._want_armed and not self.tracker.armed:
            now = time.monotonic()
            if now - self._last_arm_t >= 1.0:
                self._send_arm(True)
                self._last_arm_t = now
        return self.tracker.snapshot()

    def wait_for_pose(self, timeout_s: float = POSE_WAIT_S) -> bool:
        """Wait for ODOMETRY/ATTITUDE or ESKF boot on HIGHRES_IMU (VQ2)."""
        t0 = time.monotonic()
        imu_n = 0
        while time.monotonic() - t0 < timeout_s:
            # Blocking read so we don't miss the IMU boot window.
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except ConnectionResetError:
                msg = None
            if msg is not None and msg.get_type() != "BAD_DATA":
                if msg.get_type() == "HIGHRES_IMU":
                    imu_n += 1
                self.tracker.ingest(msg)
            # Also drain any backlog.
            s = self.drain()
            if s.has_pose:
                print(
                    f"[bus] pose source: {s.pose_source} (imu_seen={imu_n})",
                    flush=True,
                )
                return True
        print(
            f"[bus] pose timeout: imu_seen={imu_n} "
            f"est_ready={self.tracker.estimator.ready} "
            f"odom={self.tracker._seen_odometry}",
            flush=True,
        )
        return False

    def wait_for_race_go(
        self, timeout_s: float = 45.0, is_restart: bool = True
    ) -> bool:
        """Drain MAVLink until on-screen countdown hits 0 (sim GO!)."""
        from simulator.preflight import RaceGoLatch, poll_race_go

        print("[bus] waiting for race GO (countdown -> 0)...", flush=True)
        latch = RaceGoLatch()
        race = self.tracker.data.get("race_status") or {}
        armed_boot = race.get("sim_boot_time_ms")
        latch.reset_for_arm(armed_boot, is_restart=is_restart)
        t0 = time.monotonic()
        last_log = 0.0
        while time.monotonic() - t0 < timeout_s:
            self.drain()
            allowed, go_boot_ms = poll_race_go(self.tracker.data, latch)
            if allowed:
                race = self.tracker.data.get("race_status") or {}
                print(
                    "[bus] Race go! "
                    f"sim_boot={race.get('sim_boot_time_ms')}ms "
                    f"race_start={race.get('race_start_boot_time_ms')}ms "
                    f"go_boot={go_boot_ms}ms branch={latch.branch}",
                    flush=True,
                )
                return True
            now = time.monotonic()
            if now - last_log >= 1.0:
                race = self.tracker.data.get("race_status") or {}
                print(
                    "[bus] countdown... "
                    f"sim_boot={race.get('sim_boot_time_ms', -1)} "
                    f"race_start={race.get('race_start_boot_time_ms', -1)} "
                    f"latch={latch.go_boot_ms}",
                    flush=True,
                )
                last_log = now
            time.sleep(0.02)
        print("[bus] race GO timeout", flush=True)
        return False

    def wait_for_fresh_race_start(self, timeout_s: float = 30.0) -> bool:
        """After sim reset, wait for a new race_start before arming."""
        print("[bus] waiting for fresh race_start after reset...", flush=True)
        before = None
        race0 = self.tracker.data.get("race_status")
        if race0:
            before = race0.get("race_start_boot_time_ms", -1)
            self.tracker.data["_preflight_race_start_baseline"] = before
        t0 = time.monotonic()
        last_log = 0.0
        while time.monotonic() - t0 < timeout_s:
            self.drain()
            race = self.tracker.data.get("race_status") or {}
            race_start = race.get("race_start_boot_time_ms", -1)
            sim_boot = race.get("sim_boot_time_ms", 0)
            if race_start >= 0:
                # Fresh if differs from baseline, or scheduled in the future,
                # or sim_boot reset small after teleport.
                baseline = self.tracker.data.get("_preflight_race_start_baseline")
                scheduled = race_start - sim_boot > 1500
                changed = baseline is None or race_start != baseline
                rebooted = sim_boot < 10000
                if scheduled or changed or rebooted:
                    print(
                        f"[bus] fresh race_start={race_start} sim_boot={sim_boot}",
                        flush=True,
                    )
                    return True
            now = time.monotonic()
            if now - last_log >= 2.0:
                print(
                    f"[bus] waiting race_start... start={race_start} boot={sim_boot}",
                    flush=True,
                )
                last_log = now
            time.sleep(0.05)
        print(
            "[bus] race_start timeout — click Restart Race if countdown never starts",
            flush=True,
        )
        return False

    def send(
        self, roll_rate: float, pitch_rate: float, yaw_rate: float, thrust: float
    ) -> None:
        thrust = float(max(THRUST_MIN, min(THRUST_MAX, thrust)))
        # Estimator predicts velocity from commanded thrust (accel is garbage
        # under power in this sim).
        self.tracker.estimator.thrust_cmd = thrust
        _send_attitude_rates(
            self.conn,
            self.system_boot_ms,
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            thrust=thrust,
        )

    def _send_arm(self, arm: bool) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0 if arm else 0.0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def arm(self, timeout_s: float = 15.0) -> bool:
        self._want_armed = True
        t0 = time.monotonic()
        self._send_arm(True)
        self._last_arm_t = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            s = self.drain()
            if s.armed:
                print("[bus] armed", flush=True)
                return True
            time.sleep(0.05)
        print("[bus] arm timeout", flush=True)
        return False

    def disarm(self) -> None:
        self._want_armed = False
        self._send_arm(False)
        for _ in range(5):
            self.drain()
            self._send_arm(False)
            time.sleep(0.05)

    def reset(self) -> None:
        self._want_armed = False
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self.tracker.reset_estimator()
        print("[bus] sim reset (31000)", flush=True)

    def close(self) -> None:
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=2.0)
