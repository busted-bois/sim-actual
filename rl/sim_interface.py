"""Module 1 — Simulator + MAVLink interface.

Thin wrapper over the working ``simulator`` stack (pymavlink telemetry +
UDP camera). Exposes a clean, synchronized snapshot of IMU / attitude /
velocity / position / RGB frame plus the gate map, and low-level actuation
(arm, attitude-rate+thrust send, sim reset).

This is the live-sim boundary. Only Modules 1, 2 and final evaluation talk
to it; RL training (Modules 7-8) runs against the internal physics model.

Run a smoke test / dump the gate map:
    uv run -m rl.sim_interface
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from pymavlink import mavutil

from simulator.controller import _send_attitude_rates
from simulator.mavlink_rx import MAVLinkRX
from simulator.timesync import TimeSync
from simulator.vision_rx import VisionRX

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GATE_MAP_PATH = os.path.join(DATA_DIR, "gate_map.json")

DEFAULT_IP = "127.0.0.1"
DEFAULT_MAV_PORT = 14550


@dataclass
class Snapshot:
    """One synchronized view of vehicle state + camera."""

    t_mono: float
    armed: bool
    pos_ned: tuple | None  # (x,y,z) world NED, meters
    vel_ned: tuple | None  # (vx,vy,vz) world NED, m/s
    quat: tuple | None  # (w,x,y,z) body->world
    yaw_rad: float | None
    ang_vel: tuple | None  # (roll,pitch,yaw) rate rad/s
    imu: dict | None  # ax,ay,az,gx,gy,gz (body)
    frame: np.ndarray | None  # BGR HxWx3
    frame_time_ns: int | None
    gates: list = field(default_factory=list)  # [{id,pos,quat,w,h}, ...]

    def has_pose(self) -> bool:
        return self.pos_ned is not None and self.quat is not None


class SimInterface:
    def __init__(
        self,
        ip: str = DEFAULT_IP,
        mav_port: int = DEFAULT_MAV_PORT,
        start_vision: bool = True,
    ):
        self.data: dict = {}
        self.system_boot_ms = int(time.time() * 1000)
        print(f"[sim] connecting MAVLink udpin:{ip}:{mav_port} ...", flush=True)
        self.conn = mavutil.mavlink_connection(f"udpin:{ip}:{mav_port}")
        self.conn.wait_heartbeat()
        print(f"[sim] heartbeat from system {self.conn.target_system}", flush=True)
        self.mavlink_rx = MAVLinkRX.create_mavlink_rx(self.conn, self.data)
        self.timesync = TimeSync(self.conn, self.data)
        self.timesync.thread = None  # TimeSync.create starts a thread; start manually
        self._start_timesync()
        # Camera RX is only needed by vision modules; controllers that fly on the
        # hardcoded gate map (rl.fly2) skip it to avoid the per-frame log spam + CPU.
        self.vision_rx = VisionRX(self.data) if start_vision else None

    def _start_timesync(self):
        import threading

        self.timesync.is_running = True
        self.timesync.thread = threading.Thread(
            target=self.timesync.timesync_loop, daemon=True
        )
        self.timesync.thread.start()

    # ---- telemetry -------------------------------------------------------
    def wait_for_telemetry(self, timeout_s: float = 15.0) -> bool:
        """Block until odometry + a camera frame have arrived."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.data.get("odometry") is not None and self.data.get("frame"):
                return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> Snapshot:
        d = self.data
        odo = d.get("odometry")
        frame = d.get("frame")
        att = d.get("attitude")
        pos = quat = vel = ang = None
        yaw = d.get("yaw_rad")
        if odo is not None:
            pos = (odo["x"], odo["y"], odo["z"])
            vel = (odo["vx"], odo["vy"], odo["vz"])
            quat = (odo["qw"], odo["qx"], odo["qy"], odo["qz"])
            ang = (odo["roll_speed"], odo["pitch_speed"], odo["yaw_speed"])
        elif att is not None:
            ang = (att["roll_speed"], att["pitch_speed"], att["yaw_speed"])
        return Snapshot(
            t_mono=time.monotonic(),
            armed=bool(d.get("armed", False)),
            pos_ned=pos,
            vel_ned=vel,
            quat=quat,
            yaw_rad=yaw,
            ang_vel=ang,
            imu=d.get("imu"),
            frame=frame["img"] if frame else None,
            frame_time_ns=frame["sim_time_ns"] if frame else None,
            gates=self.gate_list(),
        )

    def gate_list(self) -> list:
        gates = self.data.get("gates") or []
        out = []
        for g in gates:
            out.append(
                {
                    "id": int(g.gate_id),
                    "pos": list(g.pos_ned),
                    "quat": list(g.orient_quat),  # (w,x,y,z)
                    "w": g.width_m,
                    "h": g.height_m,
                }
            )
        return out

    # ---- gate map --------------------------------------------------------
    def capture_gate_map(
        self, path: str = GATE_MAP_PATH, timeout_s: float = 20.0
    ) -> list:
        """Wait for the track gate list over MAVLink, persist it to JSON.

        The sim broadcasts gate poses (relative NED) via ENCAPSULATED_DATA;
        we snapshot them once so downstream modules have a fixed gate map.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            gates = self.gate_list()
            if gates:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump({"gates": gates}, f, indent=2)
                print(f"[sim] gate map ({len(gates)} gates) -> {path}", flush=True)
                return gates
            time.sleep(0.1)
        print("[sim] WARNING: no gate map received from sim", flush=True)
        return []

    # ---- actuation -------------------------------------------------------
    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        _send_attitude_rates(
            self.conn,
            self.system_boot_ms,
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            thrust=float(thrust),
        )

    def reset_sim(self):
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            31000,  # MAVLINK_CMD_SIM_RESET
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def close(self):
        """Stop the RX + timesync loops and join their (non-daemon) threads."""
        for rx in (self.mavlink_rx, self.vision_rx, self.timesync):
            if rx is None:
                continue
            thread = rx.get_thread_for_join()  # sets is_running=False, returns thread
            if thread is not None:
                thread.join(timeout=2.0)


def load_gate_map(path: str = GATE_MAP_PATH) -> list:
    with open(path) as f:
        return json.load(f)["gates"]


def _smoke():
    import sys

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[sim] no telemetry within timeout - is the race running?", flush=True)
        sys.stdout.flush()
        os._exit(1)
    snap = sim.snapshot()
    print("=== telemetry snapshot ===", flush=True)
    print("armed     :", snap.armed, flush=True)
    print("pos_ned   :", snap.pos_ned, flush=True)
    print("vel_ned   :", snap.vel_ned, flush=True)
    print("quat      :", snap.quat, flush=True)
    print("ang_vel   :", snap.ang_vel, flush=True)
    print("imu       :", snap.imu, flush=True)
    print("frame     :", None if snap.frame is None else snap.frame.shape, flush=True)
    # Gate map (track broadcast). Shorter wait since telemetry already confirmed.
    gates = sim.capture_gate_map(timeout_s=8.0)
    print("gates     :", len(gates), flush=True)
    if gates:
        for g in gates:
            print(f"  gate {g['id']}: pos={g['pos']} w={g['w']} h={g['h']}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # hard-exit past non-daemon receiver threads


if __name__ == "__main__":
    _smoke()
