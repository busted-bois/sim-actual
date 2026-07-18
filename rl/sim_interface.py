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
from simulator.state_estimator import StateEstimator, quat_mult
from simulator.timesync import TimeSync
from simulator.transforms import quat_to_yaw
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
        use_estimator: bool = False,
    ):
        self.data: dict = {}
        self.system_boot_ms = int(time.time() * 1000)
        # ESKF on HIGHRES_IMU: the state source under the VQ2 telemetry block.
        # use_estimator forces it even when odometry is present (dress rehearsal);
        # otherwise it's the automatic fallback when odometry is blocked.
        self.estimator = StateEstimator()
        self.use_estimator = use_estimator
        self._shadow = None  # lazy CSV: estimator-vs-odometry error log
        self._shadow_next_t = 0.0
        print(f"[sim] connecting MAVLink udpin:{ip}:{mav_port} ...", flush=True)
        self.conn = mavutil.mavlink_connection(f"udpin:{ip}:{mav_port}")
        self.conn.wait_heartbeat()
        print(f"[sim] heartbeat from system {self.conn.target_system}", flush=True)
        self.mavlink_rx = MAVLinkRX.create_mavlink_rx(
            self.conn, self.data, estimator=self.estimator
        )
        self.timesync = TimeSync(self.conn, self.data)
        self.timesync.thread = None  # TimeSync.create starts a thread; start manually
        self._start_timesync()
        self.vision_rx = VisionRX(self.data)

    def _start_timesync(self):
        import threading

        self.timesync.is_running = True
        self.timesync.thread = threading.Thread(
            target=self.timesync.timesync_loop, daemon=True
        )
        self.timesync.thread.start()

    # ---- telemetry -------------------------------------------------------
    def wait_for_telemetry(self, timeout_s: float = 15.0) -> bool:
        """Block until IMU + a camera frame have arrived (odometry optional --
        it's blocked in Qualification; the estimator covers self-state)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.data.get("imu") is not None and self.data.get("frame"):
                # Give the estimator its ground-init window before flying.
                if self.estimator.ready:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> Snapshot:
        d = self.data
        odo = d.get("odometry")
        frame = d.get("frame")
        att = d.get("attitude")
        imu = d.get("imu")
        pos = quat = vel = ang = None
        yaw = d.get("yaw_rad")
        if odo is not None:
            pos = (odo["x"], odo["y"], odo["z"])
            vel = (odo["vx"], odo["vy"], odo["vz"])
            quat = (odo["qw"], odo["qx"], odo["qy"], odo["qz"])
            ang = (odo["roll_speed"], odo["pitch_speed"], odo["yaw_speed"])
        elif att is not None:
            ang = (att["roll_speed"], att["pitch_speed"], att["yaw_speed"])
        est_pose = self.estimator.pose() if self.estimator.ready else None
        if est_pose is not None and odo is not None:
            self._shadow_log(odo, est_pose)
        if est_pose is not None and (odo is None or self.use_estimator):
            p_e, v_e, q_e = est_pose
            pos, vel, quat = tuple(p_e), tuple(v_e), tuple(q_e)
            yaw = quat_to_yaw(*q_e)
            if ang is None and imu is not None:
                ang = (imu["gx"], imu["gy"], imu["gz"])
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

    def _shadow_log(self, odo: dict, est_pose, hz: float = 5.0):
        """Estimator-vs-odometry error CSV (Training-mode validation).

        The estimator frame is boot-anchored (p=0, yaw=0 at init) while
        odometry has its own origin/heading -- the two differ by a FIXED yaw
        rotation + translation, captured from the first sample.
        """
        now = time.monotonic()
        if now < self._shadow_next_t:
            return
        self._shadow_next_t = now + 1.0 / hz
        p_e, v_e, q_e = est_pose
        p_o = np.array([odo["x"], odo["y"], odo["z"]])
        v_o = np.array([odo["vx"], odo["vy"], odo["vz"]])
        q_o = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]])
        if self._shadow is None:
            dyaw = quat_to_yaw(*q_o) - quat_to_yaw(*q_e)
            c, s = np.cos(dyaw), np.sin(dyaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(
                DATA_DIR, f"shadow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            f = open(path, "w")
            f.write("t,pos_err,vel_err,att_err_deg,z_err,ex,ey,ez,ox,oy,oz\n")
            q_align = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)])
            self._shadow = {
                "f": f,
                "t0": now,
                "Rz": Rz,
                "off": p_o - Rz @ p_e,
                "q_align": q_align,
            }
            print(f"[sim] shadow log -> {path}", flush=True)
        sh = self._shadow
        p_ea = sh["Rz"] @ p_e + sh["off"]  # estimator pose in the odometry frame
        v_ea = sh["Rz"] @ v_e
        q_ea = quat_mult(sh["q_align"], q_e)
        dq = quat_mult(np.array([q_o[0], -q_o[1], -q_o[2], -q_o[3]]), q_ea)
        att_err = np.degrees(2 * np.arccos(min(1.0, abs(float(dq[0])))))
        sh["f"].write(
            f"{now - sh['t0']:.2f},{np.linalg.norm(p_ea - p_o):.3f},"
            f"{np.linalg.norm(v_ea - v_o):.3f},{att_err:.2f},"
            f"{abs(p_ea[2] - p_o[2]):.3f},"
            f"{p_ea[0]:.2f},{p_ea[1]:.2f},{p_ea[2]:.2f},"
            f"{p_o[0]:.2f},{p_o[1]:.2f},{p_o[2]:.2f}\n"
        )
        sh["f"].flush()

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
        self, path: str = GATE_MAP_PATH, timeout_s: float = 90.0
    ) -> list:
        """Wait for the track gate list over MAVLink, persist it to JSON.

        The sim broadcasts gate poses (relative NED) via ENCAPSULATED_DATA as
        a short race-start burst. Missed / VQ2-nulled bursts fall back to a
        previously saved ``path`` (same as ``make fly`` / fly2).

        If ``path`` already exists, only wait ``min(timeout_s, 15)`` for a live
        burst before falling back so ``make fly-policy`` can arm promptly.
        """
        has_saved = os.path.isfile(path)
        live_wait = min(timeout_s, 15.0) if has_saved else timeout_s
        print(
            "[sim] waiting for gate-map burst — click Race in FlightSim now "
            f"(or restart Race) within {live_wait:.0f}s"
            + (f"; will fall back to {path}" if has_saved else "")
            + " ...",
            flush=True,
        )
        t0 = time.monotonic()
        last_status = 0.0
        while time.monotonic() - t0 < live_wait:
            gates = self.gate_list()
            # Skip VQ2-nulled bursts (all zeros) — fall through to saved JSON.
            if gates and self.data.get("track_positions_valid") is not False:
                usable = [
                    g
                    for g in gates
                    if abs(g["pos"][0]) + abs(g["pos"][1]) + abs(g["pos"][2]) >= 0.01
                ]
                if usable:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        json.dump({"gates": usable}, f, indent=2)
                    print(
                        f"[sim] gate map ({len(usable)} gates) -> {path}",
                        flush=True,
                    )
                    return usable
            now = time.monotonic()
            if now - last_status >= 5.0:
                left = max(0.0, live_wait - (now - t0))
                print(
                    f"[sim] still waiting for gate map... {left:.0f}s left",
                    flush=True,
                )
                last_status = now
            time.sleep(0.1)
        if has_saved:
            try:
                saved = load_gate_map(path)
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                saved = []
            if saved:
                print(
                    f"[sim] live burst missed; using saved {path} ({len(saved)} gates)",
                    flush=True,
                )
                return saved
        print(
            "[sim] WARNING: no gate map — run `make capture-gates`, click Race "
            "while it listens, then retry",
            flush=True,
        )
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
        # The estimator predicts velocity from the commanded thrust (this
        # sim's accelerometer is garbage under power) -- keep it informed.
        self.estimator.thrust_cmd = float(thrust)
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
