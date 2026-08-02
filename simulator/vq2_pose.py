"""VQ2 pose estimate: commanded-thrust ESKF predict + vision fusion (no sim odometry)."""

from __future__ import annotations

import math

import numpy as np

from rl.estimation.ekf import ESKF
from rl.perception.vision_fusion import (
    fuse_gate_bearing_yaw,
    fuse_gate_target_position,
    step_pnp_fusion,
)
from simulator.gp_vision import best_pose_gate

SPAWN_BEFORE_GATE_M = 15.0
SPAWN_ABOVE_GATE_M = 1.0
DEFAULT_SPAWN_Z = -5.0
CAM_HZ = 30.0
MAX_COAST_FRAMES = 30
PNP_GATE_FLOOR = 3.0


def gate_z_ned(pos, flipz: bool) -> float:
    z = float(pos[2])
    return -z if flipz else z


def spawn_position_ned(gate_map: list, flipz: bool = False) -> np.ndarray:
    if not gate_map:
        return np.array([0.0, 0.0, DEFAULT_SPAWN_Z], dtype=float)
    g0 = np.asarray(gate_map[0]["pos"], float)
    if not np.isfinite(g0).all():
        return np.array([0.0, 0.0, DEFAULT_SPAWN_Z], dtype=float)
    z0 = gate_z_ned(g0, flipz)
    return np.array(
        [g0[0] + SPAWN_BEFORE_GATE_M, g0[1], z0 - SPAWN_ABOVE_GATE_M], dtype=float
    )


def spawn_heading_ned(gate_map: list) -> float:
    if not gate_map:
        return 0.0
    p0 = spawn_position_ned(gate_map)
    g0 = np.asarray(gate_map[0]["pos"], float)
    return float(math.atan2(g0[1] - p0[1], g0[0] - p0[0]))


def _level_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


class VQ2PoseEstimator:
    def __init__(self) -> None:
        self.ekf: ESKF | None = None
        self._last_imu_time_us: int | None = None
        self._hold_z = DEFAULT_SPAWN_Z
        self._pressure_alt_ref: float | None = None
        self._flipz = False
        self._last_gate_frame_id = None
        self._last_pnp_frame_id = None
        self._nan_resets = 0
        self._last_applied_thrust = 0.0

    @property
    def last_applied_thrust(self) -> float:
        return self._last_applied_thrust

    def reset(
        self, gate_map: list, hold_z: float | None = None, flipz: bool = False
    ) -> None:
        self._flipz = flipz
        p0 = spawn_position_ned(gate_map, flipz=flipz)
        self._hold_z = float(hold_z if hold_z is not None else p0[2])
        yaw0 = spawn_heading_ned(gate_map)
        self.ekf = ESKF(p0=p0, v0=np.zeros(3), q0=_level_quat(yaw0))
        self._last_imu_time_us = None
        self._pressure_alt_ref = None
        self._last_gate_frame_id = None
        self._last_pnp_frame_id = None
        self._nan_resets = 0
        self._last_applied_thrust = 0.0

    def _check(self, stage: str, gate_map: list) -> bool:
        if self.ekf is None or self.ekf.healthy():
            return True
        self._nan_resets += 1
        if self._nan_resets <= 3:
            print(
                f"[vq2] EKF non-finite after {stage} -- reset #{self._nan_resets}",
                flush=True,
            )
        p0 = spawn_position_ned(gate_map, flipz=self._flipz)
        yaw0 = spawn_heading_ned(gate_map)
        self.ekf = ESKF(p0=p0, v0=np.zeros(3), q0=_level_quat(yaw0))
        self._last_imu_time_us = None
        self._pressure_alt_ref = None
        return False

    def tick(
        self, data: dict, gate_map: list, *, thrust_cmd: float = 0.0
    ) -> dict | None:
        if self.ekf is None:
            return None

        try:
            thrust = float(thrust_cmd)
        except (TypeError, ValueError):
            thrust = 0.0
        if not math.isfinite(thrust) or not 0.0 <= thrust <= 1.0:
            thrust = 0.0
        self._last_applied_thrust = thrust

        imu = data.get("imu")
        if imu is not None:
            self._predict_imu(imu)
            self._update_z_from_pressure(imu)
        if not self._check("imu predict", gate_map):
            return None

        gate_target = data.get("gate_target") or {}
        if not isinstance(gate_target, dict):
            gate_target = {}
        frame_id = gate_target.get("frame_id")
        is_new_frame = frame_id is None or frame_id != self._last_gate_frame_id
        if gate_target.get("detected") and gate_map and is_new_frame:
            active = int(data.get("active_gate_index", 0) or 0)
            if 0 <= active < len(gate_map):
                pos = gate_map[active]["pos"]
                gate_world = np.array(
                    [pos[0], pos[1], gate_z_ned(pos, self._flipz)], dtype=float
                )
                drone_pos = self.ekf.p.copy()
                fuse_gate_target_position(
                    self.ekf, gate_target, gate_world, self.ekf.q.copy()
                )
                fuse_gate_bearing_yaw(self.ekf, gate_target, drone_pos, gate_world)
        if frame_id is not None:
            self._last_gate_frame_id = frame_id

        pnp = data.get("pose") or {}
        if not isinstance(pnp, dict):
            pnp = {}
        pnp_frame_id = pnp.get("frame_id")
        if pnp_frame_id is not None and pnp_frame_id != self._last_pnp_frame_id:
            self._last_pnp_frame_id = pnp_frame_id
            try:
                active = int(data.get("active_gate_index", 0) or 0)
            except (TypeError, ValueError):
                active = -1
            if gate_map and 0 <= active < len(gate_map):
                pos = gate_map[active]["pos"]
                gate_world = np.array(
                    [pos[0], pos[1], gate_z_ned(pos, self._flipz)], dtype=float
                )
                try:
                    det = best_pose_gate(data)
                except (KeyError, TypeError, ValueError):
                    det = None
                cam_dt = 1.0 / CAM_HZ
                step_pnp_fusion(
                    self.ekf,
                    det,
                    gate_world,
                    cam_dt,
                    max_coast=MAX_COAST_FRAMES,
                    gate_floor=PNP_GATE_FLOOR,
                )

        if not self._check("vision fusion", gate_map):
            return None
        st = self.ekf.state()
        p, v, q = st["p"], st["v"], st["q"]
        return {
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "vx": float(v[0]),
            "vy": float(v[1]),
            "vz": float(v[2]),
            "qw": float(q[0]),
            "qx": float(q[1]),
            "qy": float(q[2]),
            "qz": float(q[3]),
        }

    def _predict_imu(self, imu: dict) -> None:
        t_us = imu.get("time_us")
        try:
            t_value = float(t_us)
        except (TypeError, ValueError):
            return
        if not math.isfinite(t_value):
            return
        if self._last_imu_time_us is not None:
            dt = (int(t_value) - int(self._last_imu_time_us)) * 1e-6
            if 0.0 < dt < 0.5:
                try:
                    gyro_arr = np.asarray(
                        [imu["gx"], imu["gy"], imu["gz"]], dtype=np.float64
                    ).reshape(3)
                except (KeyError, TypeError, ValueError):
                    gyro_arr = None
                if gyro_arr is not None and np.isfinite(gyro_arr).all():
                    self.ekf.predict_commanded(self._last_applied_thrust, gyro_arr, dt)
        self._last_imu_time_us = int(t_value)

    def _update_z_from_pressure(self, imu: dict) -> None:
        pa = imu.get("pressure_alt")
        if pa is None or self.ekf is None or not math.isfinite(float(pa)):
            return
        if self._pressure_alt_ref is None:
            self._pressure_alt_ref = float(pa)
        z_meas = -(float(pa) - self._pressure_alt_ref) + self._hold_z
        self.ekf.p[2] = 0.7 * self.ekf.p[2] + 0.3 * z_meas
