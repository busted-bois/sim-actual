"""VQ2 pose estimate: ESKF IMU predict + vision fusion (no sim odometry)."""

from __future__ import annotations

import math

import numpy as np

from rl.ekf import ESKF
from rl.vision_fusion import (
    fuse_gate_bearing_yaw,
    fuse_gate_target_position,
    fuse_pnp_gate,
)

SPAWN_BEFORE_GATE_M = 15.0
SPAWN_ABOVE_GATE_M = 1.0
DEFAULT_SPAWN_Z = -5.0


def gate_z_ned(pos, flipz: bool) -> float:
    """Gate z in NED (down-positive), normalizing captured-map convention.

    `rl/data/gate_map.json` is captured up-positive on climb courses (see
    `detect_climb_course`); `flipz` negates it to match live NED z, the same
    correction `compute_course_rates` applies to its altitude target.
    """
    z = float(pos[2])
    return -z if flipz else z


def spawn_position_ned(gate_map: list, flipz: bool = False) -> np.ndarray:
    """Approximate race spawn before gate 0 (gates run -X on this map)."""
    if not gate_map:
        return np.array([0.0, 0.0, DEFAULT_SPAWN_Z], dtype=float)
    g0 = np.asarray(gate_map[0]["pos"], float)
    if not np.isfinite(g0).all():  # corrupt map entry must not seed a NaN EKF
        return np.array([0.0, 0.0, DEFAULT_SPAWN_Z], dtype=float)
    z0 = gate_z_ned(g0, flipz)
    return np.array(
        [g0[0] + SPAWN_BEFORE_GATE_M, g0[1], z0 - SPAWN_ABOVE_GATE_M], dtype=float
    )


def spawn_heading_ned(gate_map: list) -> float:
    """Yaw (rad, NED) facing from the approximate spawn point toward gate 0."""
    if not gate_map:
        return 0.0
    p0 = spawn_position_ned(gate_map)
    g0 = np.asarray(gate_map[0]["pos"], float)
    return float(math.atan2(g0[1] - p0[1], g0[0] - p0[0]))


def _level_quat(yaw: float) -> np.ndarray:
    """Zero roll/pitch quaternion at the given yaw (w,x,y,z)."""
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

    def reset(
        self, gate_map: list, hold_z: float | None = None, flipz: bool = False
    ) -> None:
        self._flipz = flipz
        p0 = spawn_position_ned(gate_map, flipz=flipz)
        self._hold_z = float(hold_z if hold_z is not None else p0[2])
        yaw0 = spawn_heading_ned(gate_map)
        self.ekf = ESKF(p0=p0, v0=np.zeros(3), q0=_level_quat(yaw0), p0_std=5.0)
        self._last_imu_time_us = None
        self._pressure_alt_ref = None
        self._last_gate_frame_id = None
        self._last_pnp_frame_id = None
        self._nan_resets = 0

    def _check(self, stage: str, gate_map: list) -> bool:
        """True while the filter is finite; on the FIRST divergence name the
        stage that broke it (the 2026-07-02 runs were NaN from tick one with
        no way to tell where), re-seed the filter and count the reset."""
        if self.ekf is None or self.ekf.healthy():
            return True
        self._nan_resets += 1
        if self._nan_resets <= 3:  # don't spam the flight log
            print(
                f"[vq2] EKF non-finite after {stage} -- reset #{self._nan_resets}",
                flush=True,
            )
        p0 = spawn_position_ned(gate_map, flipz=self._flipz)
        yaw0 = spawn_heading_ned(gate_map)
        self.ekf = ESKF(p0=p0, v0=np.zeros(3), q0=_level_quat(yaw0), p0_std=5.0)
        self._last_imu_time_us = None
        self._pressure_alt_ref = None
        return False

    def tick(self, data: dict, gate_map: list) -> dict | None:
        if self.ekf is None:
            return None

        imu = data.get("imu")
        if imu is not None:
            self._predict_imu(imu)
            self._update_z_from_pressure(imu)
        if not self._check("imu predict", gate_map):
            return None

        gate_target = data.get("gate_target") or {}
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

        # YOLO+PnP detections: full 3D gate-in-body measurement, much stronger
        # than the HSV bearing/range above. Fuse the highest-confidence gate in
        # each new inference frame against the active gate's world position.
        pnp = data.get("pose") or {}
        pnp_frame_id = pnp.get("frame_id")
        if pnp_frame_id is not None and pnp_frame_id != self._last_pnp_frame_id:
            self._last_pnp_frame_id = pnp_frame_id
            active = int(data.get("active_gate_index", 0) or 0)
            if gate_map and 0 <= active < len(gate_map):
                pos = gate_map[active]["pos"]
                gate_world = np.array(
                    [pos[0], pos[1], gate_z_ned(pos, self._flipz)], dtype=float
                )
                dets = [g for g in pnp.get("gates", []) if g.get("pose")]
                if dets:
                    best = max(dets, key=lambda g: float(g.get("conf", 0.0) or 0.0))
                    fuse_pnp_gate(self.ekf, best, gate_world)

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
        if t_us is None or not math.isfinite(float(t_us)):
            return
        if self._last_imu_time_us is not None:
            dt = (int(t_us) - int(self._last_imu_time_us)) * 1e-6
            if 0.0 < dt < 0.5:
                accel = (imu["ax"], imu["ay"], imu["az"])
                gyro = (imu["gx"], imu["gy"], imu["gz"])
                # ekf.predict rejects non-finite samples internally
                self.ekf.predict(accel, gyro, dt)
        self._last_imu_time_us = int(t_us)

    def _update_z_from_pressure(self, imu: dict) -> None:
        pa = imu.get("pressure_alt")
        if pa is None or self.ekf is None or not math.isfinite(float(pa)):
            return
        if self._pressure_alt_ref is None:
            self._pressure_alt_ref = float(pa)
        z_meas = -(float(pa) - self._pressure_alt_ref) + self._hold_z
        self.ekf.p[2] = 0.7 * self.ekf.p[2] + 0.3 * z_meas
