"""Visual-Inertial Odometry (VIO) for the live simulator.

Fuses high-rate IMU propagation with periodic gate PnP vision updates inside
an error-state Kalman filter (ESKF). Vision corrections counter IMU drift so
the drone can navigate without GPS.
"""
from __future__ import annotations

import math
import time as _time

import numpy as np

from rl.ekf import ESKF
from rl.pnp import estimate_pose, pose_from_mask
from simulator.config import GateDetection

BASE_VISION_SIGMA_M = 0.15
MAX_REPROJ_ERR_PX = 15.0
MAX_SEED_HORIZ_M = 500.0
MAX_SEED_ALT_M = 80.0
MAX_GATE_TRACK_DELTA_M = 15.0
MIN_VISION_QUALITY = 0.15


class VisualInertialOdometry:
    """Loosely-coupled VIO: IMU predict + gate PnP position update."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.ekf = ESKF()
        self._last_imu_time_us: int | None = None
        self._initialized = False
        self._seeded_from_telemetry = False
        self._last_vision_mono: float | None = None
        self._last_reproj_err_px: float | None = None
        self._vision_updates = 0

    def try_seed_from_telemetry(self) -> None:
        """One-time bootstrap from sim odometry (not used for ongoing correction)."""
        if self._seeded_from_telemetry:
            return
        telem = self._read_telemetry()
        if telem is None:
            return
        p, v, q = telem
        self.ekf.p = p
        self.ekf.v = v
        self.ekf.q = q
        self._initialized = True
        self._seeded_from_telemetry = True
        self._publish_state(vision_valid=False)

    def _read_telemetry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        odo = self.data.get("odometry")
        if odo and self._odometry_sane(odo):
            p = np.array([odo["x"], odo["y"], odo["z"]], dtype=float)
            v = np.array([odo["vx"], odo["vy"], odo["vz"]], dtype=float)
            q = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]], dtype=float)
            return p, v, q

        pos_ned = self.data.get("pos_ned")
        if not pos_ned or not self.data.get("has_position"):
            return None
        p = np.asarray(pos_ned, dtype=float)
        if not self._position_sane(p):
            return None
        vel = self.data.get("vel_ned") or (0.0, 0.0, 0.0)
        v = np.asarray(vel, dtype=float)
        return p, v, self._quat_from_data()

    @staticmethod
    def _position_sane(p: np.ndarray) -> bool:
        if abs(float(p[2])) > MAX_SEED_ALT_M:
            return False
        return float(np.linalg.norm(p)) <= MAX_SEED_HORIZ_M

    def _quat_from_data(self) -> np.ndarray:
        odo = self.data.get("odometry")
        if odo and "qw" in odo:
            return np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]], dtype=float)

        att = self.data.get("attitude")
        if att:
            roll, pitch, yaw = att["roll"], att["pitch"], att["yaw"]
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            return np.array(
                [
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ],
                dtype=float,
            )

        yaw = float(self.data.get("yaw_rad", 0.0))
        half = yaw * 0.5
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)

    @staticmethod
    def _odometry_sane(odo: dict) -> bool:
        return VisualInertialOdometry._position_sane(
            np.array([odo["x"], odo["y"], odo["z"]], dtype=float)
        )

    def predict_imu(self, imu: dict) -> None:
        """High-rate IMU prediction step (Kalman predict)."""
        self.try_seed_from_telemetry()

        time_us = int(imu["time_us"])
        if self._last_imu_time_us is None:
            self._last_imu_time_us = time_us
            return
        if time_us == self._last_imu_time_us:
            return

        dt_s = (time_us - self._last_imu_time_us) * 1e-6
        self._last_imu_time_us = time_us
        if dt_s <= 0 or dt_s > 0.5:
            return

        accel = np.array([imu["ax"], imu["ay"], imu["az"]], dtype=float)
        gyro = np.array([imu["gx"], imu["gy"], imu["gz"]], dtype=float)
        self.ekf.predict(accel, gyro, dt_s)
        if self._seeded_from_telemetry:
            self._publish_state(vision_valid=False)

    def update_from_gate_detection(
        self,
        detection: GateDetection,
        frame_id: int,
        sim_time_ns: int,
    ) -> bool:
        """Vision update from detector corners (same PnP path as gate tracker)."""
        self.try_seed_from_telemetry()
        if not self._seeded_from_telemetry:
            return False
        if detection.corners_px is None or len(detection.corners_px) < 4:
            return False
        if detection.reproj_err_px is None:
            return False
        if detection.quality < MIN_VISION_QUALITY:
            return False

        gate = self._gate_world_for_pnp()
        if gate is None:
            return False

        gate_pos, _gate_quat = gate
        corners = np.asarray(detection.corners_px, dtype=float)
        pose = estimate_pose(
            corners,
            drone_quat=self._drone_quat_for_pnp(),
            gate_world_pos=gate_pos,
        )
        return self._apply_vision_pose(
            pose,
            frame_id,
            sim_time_ns,
            float(detection.reproj_err_px),
        )

    def update_from_gate_mask(
        self,
        mask: np.ndarray,
        frame_id: int,
        sim_time_ns: int,
    ) -> bool:
        """Vision update from a gate segmentation mask (Kalman correct)."""
        self.try_seed_from_telemetry()
        if not self._seeded_from_telemetry:
            return False

        gate = self._gate_world_for_pnp()
        if gate is None:
            return False

        gate_pos, _gate_quat = gate
        pose = pose_from_mask(
            mask,
            drone_quat=self._drone_quat_for_pnp(),
            gate_world_pos=gate_pos,
        )
        if pose is None:
            return False
        return self._apply_vision_pose(
            pose,
            frame_id,
            sim_time_ns,
            float(pose.get("reproj_err_px", 99.0)),
        )

    def _apply_vision_pose(
        self,
        pose: dict | None,
        frame_id: int,
        sim_time_ns: int,
        reproj_err: float,
    ) -> bool:
        if pose is None or "drone_pos_world" not in pose:
            return False
        if reproj_err > MAX_REPROJ_ERR_PX:
            return False

        p_vis = np.asarray(pose["drone_pos_world"], float)
        telem = self._read_telemetry()
        if telem is not None:
            if float(np.linalg.norm(p_vis - telem[0])) > MAX_GATE_TRACK_DELTA_M:
                return False

        sigma = BASE_VISION_SIGMA_M * (1.0 + reproj_err / 5.0)
        self.ekf.update_position(p_vis, sigma=sigma)
        self._initialized = True
        self._last_vision_mono = _time.monotonic()
        self._last_reproj_err_px = reproj_err
        self._vision_updates += 1
        self._publish_state(
            vision_valid=True,
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            range_m=float(pose["range_m"]),
            reproj_err_px=reproj_err,
        )
        return True

    def _drone_quat_for_pnp(self) -> np.ndarray:
        """Use live telemetry attitude for PnP, not IMU-propagated quaternion."""
        return self._quat_from_data()

    def _gate_world_for_pnp(self) -> tuple[np.ndarray, np.ndarray] | None:
        track = self._active_gate_world()
        if track is not None:
            track_pos, track_quat = track
            gt = self.data.get("gate_track")
            if gt and gt.get("initialized"):
                gt_pos = np.asarray(gt["pos_ned"], dtype=float)
                if np.linalg.norm(gt_pos - track_pos) <= MAX_GATE_TRACK_DELTA_M:
                    return gt_pos, np.asarray(gt["quat"], dtype=float)
            return track_pos, track_quat

        gt = self.data.get("gate_track")
        if gt and gt.get("initialized"):
            return np.asarray(gt["pos_ned"], dtype=float), np.asarray(
                gt["quat"], dtype=float
            )
        return None

    def _active_gate_world(self) -> tuple[np.ndarray, np.ndarray] | None:
        track_gates = self.data.get("track_gates") or []
        if not track_gates:
            return None
        idx = int(self.data.get("active_gate_index", 0))
        idx = max(0, min(idx, len(track_gates) - 1))
        gate = track_gates[idx]
        pos = gate.get("position_ned")
        quat = gate.get("orientation_ned")
        if not pos or not quat:
            return None
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    def _publish_state(
        self,
        vision_valid: bool,
        frame_id: int | None = None,
        sim_time_ns: int | None = None,
        range_m: float | None = None,
        reproj_err_px: float | None = None,
    ) -> None:
        st = self.ekf.state()
        now = _time.monotonic()
        vision_age_s = (
            None if self._last_vision_mono is None else now - self._last_vision_mono
        )
        self.data["vio"] = {
            "initialized": self._initialized,
            "seeded_from_telemetry": self._seeded_from_telemetry,
            "pos_ned": tuple(float(x) for x in st["p"]),
            "vel_ned": tuple(float(x) for x in st["v"]),
            "quat": tuple(float(x) for x in st["q"]),
            "P_trace": float(st["P_trace"]),
            "vision_valid": vision_valid,
            "vision_age_s": vision_age_s,
            "vision_updates": self._vision_updates,
            "reproj_err_px": reproj_err_px,
            "range_m": range_m,
            "frame_id": frame_id,
            "sim_time_ns": sim_time_ns,
        }
        if self._initialized:
            self.data["pos_ned_vio"] = self.data["vio"]["pos_ned"]
            self.data["vel_ned_vio"] = self.data["vio"]["vel_ned"]
            self.data["quat_vio"] = self.data["vio"]["quat"]
