"""Gate detector + Kalman filter pipeline.

Camera frames -> gate detection (corners/contour/bbox) -> gate KF measurement
updates. VIO drone pose is used only to transform detections into world frame.
"""

from __future__ import annotations

import math
import time as _time

import numpy as np

from simulator.config import GateDetection
from simulator.gate_kalman import (
    BASE_ATT_SIGMA_RAD,
    BASE_POS_SIGMA_M,
    GateKalmanFilter,
    measure_gate_world,
)

MAX_MEASUREMENT_JUMP_M = 30.0
UPCOMING_GATE_MATCH_M = 12.0


def _course_gate_index(data: dict) -> int:
    """Pilot's sequential gate index — not lagging sim active_gate_index."""
    return int(data.get("pilot_course_gate_index", data.get("active_gate_index", 0)))


class GateTracker:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.kf = GateKalmanFilter()
        self._seeded_from_track = False
        self._last_course_idx = -1
        self._last_predict_mono = _time.monotonic()
        self._updates = 0
        self._misses = 0

    def try_seed_from_track(self) -> None:
        track_gates = self.data.get("track_gates") or []
        if not track_gates:
            return
        idx = _course_gate_index(self.data)
        idx = max(0, min(idx, len(track_gates) - 1))
        if idx == self._last_course_idx and self.kf.initialized:
            return
        gate = track_gates[idx]
        pos = gate.get("position_ned")
        quat = gate.get("orientation_ned")
        if not pos or not quat:
            return
        self.kf.seed(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))
        self._seeded_from_track = True
        self._last_course_idx = idx
        self._publish_upcoming_from_map(idx)
        self._publish(detected=False, quality=0.0)

    def _publish_upcoming_from_map(self, idx: int) -> None:
        track_gates = self.data.get("track_gates") or []
        if idx + 1 >= len(track_gates):
            self.data.pop("upcoming_gate", None)
            return
        upcoming = track_gates[idx + 1]
        pos = upcoming.get("position_ned")
        if not pos:
            return
        gid = upcoming.get("gate_id", idx + 1)
        self.data["upcoming_gate"] = {
            "gate_id": gid,
            "position_ned": tuple(float(x) for x in pos[:3]),
            "course_index": idx + 1,
            "source": "map",
        }

    def _upcoming_track_gate_pos(self) -> np.ndarray | None:
        track_gates = self.data.get("track_gates") or []
        if not track_gates:
            return None
        idx = _course_gate_index(self.data)
        if idx + 1 >= len(track_gates):
            return None
        pos = track_gates[idx + 1].get("position_ned")
        if not pos:
            return None
        return np.asarray(pos, dtype=float)

    def _active_track_gate_pos(self) -> np.ndarray | None:
        track_gates = self.data.get("track_gates") or []
        if not track_gates:
            return None
        idx = _course_gate_index(self.data)
        idx = max(0, min(idx, len(track_gates) - 1))
        pos = track_gates[idx].get("position_ned")
        if not pos:
            return None
        return np.asarray(pos, dtype=float)

    def _drone_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        # Map-relative transforms use odometry (same frame as track_gates).
        odo = self.data.get("odometry")
        if odo:
            pos = np.array([odo["x"], odo["y"], odo["z"]], dtype=float)
            quat = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]], dtype=float)
            return pos, quat
        pos_ned = self.data.get("pos_ned")
        attitude = self.data.get("attitude")
        if pos_ned and attitude is not None:
            yaw = float(attitude.get("yaw", 0.0))
            half = yaw * 0.5
            quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
            return np.asarray(pos_ned, dtype=float), quat
        return None

    def process_detection(
        self,
        detection: GateDetection | None,
        frame_id: int,
        sim_time_ns: int,
    ) -> bool:
        self.try_seed_from_track()
        now = _time.monotonic()
        dt_s = max(0.0, now - self._last_predict_mono)
        self._last_predict_mono = now
        self.kf.predict(dt_s)

        if detection is None or not detection.contour_valid:
            self._misses += 1
            self._publish(detected=False, quality=0.0, frame_id=frame_id)
            return False

        if not self.kf.initialized:
            self._misses += 1
            self._publish(detected=True, quality=detection.quality, frame_id=frame_id)
            return False

        if detection.corners_px is None or len(detection.corners_px) < 4:
            self._misses += 1
            self._publish(detected=False, quality=detection.quality, frame_id=frame_id)
            return False

        drone_pose = self._drone_pose()
        if drone_pose is None:
            self._misses += 1
            self._publish(detected=True, quality=detection.quality, frame_id=frame_id)
            return False

        drone_pos, drone_quat = drone_pose
        corners = np.asarray(detection.corners_px, dtype=float)
        meas = measure_gate_world(corners, drone_pos, drone_quat)
        if meas is None:
            self._misses += 1
            self._publish(detected=True, quality=detection.quality, frame_id=frame_id)
            return False

        ref_pos = self._active_track_gate_pos()
        upcoming_pos = self._upcoming_track_gate_pos()
        if ref_pos is not None:
            jump = float(np.linalg.norm(meas["pos_ned"] - ref_pos))
            jump_upcoming = (
                float(np.linalg.norm(meas["pos_ned"] - upcoming_pos))
                if upcoming_pos is not None
                else float("inf")
            )
            if (
                upcoming_pos is not None
                and jump_upcoming < UPCOMING_GATE_MATCH_M
                and jump_upcoming < jump - 2.0
            ):
                self._publish_upcoming_detected(
                    meas,
                    upcoming_pos,
                    quality=float(detection.quality),
                    frame_id=frame_id,
                    sim_time_ns=sim_time_ns,
                    reproj_err_px=float(meas["reproj_err_px"]),
                    range_m=float(meas["range_m"]),
                )
                return True
            if jump > MAX_MEASUREMENT_JUMP_M:
                self._misses += 1
                self._publish(detected=True, quality=detection.quality, frame_id=frame_id)
                return False
        else:
            jump = float(np.linalg.norm(meas["pos_ned"] - self.kf.p))
            if jump > MAX_MEASUREMENT_JUMP_M:
                self._misses += 1
                self._publish(detected=True, quality=detection.quality, frame_id=frame_id)
                return False

        reproj = float(meas["reproj_err_px"])
        quality = float(detection.quality)
        sigma_pos = BASE_POS_SIGMA_M * (1.0 + reproj / 5.0) / max(quality, 0.15)
        sigma_att = BASE_ATT_SIGMA_RAD * (1.0 + reproj / 5.0) / max(quality, 0.15)
        if not self.kf.update(meas["pos_ned"], meas["quat"], sigma_pos, sigma_att):
            self._misses += 1
            self._publish(detected=True, quality=quality, frame_id=frame_id)
            return False
        self._updates += 1
        self._publish(
            detected=True,
            quality=quality,
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            reproj_err_px=reproj,
            range_m=float(meas["range_m"]),
        )
        return True

    def _publish_upcoming_detected(
        self,
        meas: dict,
        upcoming_pos: np.ndarray,
        *,
        quality: float,
        frame_id: int,
        sim_time_ns: int,
        reproj_err_px: float,
        range_m: float,
    ) -> None:
        track_gates = self.data.get("track_gates") or []
        idx = _course_gate_index(self.data)
        upcoming = track_gates[idx + 1] if idx + 1 < len(track_gates) else {}
        gid = upcoming.get("gate_id", idx + 1)
        self.data["upcoming_gate_detected"] = {
            "detected": True,
            "gate_id": gid,
            "pos_ned": tuple(float(x) for x in meas["pos_ned"]),
            "map_pos_ned": tuple(float(x) for x in upcoming_pos),
            "quality": quality,
            "reproj_err_px": reproj_err_px,
            "range_m": range_m,
            "frame_id": frame_id,
            "sim_time_ns": sim_time_ns,
        }
        self._publish(
            detected=True,
            quality=quality,
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            reproj_err_px=reproj_err_px,
            range_m=range_m,
        )

    def _publish(
        self,
        detected: bool,
        quality: float,
        frame_id: int | None = None,
        sim_time_ns: int | None = None,
        reproj_err_px: float | None = None,
        range_m: float | None = None,
    ) -> None:
        st = self.kf.state()
        self.data["gate_track"] = {
            "initialized": st["initialized"],
            "detected": detected,
            "pos_ned": tuple(float(x) for x in st["pos_ned"]),
            "quat": tuple(float(x) for x in st["quat"]),
            "P_trace": float(st["P_trace"]),
            "quality": quality,
            "updates": self._updates,
            "misses": self._misses,
            "reproj_err_px": reproj_err_px,
            "range_m": range_m,
            "frame_id": frame_id,
            "sim_time_ns": sim_time_ns,
        }
