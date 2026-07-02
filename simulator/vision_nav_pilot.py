"""VQ2 auto-flight pilot: main's vision-only navigator behind the auto loop.

Wraps simulator.vision_nav.VisionGuidance (YOLO+PnP gate detections -> persistent
world gate map -> attitude targets) and converts its targets to attitude-rate
commands with the measured fly2 law. Pose comes from sim odometry when the
session provides it, else the VQ2 ESKF estimate (logged once per change so a
bad fallback is diagnosable from the flight log).
"""

from __future__ import annotations

import time

import numpy as np

from rl import spec
from rl.fly2_course import (
    HOVER_T,
    detect_climb_course,
    rates_from_attitude_targets,
    resolve_gate_map,
    rpy,
    track_gates_to_gate_map,
)
from simulator.vision_nav import VisionGuidance
from simulator.vq2_pose import VQ2PoseEstimator

STATUS_LOG_INTERVAL_S = 1.0


class VisionNavPilot:
    """Drop-in auto-loop pilot: VisionGuidance targets -> fly2 rate commands."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self.guide = VisionGuidance()
        self.gate_map: list = []
        self.hold_z: float | None = None
        self._pose = VQ2PoseEstimator()
        self._pose_source: str | None = None
        self._unsafe_ticks = 0
        self._last_status_log = 0.0
        self._last_pose_frame_id = None
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_T)
        print("[vnav] vision navigator pilot ready (YOLO+PnP gates)", flush=True)

    @property
    def gates_passed(self) -> int:
        return self.guide.n_passed

    def on_attempt_start(self) -> None:
        self.guide = VisionGuidance()
        # Gate map is used only to seed/fuse the EKF fallback pose — the
        # navigator itself flies purely from detections.
        self.gate_map = resolve_gate_map(self.data)
        track = self.data.get("track_gates") or self.data.get("gates") or []
        from_track = self.data.get("track_positions_valid") is not False and bool(
            track_gates_to_gate_map(track)
        )
        # Live track burst is already NED; the climb heuristic only applies to
        # the up-positive captured gate_map.json convention.
        flipz = False if from_track else detect_climb_course(self.gate_map)
        self._pose.reset(self.gate_map, flipz=flipz)
        self.hold_z = None
        self._pose_source = None
        self._unsafe_ticks = 0
        self._last_status_log = 0.0
        self._last_pose_frame_id = None

    def reset_for_attempt(self) -> None:
        self.guide = VisionGuidance()
        self.gate_map = []
        self.hold_z = None
        self._pose = VQ2PoseEstimator()
        self._pose_source = None
        self._unsafe_ticks = 0
        self._last_pose_frame_id = None
        self.data.pop("gate_target", None)
        self.data.pop("pose", None)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, HOVER_T)

    def _log_pose_source(self, source: str) -> None:
        if source != self._pose_source:
            self._pose_source = source
            print(f"[vnav] pose source: {source}", flush=True)

    def _current_odometry(self) -> dict | None:
        odo = self.data.get("odometry")
        if odo is not None:
            self._log_pose_source("odometry")
            return odo
        est = self._pose.tick(self.data, self.gate_map)
        if est is not None:
            self._log_pose_source("EKF")
        return est

    def tick(self) -> None:
        odo = self._current_odometry()
        if not odo:
            self.controller.set_attitude_rates(0, 0, 0, HOVER_T)
            return

        quat = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]], float)
        pos = np.array([odo["x"], odo["y"], odo["z"]], float)
        vel = np.array(
            [odo.get("vx", 0.0), odo.get("vy", 0.0), odo.get("vz", 0.0)], float
        )
        roll, pitch, yaw = rpy(quat)
        z, vz = pos[2], vel[2]
        if self.hold_z is None:
            self.hold_z = z

        # Feed detections only once per inference frame: the 250 Hz loop would
        # otherwise re-count the same frame's gates every tick, confirming a
        # map entry (min_hits) off a single -- possibly corrupt -- frame.
        pose_data = self.data.get("pose")
        gates = []
        if pose_data and pose_data.get("frame_id") != self._last_pose_frame_id:
            self._last_pose_frame_id = pose_data.get("frame_id")
            gates = pose_data["gates"]
        cmd = self.guide.update(gates, pos, vel, quat, yaw, time.monotonic())

        now = time.monotonic()
        if now - self._last_status_log >= STATUS_LOG_INTERVAL_S:
            self._last_status_log = now
            print(f"[vnav] {cmd.status}", flush=True)

        roll_cmd, pitch_cmd, yaw_cmd, thrust = rates_from_attitude_targets(
            roll, pitch, z, vz, cmd.tgt_roll, cmd.tgt_pitch, cmd.yaw_err, cmd.tgt_z
        )

        gb_z = (spec.quat_to_R(quat).T @ np.array([0.0, 0.0, 1.0]))[2]
        unsafe = gb_z < 0.0 or z < self.hold_z - 30 or z > self.hold_z + 30
        if unsafe:
            self._unsafe_ticks += 1
            if self._unsafe_ticks >= 5:
                self.controller.set_attitude_rates(0, 0, 0, HOVER_T)
            else:
                self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        else:
            self._unsafe_ticks = 0
            self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
