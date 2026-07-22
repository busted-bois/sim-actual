"""VIO auto-flight pilot: StateEstimator (thrust-model ESKF) + VisionGuidance.

Wires simulator.state_estimator.StateEstimator (IMU strapdown with a
commanded-thrust prediction model in flight -- this sim's accelerometer is
measured-garbage the instant thrust is applied, see state_estimator.py's
module docstring) into the live control loop for the first time, fused with
gate landmark corrections from simulator.vision_nav.VisionGuidance's
detection-to-map association (the same wiring proven offline in
rl/fly2.py --mode vision --est, just running on the live sim instead).

No GPS/odometry dependency anywhere in this pilot -- pose is IMU + vision
only, matching the VQ2 competitive constraint (SPEC.md).
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from rl import spec
from rl.fly2_course import HOVER_T, rates_from_attitude_targets, rpy
from simulator.state_estimator import StateEstimator
from simulator.vision_nav import VisionGuidance

STATUS_LOG_INTERVAL_S = 1.0
NAV_LOG_INTERVAL_S = 0.1


class VIOPilot:
    """Drop-in auto-loop pilot: VisionGuidance targets flown off a live VIO pose."""

    def __init__(self, controller, data, estimator: StateEstimator | None = None):
        self.controller = controller
        self.data = data
        self.estimator = estimator if estimator is not None else StateEstimator()
        self.guide = VisionGuidance()
        self.hold_z: float | None = None
        self._unsafe_ticks = 0
        self._last_status_log = 0.0
        self._last_pose_frame_id = None
        self._last_collision = None
        self._nav_log = None
        self._nav_wr = None
        self._last_nav_log = 0.0
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_T)
        print("[vio] VIO pilot ready (thrust-model ESKF + gate landmarks)", flush=True)

    @property
    def gates_passed(self) -> int:
        return self.guide.n_passed

    def on_attempt_start(self) -> None:
        self.guide = VisionGuidance()
        self.estimator.reset()
        self.hold_z = None
        self._unsafe_ticks = 0
        self._last_status_log = 0.0
        self._last_pose_frame_id = None
        self._last_collision = None
        self._open_nav_log()

    def _open_nav_log(self) -> None:
        self._close_nav_log()
        try:
            os.makedirs(os.path.join("rl", "data"), exist_ok=True)
            path = os.path.join(
                "rl", "data", time.strftime("nav_log_vio_%Y%m%d_%H%M%S.csv")
            )
            self._nav_log = open(path, "w", newline="")
            self._nav_wr = csv.writer(self._nav_log)
            self._nav_wr.writerow(
                "t phase s lat vert dist pn pe pd status n_passed".split()
            )
            print(f"[vio] nav log -> {path}", flush=True)
        except OSError as e:  # telemetry must never ground the pilot
            print(f"[vio] nav log unavailable: {e}", flush=True)
            self._nav_log, self._nav_wr = None, None

    def _close_nav_log(self) -> None:
        if self._nav_log is not None:
            try:
                self._nav_log.close()
            except OSError:
                pass
        self._nav_log, self._nav_wr = None, None

    def reset_for_attempt(self) -> None:
        self.guide = VisionGuidance()
        self.estimator.reset()
        self.hold_z = None
        self._unsafe_ticks = 0
        self._last_pose_frame_id = None
        self._last_collision = None
        self._close_nav_log()
        self.data.pop("gate_target", None)
        self.data.pop("pose", None)
        self.data.pop("vio", None)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, HOVER_T)

    def _publish_vio_state(self, pos, vel, quat) -> None:
        self.data["vio"] = {
            "ready": True,
            "pos_ned": tuple(float(x) for x in pos),
            "vel_ned": tuple(float(x) for x in vel),
            "quat": tuple(float(x) for x in quat),
            "n_landmarks": self.estimator.n_landmarks,
            "n_landmarks_rejected": self.estimator.n_landmarks_rejected,
        }

    def tick(self) -> None:
        col = self.data.get("last_collision")
        if col is not None and col != self._last_collision:
            self._last_collision = col
            self.estimator.notify_collision()

        pose = self.estimator.pose()
        if pose is None:
            self.data["vio"] = {"ready": False}
            self.controller.set_attitude_rates(0, 0, 0, HOVER_T)
            return
        pos, vel, quat = pose

        roll, pitch, yaw = rpy(quat)
        z, vz = float(pos[2]), float(vel[2])
        if self.hold_z is None:
            self.hold_z = z

        # Feed detections only once per inference frame (same reasoning as
        # VisionNavPilot: the control loop runs far faster than inference).
        pose_data = self.data.get("pose")
        gates = []
        if pose_data and pose_data.get("frame_id") != self._last_pose_frame_id:
            self._last_pose_frame_id = pose_data.get("frame_id")
            gates = pose_data["gates"]
        cmd = self.guide.update(gates, pos, vel, quat, yaw, time.monotonic())

        # Landmark fixes: re-associated detections against the guide's own
        # persistent gate map correct the estimator's dead-reckoning drift
        # (same call as rl/fly2.py's live-verified --mode vision --est path).
        if self.guide.last_matches:
            R_wb = spec.quat_to_R(quat)
            for map_p, gate_body in self.guide.last_matches:
                self.estimator.update_landmark(map_p - R_wb @ np.asarray(gate_body))

        self._publish_vio_state(pos, vel, quat)

        now = time.monotonic()
        if now - self._last_status_log >= STATUS_LOG_INTERVAL_S:
            self._last_status_log = now
            print(f"[vio] {cmd.status}", flush=True)
        if self._nav_wr is not None and now - self._last_nav_log >= NAV_LOG_INTERVAL_S:
            self._last_nav_log = now
            dbg = getattr(self.guide, "debug", None) or {}
            self._nav_wr.writerow(
                [
                    round(now, 2),
                    dbg.get("phase", ""),
                    *(
                        None if x is None else round(float(x), 2)
                        for x in (
                            dbg.get("s"),
                            dbg.get("lat"),
                            dbg.get("vert"),
                            dbg.get("dist"),
                        )
                    ),
                    *(round(float(x), 2) for x in pos),
                    cmd.status,
                    self.guide.n_passed,
                ]
            )

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
