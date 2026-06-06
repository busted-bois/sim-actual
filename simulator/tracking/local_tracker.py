import csv
import os
import time

from simulator.tracking.imu_propagator import propagate_position_velocity
from simulator.tracking.snapshot import TrackingSnapshot
from simulator.tracking.vision_correction import apply_gate_yaw_correction
from simulator.tracking.vision_sync import StateRingBuffer, _StateSample

LOCAL_NED_BLEND = 0.2
LOG_DIR = "logs"


class LocalTracker:
    def __init__(self, pitch_up_degrees=20.0, log_csv=True):
        self.pitch_up_degrees = pitch_up_degrees
        self.log_csv = log_csv
        self._armed_prev = False
        self._origin_set = False
        self._last_imu_us = None
        self._last_camera_sim_time_ns = None
        self._state = self._zero_state()
        self._ring = StateRingBuffer()
        self._log_rows: list[dict] = []
        self._status = "idle"

    def _zero_state(self):
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sim_time_ns": 0,
        }

    def tick(self, data):
        armed = bool(data.get("armed"))
        if armed and not self._armed_prev:
            self._set_origin()
        self._armed_prev = armed

        if not self._origin_set:
            self._publish_snapshot(data, "waiting_arm")
            return

        self._integrate_highres_imu(data)
        self._blend_local_position_ned(data)
        self._apply_attitude(data)
        self._apply_vision_correction(data)
        self._push_ring_sample()
        self._publish_snapshot(data, "tracking")

    def _set_origin(self):
        self._state = self._zero_state()
        self._origin_set = True
        self._last_imu_us = None
        self._status = "origin_set"

    def _integrate_highres_imu(self, data):
        imu = data.get("highres_imu")
        if imu is None:
            return

        time_us = int(imu.get("time_boot_us", 0))
        if self._last_imu_us is None:
            self._last_imu_us = time_us
            return

        dt_s = max(0.0, (time_us - self._last_imu_us) * 1e-6)
        self._last_imu_us = time_us
        if dt_s <= 0.0 or dt_s > 0.1:
            return

        accel_body = (
            float(imu["xacc"]),
            float(imu["yacc"]),
            float(imu["zacc"]),
        )
        gyro = (
            float(imu["xgyro"]),
            float(imu["ygyro"]),
            float(imu["zgyro"]),
        )
        self._state.update(
            propagate_position_velocity(self._state, accel_body, gyro, dt_s)
        )
        self._state["sim_time_ns"] = time_us * 1000

    def _blend_local_position_ned(self, data):
        pos = data.get("local_position_ned")
        if pos is None:
            return
        alpha = LOCAL_NED_BLEND
        self._state["x"] = (1.0 - alpha) * self._state["x"] + alpha * float(pos["x"])
        self._state["y"] = (1.0 - alpha) * self._state["y"] + alpha * float(pos["y"])
        self._state["z"] = (1.0 - alpha) * self._state["z"] + alpha * float(pos["z"])
        self._state["vx"] = (1.0 - alpha) * self._state["vx"] + alpha * float(pos["vx"])
        self._state["vy"] = (1.0 - alpha) * self._state["vy"] + alpha * float(pos["vy"])
        self._state["vz"] = (1.0 - alpha) * self._state["vz"] + alpha * float(pos["vz"])

    def _apply_attitude(self, data):
        attitude = data.get("attitude")
        if attitude is None:
            return
        self._state["roll"] = float(attitude["roll"])
        self._state["pitch"] = float(attitude["pitch"])
        self._state["yaw"] = float(attitude["yaw"])

    def _apply_vision_correction(self, data):
        camera = data.get("camera")
        gate_target = data.get("gate_target")
        if camera is None or gate_target is None:
            return

        sim_time_ns = int(camera.get("sim_time_ns", 0))
        if sim_time_ns == self._last_camera_sim_time_ns:
            return
        self._last_camera_sim_time_ns = sim_time_ns

        aligned = self._ring.interpolate_at(sim_time_ns)
        if aligned is not None:
            self._state["x"] = aligned.x
            self._state["y"] = aligned.y
            self._state["z"] = aligned.z
            self._state["vx"] = aligned.vx
            self._state["vy"] = aligned.vy
            self._state["vz"] = aligned.vz
            self._state["roll"] = aligned.roll
            self._state["pitch"] = aligned.pitch
            self._state["yaw"] = aligned.yaw

        self._state["sim_time_ns"] = sim_time_ns
        self._state = apply_gate_yaw_correction(
            self._state, gate_target, self.pitch_up_degrees
        )

    def _push_ring_sample(self):
        self._ring.push(
            _StateSample(
                sim_time_ns=int(self._state["sim_time_ns"]),
                x=self._state["x"],
                y=self._state["y"],
                z=self._state["z"],
                vx=self._state["vx"],
                vy=self._state["vy"],
                vz=self._state["vz"],
                roll=self._state["roll"],
                pitch=self._state["pitch"],
                yaw=self._state["yaw"],
            )
        )

    def _publish_snapshot(self, data, status):
        snapshot = TrackingSnapshot(
            sim_time_ns=int(self._state["sim_time_ns"]),
            x=self._state["x"],
            y=self._state["y"],
            z=self._state["z"],
            vx=self._state["vx"],
            vy=self._state["vy"],
            vz=self._state["vz"],
            roll=self._state["roll"],
            pitch=self._state["pitch"],
            yaw=self._state["yaw"],
            status=status,
        )
        data["tracking_snapshot"] = snapshot
        if self.log_csv and status == "tracking":
            self._log_rows.append(snapshot.as_dict())

    def flush_log(self):
        if not self.log_csv or not self._log_rows:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"tracking_state_{int(time.time())}.csv")
        fieldnames = list(self._log_rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._log_rows)

    def get_snapshot(self):
        return TrackingSnapshot(
            sim_time_ns=int(self._state["sim_time_ns"]),
            x=self._state["x"],
            y=self._state["y"],
            z=self._state["z"],
            vx=self._state["vx"],
            vy=self._state["vy"],
            vz=self._state["vz"],
            roll=self._state["roll"],
            pitch=self._state["pitch"],
            yaw=self._state["yaw"],
            status=self._status,
        )
