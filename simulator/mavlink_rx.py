import struct
import time
import threading

from simulator.config import TrackGate
from simulator.imu_kalman import ImuKalmanPredictor
from simulator.transforms import quat_to_yaw

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2


class MAVLinkRX:
    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False
        self.imu_predictor = ImuKalmanPredictor()
        self.last_imu_time_us = None

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(target=rx.mavlink_receive_loop, daemon=False)
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        """
        while self.is_running:
            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print(
                    "WARNING: ConnectionResetError was thrown. No longer listening to MAVLink port."
                )
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            # --------------------------------------------------------------------------------------
            # HEARTBEAT
            # --------------------------------------------------------------------------------------
            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            # --------------------------------------------------------------------------------------
            # TIMESYNC
            # --------------------------------------------------------------------------------------
            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            # --------------------------------------------------------------------------------------
            # ATTITUDE
            # --------------------------------------------------------------------------------------
            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            # --------------------------------------------------------------------------------------
            # LOCAL_POSITION_NED
            # --------------------------------------------------------------------------------------
            elif msg_type == "LOCAL_POSITION_NED":
                self.on_local_position_ned(msg)

            # --------------------------------------------------------------------------------------
            # ODOMETRY
            # --------------------------------------------------------------------------------------
            elif msg_type == "ODOMETRY":
                self.on_odometry(msg)

            # --------------------------------------------------------------------------------------
            # HIGHRES_IMU
            # --------------------------------------------------------------------------------------
            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # --------------------------------------------------------------------------------------
            # ENCAPSULATED_DATA
            # --------------------------------------------------------------------------------------
            elif msg_type == "ENCAPSULATED_DATA":
                self.on_encapsulated_data(msg)

            # --------------------------------------------------------------------------------------
            # ACTUATOR_OUTPUT_STATUS
            # --------------------------------------------------------------------------------------
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self.on_actuator_output_status(msg)

            # --------------------------------------------------------------------------------------
            # COLLISION
            # --------------------------------------------------------------------------------------
            elif msg_type == "COLLISION":
                self.on_collision(msg)

            # --------------------------------------------------------------------------------------
            # DATA_TRANSMISSION_HANDSHAKE - Repurposed and used for upcoming 'Track Data' packets
            # --------------------------------------------------------------------------------------
            elif msg.get_type() == "DATA_TRANSMISSION_HANDSHAKE":
                track_data_transfer_id = msg.width
                self.track_chunks[track_data_transfer_id] = {}
                self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

    def on_heartbeat(self, msg):
        self.data["armed"] = bool(msg.base_mode & 0b10000000)

    def on_timesync(self, msg):
        pass

    def on_attitude(self, msg):
        self.data["yaw_rad"] = msg.yaw
        self.data["yaw_rate"] = msg.yawspeed
        self.data["att_time_ms"] = msg.time_boot_ms
        self.data["attitude"] = {
            "roll": msg.roll,
            "pitch": msg.pitch,
            "yaw": msg.yaw,
            "roll_speed": msg.rollspeed,
            "pitch_speed": msg.pitchspeed,
            "yaw_speed": msg.yawspeed,
        }

    def on_local_position_ned(self, msg):
        self.data["pos_ned"] = (msg.x, msg.y, msg.z)
        self.data["vel_ned"] = (msg.vx, msg.vy, msg.vz)
        self.data["pos_time_ms"] = msg.time_boot_ms
        self.data["has_position"] = True

    def on_odometry(self, msg):
        self.data["pos_ned"] = (msg.x, msg.y, msg.z)
        self.data["vel_ned"] = (msg.vx, msg.vy, msg.vz)
        qw, qx, qy, qz = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        yaw = quat_to_yaw(qw, qx, qy, qz)
        self.data["yaw_rad"] = yaw
        self.data["yaw_rate"] = msg.yawspeed
        self.data["has_position"] = True
        self.data["odometry"] = {
            "x": msg.x,
            "y": msg.y,
            "z": msg.z,
            "vx": msg.vx,
            "vy": msg.vy,
            "vz": msg.vz,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "roll_speed": msg.rollspeed,
            "pitch_speed": msg.pitchspeed,
            "yaw_speed": msg.yawspeed,
        }

    def on_highres_imu(self, msg):
        # Accel (m/s^2) + gyro (rad/s) in body FRD; consumed by the EKF (Module 5).
        self.data["imu"] = {
            "ax": msg.xacc,
            "ay": msg.yacc,
            "az": msg.zacc,
            "gx": msg.xgyro,
            "gy": msg.ygyro,
            "gz": msg.zgyro,
            "time_us": msg.time_usec,
        }

        if self.last_imu_time_us is None:
            self.last_imu_time_us = msg.time_usec
            return

        dt_s = (msg.time_usec - self.last_imu_time_us) * 1e-6
        self.last_imu_time_us = msg.time_usec
        if dt_s <= 0 or dt_s > 0.5:
            return

        pred = self.imu_predictor.predict(
            accel_body_mps2=(msg.xacc, msg.yacc, msg.zacc),
            gyro_body_rps=(msg.xgyro, msg.ygyro, msg.zgyro),
            dt_s=dt_s,
        )
        if pred is None:
            return

        self.data["imu_prediction"] = {
            "dt_s": pred.dt_s,
            "roll_rad": pred.roll_rad,
            "pitch_rad": pred.pitch_rad,
            "yaw_rad": pred.yaw_rad,
            "pos_ned": pred.pos_ned,
            "vel_ned": pred.vel_ned,
            "accel_ned": pred.accel_ned,
            "covariance_trace": pred.covariance_trace,
        }

        # IMU-only attitude estimate (high-rate). No GPS/camera correction yet.
        self.data["attitude_imu"] = {
            "roll": pred.roll_rad,
            "pitch": pred.pitch_rad,
            "yaw": pred.yaw_rad,
        }
        self.data["yaw_rad_imu"] = pred.yaw_rad

    def on_encapsulated_data(self, msg):
        if msg:
            raw_payload = bytes(msg.data)
            data_type = raw_payload[0]

            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        (
            data_type,
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time,
        ) = struct.unpack_from("<BQqqIq", raw_payload)
        self.data["active_gate_index"] = active_gate_index
        self.data["race_started"] = race_start_boot_time_ms >= 0
        self.data["race_finish_time_ns"] = race_finish_time_ns
        self.data["race_status"] = {
            "active_gate_index": active_gate_index,
            "race_start_boot_time_ms": race_start_boot_time_ms,
            "sim_boot_time_ms": sim_boot_time_ms,
        }

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        # header:
        #   data_type - ID of this message
        #   transfer_id - ID of the group of packets this chunk belongs to
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        if (
            len(self.track_chunks[transfer_id])
            == self.expected_num_track_chunks[transfer_id]
        ):
            full_payload = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full_payload = full_payload + self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        (num_gates,) = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for i in range(num_gates):
            (
                gate_id,
                position_ned_x,
                position_ned_y,
                position_ned_z,
                orientation_ned_w,
                orientation_ned_x,
                orientation_ned_y,
                orientation_ned_z,
                width,
                height,
            ) = struct.unpack_from("<Hfffffffff", payload)
            gates.append(
                TrackGate(
                    gate_id=gate_id,
                    pos_ned=(position_ned_x, position_ned_y, position_ned_z),
                    orient_quat=(
                        orientation_ned_w,
                        orientation_ned_x,
                        orientation_ned_y,
                        orientation_ned_z,
                    ),
                    width_m=width,
                    height_m=height,
                )
            )
            payload = payload[38:]
        self.data["gates"] = gates
        self.data["track_gates"] = [
            {
                "position_ned": g.pos_ned,
                "orientation_ned": g.orient_quat,
                "width": g.width_m,
                "height": g.height_m,
            }
            for g in gates
        ]

    def on_actuator_output_status(self, msg):
        pass

    def on_collision(self, msg):
        self.data["last_collision"] = (
            msg.id,
            msg.threat_level,
            msg.horizontal_minimum_delta,
        )
        self.data["collision"] = {
            "id": msg.id,
            "threat_level": msg.threat_level,
            "delta": msg.horizontal_minimum_delta,
        }
