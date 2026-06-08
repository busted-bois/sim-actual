"""Pilot — attitude-mode gate racer with altitude PID.

Called at ~250 Hz by controller.update(). Uses ATTITUDE mode with pitch_rate
for forward motion and an altitude PID for thrust control. Reads shared_data
(written by mavlink_rx and vision_rx) and sets controller commands directly.
"""

from __future__ import annotations

import math
import time as _time

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
HOVER_THRUST = 0.5
CRUISE_THRUST = 0.55
CRUISE_PITCH_RATE = -0.2
COLLISION_THRUST = 0.4
COLLISION_HOLD_S = 2.0

ALTITUDE_TRIM = 0.55
KP_Z = 0.15
KI_Z = 0.01
KD_Z = 0.20
Z_TARGET_NED = -5.0

VISION_YAW_GAIN = math.radians(40)
VISION_CENTER_DEADBAND = 0.14
VISION_PROXIMITY_R_FRAC = 0.10
VISION_MAX_AGE_S = 0.5
VISION_VY_GAIN = 6.0
VISION_MAX_ALT_ADJUST = 2.0
STABILIZE_HOLD_S = 0.3
VISION_ALIGN_PITCH_RATE = -0.15

TELEMETRY_YAW_GAIN = 1.0
TELEMETRY_PROXIMITY_M = 3.0

OBSTACLE_CLEAR_ZONE = 0.25

CONTROL_DT_S = 1 / 250

_LOG_INTERVAL = 2.0
_DBG_INTERVAL = 0.5


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class Pilot:
    """Gate-traversal pilot using ATTITUDE mode + altitude PID."""

    def __init__(self, controller, data):  # type: ignore[type-arg]
        self.controller = controller
        self.data = data
        self._z_integral = 0.0
        self._last_z_target: float | None = None
        self._collision_time: float | None = None
        self._stabilize_start: float | None = None
        self._advancing: bool = False
        self._last_gate_id: str | None = None
        self._last_log = 0.0
        self._dbg_timer = 0.0
        self._mode_str = "???"
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_THRUST)
        print("[pilot] init done, waiting for armed + track_gates", flush=True)

    def _log_status(self, reason: str) -> None:
        now = _time.monotonic()
        if now - self._last_log < _LOG_INTERVAL:
            return
        self._last_log = now
        armed = self.data.get("armed", False)
        tg = self.data.get("track_gates")
        odo = self.data.get("odometry")
        gt = self.data.get("gate_target")
        col = self.data.get("collision")
        z = odo.get("z", "?") if odo else "?"
        n_gates = len(tg) if tg else 0
        gt_det = gt.get("detected") if gt else None
        c = self.controller
        print(
            f"[pilot] {reason:25s} | mode={self._mode_str:8s} | "
            f"armed={armed} gates={n_gates} z={z} "
            f"pitch={c._pitch_rate:+.3f} yaw_r={c._yaw_rate:+.3f} "
            f"thr={c._thrust:.3f} ctrl={c.control_mode} | "
            f"gt_det={gt_det} collision={col is not None}",
            flush=True,
        )

    def _dbg(self, msg: str) -> None:
        """Print debug message with 0.5s throttle."""
        now = _time.monotonic()
        if now - self._dbg_timer < _DBG_INTERVAL:
            return
        self._dbg_timer = now
        print(f"[pilot_dbg] {msg}", flush=True)

    # ------------------------------------------------------------------
    # Gate selection
    # ------------------------------------------------------------------
    def _find_nearest_gate(self, track_gates: list, odometry: dict) -> dict | None:  # type: ignore[type-arg]
        if not odometry or not track_gates:
            return None
        ox, oy, oz = odometry.get("x", 0), odometry.get("y", 0), odometry.get("z", 0)
        best_gate = None
        best_dist = float("inf")
        gates_checked = 0
        for gate in track_gates:
            pos = gate.get("position_ned")
            if not pos or len(pos) < 3:
                continue
            gates_checked += 1
            dx = pos[0] - ox
            dy = pos[1] - oy
            dz = pos[2] - oz
            dist = dx * dx + dy * dy + dz * dz
            if dist < best_dist:
                best_dist = dist
                best_gate = gate

        if best_gate:
            pos = best_gate.get("position_ned", [0, 0, 0])
            self._dbg(
                f"find_nearest gates_checked={gates_checked} best_dist={best_dist:.2f} "
                f"selected_pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
            )

        return best_gate

    def _reset_approach_state(self) -> None:
        self._advancing = False
        self._stabilize_start = None

    def _gate_id(self, gate: dict) -> str | None:  # type: ignore[type-arg]
        pos = gate.get("position_ned")
        if not pos or len(pos) < 3:
            return None
        return f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"

    # ------------------------------------------------------------------
    # Main tick — called every cycle at 250 Hz
    # ------------------------------------------------------------------
    def tick(self) -> None:
        armed = self.data.get("armed", False)

        if not armed:
            self._mode_str = "disarmed"
            self._hover()
            self._log_status("not armed")
            return

        odo = self.data.get("odometry")
        z = odo.get("z", "?") if odo else "?"
        tg = self.data.get("track_gates")
        gt = self.data.get("gate_target")
        gt_det = gt.get("detected") if gt else None
        c = self.controller
        col = self.data.get("collision")
        self._dbg(
            f"tick state={self._mode_str} armed={armed} gates={len(tg) if tg else 0} "
            f"z={z} pitch={c._pitch_rate:+.3f} thrust={c._thrust:.3f} "
            f"gt_det={gt_det} advancing={self._advancing} "
            f"stabilize_start={self._stabilize_start} collision={col is not None}"
        )

        # Collision hold
        if self._collision_time is not None:
            elapsed = _time.monotonic() - self._collision_time
            if elapsed < COLLISION_HOLD_S:
                self.controller.set_control_mode("attitude")
                self.controller.set_attitude_rates(0, 0, 0, COLLISION_THRUST)
                self._mode_str = "collision_hold"
                self._log_status("collision hold")
                return
            self._collision_time = None
            self.data.pop("collision", None)

        collision = self.data.get("collision")
        if collision is not None:
            self._collision_time = _time.monotonic()

        track_gates = self.data.get("track_gates")

        if not track_gates:
            self._mode_str = "no_gates"
            self._reset_approach_state()
            self._hover()
            self._log_status("no track_gates")
            return

        # Gate from vision
        gate_target = self.data.get("gate_target")
        cam = self.data.get("camera")
        if gate_target and gate_target.get("detected"):
            if cam is not None:
                age = _time.monotonic() - cam.get("received_at", 0)
                if age < VISION_MAX_AGE_S:
                    self._mode_str = "vision"
                    self._fly_toward_gate_vision(gate_target)
                    self._log_status("vision gate")
                    return

        # Gate from telemetry — find nearest by 3D distance
        odometry = self.data.get("odometry")
        if odometry is not None:
            nearest = self._find_nearest_gate(track_gates, odometry)
            if nearest is not None:
                gid = self._gate_id(nearest)
                if gid != self._last_gate_id:
                    self._reset_approach_state()
                    old_gid = self._last_gate_id
                    self._last_gate_id = gid
                    print(
                        f"[pilot_dbg] state_transition: gate_id changed {old_gid} -> {gid}",
                        flush=True,
                    )
                self._mode_str = "telemetry"
                self._fly_toward_gate_telemetry(nearest, odometry)
                self._log_status("telemetry gate")
                return

        self._mode_str = "no_target"
        self._reset_approach_state()
        self._hover()
        self._log_status("no target")

    # ------------------------------------------------------------------
    # Flight primitives
    # ------------------------------------------------------------------
    def _hover(self, z_target: float | None = None) -> None:
        thrust = self._altitude_thrust(HOVER_THRUST, z_target)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, thrust)

    def _cruise_forward(self) -> None:
        thrust = self._altitude_thrust(CRUISE_THRUST)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, CRUISE_PITCH_RATE, 0, thrust)

    def _fly_toward_gate_vision(self, gate_target: dict) -> None:  # type: ignore[type-arg]
        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        r_frac = gate_target.get("r_frac", 0.0)

        yaw_rate = _clamp(VISION_YAW_GAIN * nx, -2.0, 2.0)

        ny_offset = _clamp(
            ny * VISION_VY_GAIN, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
        )
        odometry = self.data.get("odometry")
        z_now = odometry.get("z", 0.0) if odometry else 0.0

        centered = abs(nx) < VISION_CENTER_DEADBAND and abs(ny) < VISION_CENTER_DEADBAND
        z_target = z_now + ny_offset

        if self._advancing:
            # ADVANCE phase — flying forward through gate
            if not centered:
                # Lost centering → back to stabilize
                self._advancing = False
                self._stabilize_start = None
                print(
                    "[pilot_dbg] state_transition: advancing=False (lost centering)",
                    flush=True,
                )
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif r_frac >= VISION_PROXIMITY_R_FRAC:
                # Very close — stop pitching, fine-tune altitude+yaw only
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            else:
                obstacles = self.data.get("obstacles", [])
                obstacle_blocking = any(
                    abs(o["nx"]) < OBSTACLE_CLEAR_ZONE and o["r_frac"] > 0.005
                    for o in obstacles
                )
                if obstacle_blocking:
                    nearest_obs = min(obstacles, key=lambda o: abs(o["nx"]))
                    yaw_rate = _clamp(-nearest_obs["nx"] * 2.0, -1.0, 1.0)
                    pitch = 0.0
                    thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
                else:
                    alignment = max(0.0, 1.0 - abs(nx))
                    pitch = CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment)
                    thrust = self._altitude_thrust(CRUISE_THRUST, z_target=z_target)
        else:
            # STABILIZE phase — hover, align yaw+altitude only
            pitch = 0.0
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)

            if centered:
                if self._stabilize_start is None:
                    self._stabilize_start = _time.monotonic()
                    print(
                        "[pilot_dbg] state_transition: stabilize_start set", flush=True
                    )
                elif _time.monotonic() - self._stabilize_start >= STABILIZE_HOLD_S:
                    # Held center long enough → advance
                    self._advancing = True
                    old_stabilize_start = self._stabilize_start
                    self._stabilize_start = None
                    print(
                        f"[pilot_dbg] state_transition: advancing=True (held center for {old_stabilize_start}s)",
                        flush=True,
                    )
            else:
                # Not centered — reset hold timer
                if self._stabilize_start is not None:
                    print(
                        "[pilot_dbg] state_transition: stabilize_start cleared (not centered)",
                        flush=True,
                    )
                    self._stabilize_start = None

        phase = "ADVANCE" if self._advancing else "STABILIZE"
        self._dbg(
            f"vision ny={ny:+.3f} nx={nx:+.3f} r_frac={r_frac:.3f} ny_offset={ny_offset:+.3f} "
            f"z_target={z_target:.2f} centered={centered} phase={phase} "
            f"obstacle_blocking={obstacle_blocking if self._advancing else 'N/A'} pitch={pitch:+.3f}"
        )

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    def _fly_toward_gate_telemetry(self, gate: dict, odometry: dict) -> None:  # type: ignore[type-arg]
        gx, gy, gz = gate["position_ned"]
        ox, oy, oz = odometry["x"], odometry["y"], odometry.get("z", 0.0)

        dx = gx - ox
        dy = gy - oy
        dist = math.sqrt(dx * dx + dy * dy)

        # Yaw from quaternion in odometry
        qw = odometry.get("qw", 1.0)
        qx = odometry.get("qx", 0.0)
        qy = odometry.get("qy", 0.0)
        qz = odometry.get("qz", 0.0)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Bearing error: angle from drone heading to gate direction
        bearing_to_gate = math.atan2(dy, dx)
        bearing_error = bearing_to_gate - yaw
        # Normalize to [-pi, pi]
        bearing_error = (bearing_error + math.pi) % (2 * math.pi) - math.pi

        yaw_rate = _clamp(TELEMETRY_YAW_GAIN * bearing_error, -2.0, 2.0)

        # Normalized horizontal alignment: 0 = perfectly aligned, 1 = 180 deg off
        nx_telemetry = bearing_error / math.pi  # [-1, 1]
        # Normalized vertical alignment
        ny_telemetry = _clamp((gz - oz) / 5.0, -1, 1)

        centered = abs(bearing_error) < 0.2 and abs(ny_telemetry) < 0.3

        if self._advancing:
            # ADVANCE phase — flying forward toward gate
            if abs(bearing_error) > 0.5:
                # Lost heading → back to stabilize
                self._advancing = False
                self._stabilize_start = None
                print(
                    "[pilot_dbg] state_transition: advancing=False (lost heading)",
                    flush=True,
                )
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
            elif dist < TELEMETRY_PROXIMITY_M:
                # Very close — stop pitching, fine-tune yaw+altitude
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
            else:
                obstacles = self.data.get("obstacles", [])
                obstacle_blocking = any(
                    abs(o["nx"]) < OBSTACLE_CLEAR_ZONE and o["r_frac"] > 0.005
                    for o in obstacles
                )
                if obstacle_blocking:
                    nearest_obs = min(obstacles, key=lambda o: abs(o["nx"]))
                    yaw_rate = _clamp(-nearest_obs["nx"] * 2.0, -1.0, 1.0)
                    pitch = 0.0
                    thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
                else:
                    alignment = max(0.0, 1.0 - abs(nx_telemetry))
                    pitch = CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment)
                    thrust = self._altitude_thrust(CRUISE_THRUST, z_target=gz)
        else:
            # STABILIZE phase — hover, align yaw+altitude only
            pitch = 0.0
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)

            if centered:
                if self._stabilize_start is None:
                    self._stabilize_start = _time.monotonic()
                    print(
                        "[pilot_dbg] state_transition: stabilize_start set", flush=True
                    )
                elif _time.monotonic() - self._stabilize_start >= STABILIZE_HOLD_S:
                    # Held heading long enough → advance
                    self._advancing = True
                    old_stabilize_start = self._stabilize_start
                    self._stabilize_start = None
                    print(
                        f"[pilot_dbg] state_transition: advancing=True (held heading for {old_stabilize_start}s)",
                        flush=True,
                    )
            else:
                # Not centered — reset hold timer
                if self._stabilize_start is not None:
                    print(
                        "[pilot_dbg] state_transition: stabilize_start cleared (not centered)",
                        flush=True,
                    )
                    self._stabilize_start = None

        phase = "ADVANCE" if self._advancing else "STABILIZE"
        self._dbg(
            f"telemetry bearing_err={bearing_error:+.3f} dist={dist:.2f} centered={centered} "
            f"phase={phase} obstacle_blocking={obstacle_blocking if self._advancing else 'N/A'} pitch={pitch:+.3f}"
        )

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    # ------------------------------------------------------------------
    # Altitude PID
    # ------------------------------------------------------------------
    def _altitude_thrust(self, fallback: float, z_target: float | None = None) -> float:
        odometry = self.data.get("odometry")
        if odometry is None:
            return fallback

        z = odometry.get("z", 0.0)
        vz = odometry.get("vz", 0.0)
        target = z_target if z_target is not None else Z_TARGET_NED

        # Reset integral when target changes significantly
        if self._last_z_target is not None and abs(target - self._last_z_target) > 2.0:
            self._z_integral = 0.0
        self._last_z_target = target

        error = z - target
        self._z_integral += error * CONTROL_DT_S
        # Anti-windup
        self._z_integral = _clamp(self._z_integral, -0.5, 0.5)

        raw_thrust = ALTITUDE_TRIM + KP_Z * error + KI_Z * self._z_integral + KD_Z * vz
        clamped_thrust = _clamp(raw_thrust, 0.0, 1.0)

        self._dbg(
            f"altitude z={z:.2f} z_target={target:.2f} error={error:+.3f} "
            f"z_integral={self._z_integral:+.3f} raw_thrust={raw_thrust:.3f} clamped_thrust={clamped_thrust:.3f}"
        )

        return clamped_thrust
