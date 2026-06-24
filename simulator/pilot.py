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
CRUISE_PITCH_RATE = -0.16
TELEM_PITCH_RATE = -0.07
MAX_PITCH_RATE = 0.14
COLLISION_THRUST = 0.4
COLLISION_HOLD_S = 2.0

ALTITUDE_TRIM = 0.55
KP_Z = 0.15
KI_Z = 0.01
KD_Z = 0.20
Z_TARGET_NED = -5.0

VISION_YAW_GAIN = math.radians(40)
VISION_CENTER_DEADBAND = 0.30
VISION_PROXIMITY_R_FRAC = 0.10
VISION_MAX_AGE_S = 0.5
VISION_VY_GAIN = 6.0
VISION_MAX_ALT_ADJUST = 2.0
STABILIZE_HOLD_S = 0.1
VISION_ALIGN_PITCH_RATE = -0.15

TELEMETRY_YAW_GAIN = 1.0
TELEMETRY_PROXIMITY_M = 3.0

OBSTACLE_CLEAR_ZONE = 0.25
AMBIGUOUS_YAW_GAIN = math.radians(15)

POST_GATE_HOVER_S = 2.5
POST_GATE_VISION_COOLDOWN_S = 4.0
FORCE_TELEM_AFTER_PASS_S = 3.0
VISION_MAX_NX = 0.52
VISION_MAX_NY = 0.95

# Search mode — yaw scan when no gate map / vision
SEARCH_SWEEP_YAW_RATE = 0.8
SEARCH_SWEEP_PERIOD_S = 2.0
SEARCH_FORWARD_PITCH = -0.04
SEARCH_WARMUP_S = 0.5

# Passed-gate rejection — only ignore vision toward gates already flown through
PASSED_GATE_NEAR_M = 1.5
PASSED_GATE_ANGLE_RAD = math.radians(35)

CONTROL_DT_S = 1 / 250


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
        self._peak_r_frac: float = 0.0
        self._post_gate_time: float | None = None
        self._last_gate_id: str | None = None
        self._completed_gates: set[str] = set()
        self._vision_suppress_until: float = 0.0
        self._passed_gate_positions: list[tuple[float, float, float]] = []
        self._gates_passed: int = 0
        self._searching: bool = False
        self._search_yaw_dir: float = 1.0
        self._search_start_time: float | None = None
        self._course_gate_index: int = 0
        self._force_telem_until: float = 0.0
        self._telem_min_dist: float = float("inf")
        self._vision_cooldown_until: float = 0.0
        self._mode_str = "???"
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_THRUST)
        print("[pilot] init done, waiting for armed + vision/telemetry", flush=True)

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
            gid = self._gate_id(gate)
            if gid in self._completed_gates:
                continue
            gates_checked += 1
            dx = pos[0] - ox
            dy = pos[1] - oy
            dz = pos[2] - oz
            dist = dx * dx + dy * dy + dz * dz
            if dist < best_dist:
                best_dist = dist
                best_gate = gate

        return best_gate

    def _reset_approach_state(self) -> None:
        self._advancing = False
        self._stabilize_start = None
        self._post_gate_time = None
        self._peak_r_frac = 0.0
        self._vision_suppress_until = 0.0
        self._telem_min_dist = float("inf")

    def _mark_gate_passed(self) -> str | None:
        self._gates_passed += 1
        drone_pos = self._get_position()
        if drone_pos is not None:
            self._passed_gate_positions.append(drone_pos)

        gate_id = None
        course_gate = self._get_course_gate()
        if course_gate is not None:
            gate_id = self._gate_id(course_gate)
        if gate_id:
            self._completed_gates.add(gate_id)
        self._course_gate_index += 1
        self._force_telem_until = _time.monotonic() + FORCE_TELEM_AFTER_PASS_S
        self._vision_cooldown_until = _time.monotonic() + POST_GATE_VISION_COOLDOWN_S
        self._reset_approach_state()
        self._post_gate_time = _time.monotonic()

        print(
            f"[pilot] Gate {gate_id} marked COMPLETE, "
            f"total passed: {self._gates_passed}",
            flush=True,
        )
        print(
            f"[pilot] FLY-THROUGH at pos={drone_pos}, hovering to re-acquire",
            flush=True,
        )
        return gate_id

    def _cruise_pitch(self) -> float:
        return CRUISE_PITCH_RATE if self._gates_passed == 0 else TELEM_PITCH_RATE

    def _clamp_pitch(self, pitch: float) -> float:
        return max(pitch, -MAX_PITCH_RATE)

    def _pitch_scale(self, r_frac: float) -> float:
        if r_frac > 0.18:
            return 0.30
        if r_frac > 0.10:
            return 0.50
        if r_frac > 0.06:
            return 0.70
        return 1.0

    def _gate_id(self, gate: dict) -> str | None:  # type: ignore[type-arg]
        gid = gate.get("gate_id")
        if gid is not None:
            return str(gid)
        pos = gate.get("position_ned")
        if not pos or len(pos) < 3:
            return None
        return f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"

    def _get_course_gate(self) -> dict | None:  # type: ignore[type-arg]
        track_gates = self.data.get("track_gates") or []
        while self._course_gate_index < len(track_gates):
            gate = track_gates[self._course_gate_index]
            gid = self._gate_id(gate)
            if gid and gid in self._completed_gates:
                self._course_gate_index += 1
                continue
            return gate
        odometry = self.data.get("odometry")
        if odometry is not None:
            return self._find_nearest_gate(track_gates, odometry)
        return None

    def _vision_usable(self, gate_target: dict) -> bool:  # type: ignore[type-arg]
        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        if abs(nx) > VISION_MAX_NX:
            return False
        # First gate often appears below center — pilot corrects altitude from ny.
        if self._gates_passed > 0 and abs(ny) > VISION_MAX_NY:
            return False
        if self._is_passed_gate(nx):
            return False
        if gate_target.get("detected"):
            return True
        if gate_target.get("raw_detected"):
            streak = gate_target.get("temporal_streak", 0)
            r_frac = gate_target.get("r_frac", 0.0)
            if streak >= 2 and abs(nx) < VISION_MAX_NX:
                if r_frac > 0.04 or gate_target.get("gate_confidence", 0.0) >= 0.20:
                    return True
        return False

    def _get_position(self) -> tuple[float, float, float] | None:
        odometry = self.data.get("odometry")
        if odometry is not None:
            return (odometry["x"], odometry["y"], odometry["z"])
        pos_ned = self.data.get("pos_ned")
        if pos_ned is not None and len(pos_ned) >= 3:
            return (pos_ned[0], pos_ned[1], pos_ned[2])
        return None

    def _is_passed_gate(self, nx: float) -> bool:
        """True if detection bearing points back at a gate we already flew through."""
        drone_pos = self._get_position()
        if drone_pos is None or not self._passed_gate_positions:
            return False

        yaw = self.data.get("yaw_rad", 0.0)
        detected_bearing = yaw + math.atan(nx)

        for pgx, pgy, _pgz in self._passed_gate_positions:
            dx = pgx - drone_pos[0]
            dy = pgy - drone_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.5:
                continue

            passed_dir = math.atan2(dy, dx)
            diff = detected_bearing - passed_dir
            diff = (diff + math.pi) % (2 * math.pi) - math.pi

            if abs(diff) < PASSED_GATE_ANGLE_RAD and dist < PASSED_GATE_NEAR_M:
                return True

        return False

    def _telemetry_target(self) -> tuple[dict, dict] | None:  # type: ignore[type-arg]
        odometry = self.data.get("odometry")
        if odometry is None:
            return None
        gate = self._get_course_gate()
        if gate is None:
            return None
        return gate, odometry

    def _do_search(self) -> None:
        now = _time.monotonic()
        elapsed = now - (self._search_start_time or now)
        period_count = int(elapsed / SEARCH_SWEEP_PERIOD_S)
        new_dir = 1.0 if period_count % 2 == 0 else -1.0

        if new_dir != self._search_yaw_dir:
            self._search_yaw_dir = new_dir
            print(
                f"[pilot] SEARCH sweep dir={'CW' if new_dir > 0 else 'CCW'} "
                f"elapsed={elapsed:.1f}s gates_passed={self._gates_passed}",
                flush=True,
            )

        yaw_rate = SEARCH_SWEEP_YAW_RATE * self._search_yaw_dir
        drone_pos = self._get_position()
        z_hold = drone_pos[2] if drone_pos else None
        thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_hold)
        pitch = 0.0 if elapsed < SEARCH_WARMUP_S else SEARCH_FORWARD_PITCH
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    # ------------------------------------------------------------------
    # Main tick — called every cycle at 250 Hz
    # ------------------------------------------------------------------
    def tick(self) -> None:
        armed = self.data.get("armed", False)

        if not armed:
            self._mode_str = "disarmed"
            self._hover()
            return

        # Collision hold
        if self._collision_time is not None:
            elapsed = _time.monotonic() - self._collision_time
            if elapsed < COLLISION_HOLD_S:
                self.controller.set_control_mode("attitude")
                self.controller.set_attitude_rates(0, 0, 0, COLLISION_THRUST)
                self._mode_str = "collision_hold"
                return
            self._collision_time = None
            self.data.pop("collision", None)

        collision = self.data.get("collision")
        if collision is not None:
            self._collision_time = _time.monotonic()

        # Post-gate hover — stop and re-acquire next gate after passing through
        if self._post_gate_time is not None:
            elapsed = _time.monotonic() - self._post_gate_time
            if elapsed < POST_GATE_HOVER_S:
                self._mode_str = "post_gate_hover"
                drone_pos = self._get_position()
                self._hover(drone_pos[2] if drone_pos else None)
                return
            else:
                self._post_gate_time = None
                target = self._telemetry_target()
                if target is not None:
                    self._searching = False
                    self._search_start_time = None
                    print(
                        "[pilot] POST-GATE hover done, telemetry to next gate",
                        flush=True,
                    )
                else:
                    self._searching = True
                    self._search_start_time = _time.monotonic()
                    self._search_yaw_dir = 1.0
                    print(
                        "[pilot] POST-GATE hover done, waiting for gate map",
                        flush=True,
                    )

        telem = self._telemetry_target()
        force_telem = _time.monotonic() < self._force_telem_until
        vision_cooldown = _time.monotonic() < self._vision_cooldown_until

        gate_target = self.data.get("gate_target")
        cam = self.data.get("camera")
        vision_ok = False
        if (
            not vision_cooldown
            and gate_target
            and cam is not None
        ):
            age = _time.monotonic() - cam.get("received_at", 0)
            if age < VISION_MAX_AGE_S and self._vision_usable(gate_target):
                vision_ok = True

        # Post-pass recovery: map steering only, no vision chase
        if vision_cooldown and self._gates_passed > 0:
            if telem is not None:
                nearest, odometry = telem
                gid = self._gate_id(nearest)
                if gid != self._last_gate_id:
                    self._reset_approach_state()
                    self._last_gate_id = gid
                    print(
                        f"[pilot] NEW TARGET gate {gid} "
                        f"(course {self._course_gate_index + 1}/"
                        f"{len(self.data.get('track_gates') or [])})",
                        flush=True,
                    )
                self._mode_str = "telemetry_recover"
                self._fly_toward_gate_telemetry(nearest, odometry)
                return
            self._mode_str = "post_pass_hover"
            drone_pos = self._get_position()
            self._hover(drone_pos[2] if drone_pos else None)
            return

        # Gate 1: always prefer vision over map (map arrives mid-countdown)
        if (
            self._gates_passed == 0
            and not force_telem
            and vision_ok
        ):
            if self._searching:
                self._searching = False
                self._search_start_time = None
            self._mode_str = "vision"
            self._fly_toward_gate_vision(gate_target)
            return

        # Post-pass: telemetry-first to reach next gate
        if force_telem and telem is not None and self._gates_passed > 0:
            nearest, odometry = telem
            gid = self._gate_id(nearest)
            if gid != self._last_gate_id:
                self._reset_approach_state()
                self._last_gate_id = gid
                print(
                    f"[pilot] NEW TARGET gate {gid} "
                    f"(course {self._course_gate_index + 1}/"
                    f"{len(self.data.get('track_gates') or [])})",
                    flush=True,
                )
            self._searching = False
            self._search_start_time = None
            self._mode_str = "telemetry"
            self._fly_toward_gate_telemetry(nearest, odometry)
            return

        if not force_telem and vision_ok:
            if self._searching:
                self._searching = False
                self._search_start_time = None
                nx = gate_target.get("nx", 0.0)
                print(
                    f"[pilot] SEARCH → vision gate (nx={nx:+.3f})",
                    flush=True,
                )
            self._mode_str = "vision"
            self._fly_toward_gate_vision(gate_target)
            return

        if (
            gate_target
            and gate_target.get("raw_detected")
            and gate_target.get("ambiguous")
            and telem is None
            and self._vision_usable(gate_target)
        ):
            self._mode_str = "vision_ambiguous"
            self._fly_ambiguous(gate_target)
            return

        if self._searching and telem is None:
            self._mode_str = "search"
            self._do_search()
            return

        # Telemetry — only after gate 1 (gate 0 is vision-only)
        if telem is not None and self._gates_passed > 0:
            nearest, odometry = telem
            gid = self._gate_id(nearest)
            if gid != self._last_gate_id:
                self._reset_approach_state()
                self._last_gate_id = gid
                print(
                    f"[pilot] NEW TARGET gate {gid} "
                    f"(course {self._course_gate_index + 1}/"
                    f"{len(self.data.get('track_gates') or [])})",
                    flush=True,
                )
            if self._searching:
                self._searching = False
                self._search_start_time = None
            self._mode_str = "telemetry"
            self._fly_toward_gate_telemetry(nearest, odometry)
            return

        if self._searching:
            self._mode_str = "search"
            self._do_search()
            return

        self._mode_str = "no_target"
        self._reset_approach_state()
        self._hover()

    # ------------------------------------------------------------------
    # Flight primitives
    # ------------------------------------------------------------------
    def _obstacle_threats(self, r_frac: float = 0.0) -> list[dict]:  # type: ignore[type-arg]
        from simulator.config import OBSTACLE_CONFIDENCE_AVOID

        gate = self.data.get("gate_target") or {}
        if gate.get("raw_detected") and gate.get("temporal_streak", 0) >= 1:
            return []

        # Close approach: gate fills frame; center blobs are the gate opening.
        if r_frac > 0.04:
            return []
        gnx = gate.get("nx", 0.0)
        gny = gate.get("ny", 0.0)
        gate_active = gate.get("raw_detected") or gate.get("detected")
        gate_span = max(gate.get("r_frac", 0.0) ** 0.5, 0.12)

        threats = []
        for o in self.data.get("obstacles", []):
            if (
                o.get("confidence", 0) >= OBSTACLE_CONFIDENCE_AVOID
                and abs(o.get("nx", 0)) < OBSTACLE_CLEAR_ZONE
                and o.get("r_frac", 0) > 0.005
            ):
                if gate_active:
                    onx = o.get("nx", 0.0)
                    ony = o.get("ny", 0.0)
                    if (
                        abs(onx - gnx) < gate_span
                        and abs(ony - gny) < gate_span * 1.5
                    ):
                        continue
                threats.append(o)
        return threats

    def _avoid_obstacles(
        self,
        yaw_rate: float,
        z_target: float,
        fallback_thrust: float,
        r_frac: float = 0.0,
    ) -> tuple[float, float, float]:
        """Safety-first: stop forward motion when obstacles block center path."""
        threats = self._obstacle_threats(r_frac)
        if not threats:
            return yaw_rate, 0.0, fallback_thrust

        nearest = max(threats, key=lambda o: o.get("confidence", 0) * o.get("r_frac", 0))
        conf = nearest.get("confidence", 0.5)
        steer = _clamp(-nearest["nx"] * (1.5 + conf), -1.0, 1.0)
        yaw_rate = _clamp(yaw_rate * 0.3 + steer * 0.7, -1.0, 1.0)
        print(
            f"[pilot] OBSTACLE avoid conf={conf:.2f} nx={nearest.get('nx', 0):+.3f}",
            flush=True,
        )
        return yaw_rate, 0.0, self._altitude_thrust(HOVER_THRUST, z_target=z_target)

    def _fly_ambiguous(self, gate_target: dict) -> None:  # type: ignore[type-arg]
        """Conservative when gate vs obstacle classification is uncertain."""
        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        conf = gate_target.get("gate_confidence", 0.0)

        odometry = self.data.get("odometry")
        z_now = odometry.get("z", 0.0) if odometry else 0.0
        ny_offset = _clamp(
            ny * VISION_VY_GAIN * 0.5, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
        )
        z_target = z_now + ny_offset

        yaw_rate = _clamp(AMBIGUOUS_YAW_GAIN * nx * conf, -0.8, 0.8)
        yaw_rate, pitch, thrust = self._avoid_obstacles(
            yaw_rate, z_target, HOVER_THRUST, gate_target.get("r_frac", 0.0)
        )
        if pitch == 0.0 and not self._obstacle_threats(gate_target.get("r_frac", 0.0)):
            # Unidentified center blob — treat as potential obstacle, no forward pitch.
            if abs(nx) < OBSTACLE_CLEAR_ZONE and gate_target.get("r_frac", 0) > 0.01:
                yaw_rate = _clamp(-nx * 1.0, -0.6, 0.6)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    def _hover(self, z_target: float | None = None) -> None:
        thrust = self._altitude_thrust(HOVER_THRUST, z_target)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, thrust)

    def _cruise_forward(self) -> None:
        thrust = self._altitude_thrust(CRUISE_THRUST)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, self._cruise_pitch(), 0, thrust)

    def _fly_toward_gate_vision(self, gate_target: dict) -> None:  # type: ignore[type-arg]
        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        r_frac = gate_target.get("r_frac", 0.0)
        gate_conf = min(gate_target.get("gate_confidence", 1.0), 0.55)
        cruise = self._cruise_pitch()
        p_scale = self._pitch_scale(r_frac)

        yaw_rate = _clamp(VISION_YAW_GAIN * nx * gate_conf, -1.2, 1.2)

        ny_offset = _clamp(
            ny * VISION_VY_GAIN, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
        )
        odometry = self.data.get("odometry")
        z_now = odometry.get("z", 0.0) if odometry else 0.0

        aligned_x = abs(nx) < 0.22
        z_target = z_now + ny_offset
        eff_conf = gate_conf
        allow_advance = self._gates_passed == 0 and abs(nx) < 0.22

        self._peak_r_frac = max(self._peak_r_frac, r_frac if abs(nx) < 0.35 else self._peak_r_frac)

        if (
            allow_advance
            and self._peak_r_frac > 0.08
            and r_frac < self._peak_r_frac * 0.55
            and abs(nx) < 0.30
        ):
            self._mark_gate_passed()
            pitch = self._clamp_pitch(abs(cruise) * 0.3)
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
            return

        if allow_advance and not self._advancing and r_frac > 0.05 and abs(nx) < 0.22:
            self._advancing = True
            self._stabilize_start = None
            print("[pilot] ADVANCE → gate aligned, pitching forward", flush=True)

        if self._advancing:
            # ADVANCE phase — flying forward through gate
            if abs(nx) > 0.75 or (abs(ny) > 0.88 and r_frac < 0.08):
                self._advancing = False
                self._stabilize_start = None
                print(
                    "[pilot] STABILIZE → gate too far off-center, re-aligning",
                    flush=True,
                )
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif r_frac >= VISION_PROXIMITY_R_FRAC:
                pitch = self._clamp_pitch(cruise * 0.45 * p_scale)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            else:
                threats = self._obstacle_threats(r_frac)
                if threats:
                    yaw_rate, pitch, thrust = self._avoid_obstacles(
                        yaw_rate, z_target, HOVER_THRUST, r_frac
                    )
                else:
                    alignment = max(0.0, 1.0 - abs(nx))
                    pitch = self._clamp_pitch(
                        cruise * (0.35 + 0.45 * alignment) * eff_conf * p_scale
                    )
                    thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        else:
            if allow_advance and aligned_x and abs(nx) < 0.15:
                pitch = self._clamp_pitch(cruise * 0.18 * eff_conf)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            else:
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)

            if allow_advance and aligned_x and self._gates_passed == 0:
                if self._stabilize_start is None:
                    self._stabilize_start = _time.monotonic()
                    print(
                        f"[pilot] GATE aligned-x, holding {STABILIZE_HOLD_S}s...",
                        flush=True,
                    )
                elif _time.monotonic() - self._stabilize_start >= STABILIZE_HOLD_S:
                    self._advancing = True
                    self._stabilize_start = None
                    print(
                        "[pilot] ADVANCE → gate aligned-x, pitching forward",
                        flush=True,
                    )
            else:
                if self._stabilize_start is not None:
                    self._stabilize_start = None

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    def _fly_toward_gate_telemetry(self, gate: dict, odometry: dict) -> None:  # type: ignore[type-arg]
        gx, gy, gz = gate["position_ned"]
        ox, oy, oz = odometry["x"], odometry["y"], odometry.get("z", 0.0)
        cruise = self._cruise_pitch()

        dx = gx - ox
        dy = gy - oy
        dist = math.sqrt(dx * dx + dy * dy)
        self._telem_min_dist = min(self._telem_min_dist, dist)

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

        yaw_rate = _clamp(TELEMETRY_YAW_GAIN * bearing_error, -1.0, 1.0)

        if (
            self._advancing
            and self._telem_min_dist < 2.5
            and dist > max(self._telem_min_dist + 1.5, 3.0)
        ):
            self._mark_gate_passed()
            pitch = self._clamp_pitch(abs(cruise) * 0.25)
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
            return

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
                    "[pilot] STABILIZE → lost centering, re-aligning",
                    flush=True,
                )
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
            elif dist < TELEMETRY_PROXIMITY_M:
                # Very close — stop pitching, fine-tune yaw+altitude
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
            else:
                threats = self._obstacle_threats(0.0)
                if threats:
                    yaw_rate, pitch, thrust = self._avoid_obstacles(
                        yaw_rate, gz, HOVER_THRUST, 0.0
                    )
                else:
                    alignment = max(0.0, 1.0 - abs(nx_telemetry))
                    pitch = self._clamp_pitch(cruise * (0.30 + 0.50 * alignment))
                    thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)
        else:
            # STABILIZE phase — hover, align yaw+altitude only
            pitch = 0.0
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=gz)

            if centered:
                if self._stabilize_start is None:
                    self._stabilize_start = _time.monotonic()
                elif _time.monotonic() - self._stabilize_start >= STABILIZE_HOLD_S:
                    # Held heading long enough → advance
                    self._advancing = True
                    self._stabilize_start = None
                    print(
                        "[pilot] ADVANCE → gate centered, pitching forward",
                        flush=True,
                    )
            else:
                # Not centered — reset hold timer
                if self._stabilize_start is not None:
                    self._stabilize_start = None

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

        return clamped_thrust
