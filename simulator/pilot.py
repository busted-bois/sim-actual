"""Pilot — attitude-mode gate racer with altitude PID.

Called at ~250 Hz by controller.update(). Uses ATTITUDE mode with pitch_rate
for forward motion and an altitude PID for thrust control. Reads shared_data
(written by mavlink_rx and vision_rx) and sets controller commands directly.
"""

from __future__ import annotations

import math
import time as _time

from simulator.transforms import bearing_to_yaw_delta

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
HOVER_THRUST = 0.5
CRUISE_THRUST = 0.55
CRUISE_PITCH_RATE = -0.10
TELEM_PITCH_RATE = -0.045
MAX_PITCH_RATE = 0.10
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
VISION_ALIGN_PITCH_RATE = -0.10

TELEMETRY_YAW_GAIN = 1.0
TELEMETRY_PROXIMITY_M = 3.0

OBSTACLE_CLEAR_ZONE = 0.25
AMBIGUOUS_YAW_GAIN = math.radians(15)

POST_GATE_HOVER_S = 0.0
POST_GATE_VISION_COOLDOWN_S = 8.0
FORCE_TELEM_AFTER_PASS_S = 5.0
POST_PASS_BRAKE_S = 2.0
POST_PASS_BRAKE_PITCH = 0.03
POST_PASS_ALIGN_S = 2.5
POST_PASS_SPEED_CAP_S = 4.0
INTER_GATE_MAX_PITCH = 0.035
RECOVERY_BRAKE_PITCH = 0.0
INTER_GATE_ALIGN_RAD = math.radians(12)
INTER_GATE_DIRECT_UNTIL_M = 18.0
INTER_GATE_YAW_GAIN = 0.75
PITCH_TAPER_START_FRAC = 0.55
PATH_REJOIN_XTE_M = 4.0
INTER_GATE_FLYTHROUGH_M = 10.0
INTER_GATE_RECOVERY_XTE_M = 6.0
INTER_GATE_RECOVERY_BRAKE_PITCH = 0.02
MIN_PASS_INTERVAL_S = 4.0
TELEM_PASS_MIN_APPROACH_M = 3.0
PATH_PASS_MIN_APPROACH_M = 5.0
PAST_GATE_PLANE_M = 2.0
PASS_RECEDE_MAX_M = 7.0
GATE_PASS_SNAP_M = 5.5
GATE_PASS_MAX_XTE_M = 4.0
TRANSIT_MAX_PITCH = 0.045
TRANSIT_HANDOFF_M = 22.0
FINAL_APPROACH_M = 22.0
GATE_OVERSHOOT_RECOVERY_M = 28.0
PATH_LOOKAHEAD_DIST_M = 12.0
PATH_XTE_YAW_GAIN = 0.14
VISION_FINAL_APPROACH_M = 18.0
VISION_PASS_MAX_DIST_M = 5.0
VISION_MAX_NX = 0.42
VISION_CHASE_MAX_NX = 0.35
VISION_MAX_NY = 0.95

VIO_MAX_NAV_AGE_S = 2.0
VIO_MIN_NAV_QUALITY = 0.25

# Search mode — yaw scan when no gate map / vision
SEARCH_SWEEP_YAW_RATE = 0.8
SEARCH_SWEEP_PERIOD_S = 2.0
SEARCH_FORWARD_PITCH = -0.03
SEARCH_WARMUP_S = 0.5

# Passed-gate rejection — only ignore vision toward gates already flown through
PASSED_GATE_NEAR_M = 22.0
PASSED_GATE_ANGLE_RAD = math.radians(50)

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
        self._last_target_course_idx: int = -1
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
        self._segment_min_dist: float = float("inf")
        self._vision_cooldown_until: float = 0.0
        self._mode_str = "???"
        self._pending_map_sync: bool = False
        self._last_pass_mono: float = 0.0
        self._path_lookahead: bool = False
        self._last_logged_sim_idx: int = -1
        self._last_progress_log: float = 0.0
        self._brake_until: float = 0.0
        self._prefetch_logged_idx: int = -1
        self._last_yaw_cmd: float = 0.0
        self._overshoot_logged: bool = False
        self._post_pass_speed_cap_until: float = 0.0
        self._post_pass_align_until: float = 0.0
        self._inter_gate_recover: bool = False
        self._inter_gate_aligned: bool = False
        self.data["pilot_course_gate_index"] = 0
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

    def _reset_approach_state(
        self, *, keep_post_gate: bool = False, keep_yaw: bool = False
    ) -> None:
        self._advancing = False
        self._path_lookahead = False
        self._stabilize_start = None
        if not keep_post_gate:
            self._post_gate_time = None
        self._peak_r_frac = 0.0
        self._vision_suppress_until = 0.0
        self._telem_min_dist = float("inf")
        self._segment_min_dist = float("inf")
        if not keep_yaw:
            self._last_yaw_cmd = 0.0
        self._overshoot_logged = False
        self._inter_gate_recover = False
        self._inter_gate_aligned = False

    def _mark_gate_passed(self) -> str | None:
        now = _time.monotonic()
        if now - self._last_pass_mono < MIN_PASS_INTERVAL_S:
            return None
        if not self._upcoming_gate_ready():
            return None
        self._last_pass_mono = now
        self._gates_passed += 1
        drone_pos = self._get_position(for_telemetry=True)
        if drone_pos is not None:
            self._passed_gate_positions.append(drone_pos)

        gate_id = None
        course_gate = self._get_course_gate()
        if course_gate is not None:
            gate_id = self._gate_id(course_gate)
            if gate_id:
                self._completed_gates.add(gate_id)
            self._course_gate_index += 1
        else:
            self._pending_map_sync = True
        self.data["pilot_course_gate_index"] = self._course_gate_index
        self._force_telem_until = _time.monotonic() + FORCE_TELEM_AFTER_PASS_S
        self._vision_cooldown_until = _time.monotonic() + POST_GATE_VISION_COOLDOWN_S
        self._brake_until = _time.monotonic() + POST_PASS_BRAKE_S
        self._post_pass_speed_cap_until = _time.monotonic() + POST_PASS_SPEED_CAP_S
        self._post_pass_align_until = _time.monotonic() + POST_PASS_ALIGN_S
        self._reset_approach_state()
        self._post_gate_time = None
        # Force NEW TARGET log/handoff for the next gate on next tick.
        self._last_target_course_idx = self._course_gate_index - 1
        self._path_lookahead = False
        self._last_yaw_cmd = 0.0
        self._inter_gate_recover = False
        self._inter_gate_aligned = False
        self._on_course_target_changed(keep_yaw=True)
        self._snap_yaw_toward_course_gate()

        print(
            f"[pilot] Gate {gate_id} marked COMPLETE, "
            f"total passed: {self._gates_passed}",
            flush=True,
        )
        print(
            f"[pilot] FLY-THROUGH at pos={drone_pos}, transit to next gate",
            flush=True,
        )
        next_target = self._get_course_gate()
        if next_target is not None:
            pos = next_target["position_ned"]
            print(
                f"[pilot] next target → gate {self._gate_id(next_target)} "
                f"at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                flush=True,
            )
            odometry = self.data.get("odometry")
            if odometry is not None and drone_pos is not None:
                yaw = self._yaw_from_odometry(odometry)
                bearing = math.atan2(
                    pos[1] - drone_pos[1], pos[0] - drone_pos[0]
                )
                err = math.degrees(bearing_to_yaw_delta(bearing, yaw))
                print(
                    f"[pilot] reheading toward gate {self._gate_id(next_target)} "
                    f"(bearing err {err:+.0f}°)",
                    flush=True,
                )
        return gate_id

    def _course_steering_bearing(self, odometry: dict) -> float | None:  # type: ignore[type-arg]
        """Bearing for steering — path tangent after pass, direct near gate."""
        from simulator.course_path import segment_for_index

        gate = self._get_course_gate()
        if gate is None:
            return None
        cx, cy, _ = gate["position_ned"]
        ox, oy = odometry["x"], odometry["y"]
        dist = math.hypot(cx - ox, cy - oy)
        seg = segment_for_index(self.data, self._course_gate_index)
        if seg is not None and self._gates_passed > 0 and dist > INTER_GATE_DIRECT_UNTIL_M:
            return float(seg["bearing_rad"])
        return math.atan2(cy - oy, cx - ox)

    def _snap_yaw_toward_course_gate(self) -> None:
        """After pass — hold path tangent toward next gate, not rear vision."""
        odometry = self.data.get("odometry")
        if odometry is None:
            self._last_yaw_cmd = 0.0
            return
        bearing = self._course_steering_bearing(odometry)
        if bearing is None:
            self._last_yaw_cmd = 0.0
            return
        yaw = self._yaw_from_odometry(odometry)
        err = bearing_to_yaw_delta(bearing, yaw)
        self._last_yaw_cmd = _clamp(1.0 * err, -0.95, 0.95)

    def _yaw_rate_toward_course_gate(self, odometry: dict) -> float:  # type: ignore[type-arg]
        bearing = self._course_steering_bearing(odometry)
        if bearing is None:
            return self._smooth_yaw_rate(0.0)
        yaw = self._yaw_from_odometry(odometry)
        err = bearing_to_yaw_delta(bearing, yaw)
        return self._smooth_yaw_rate(_clamp(0.90 * err, -0.95, 0.95))

    def _vision_steering_allowed(self) -> bool:
        """Gate 0 vision only; after pass use map until final approach."""
        if self._gates_passed == 0:
            return True
        if _time.monotonic() < self._vision_cooldown_until:
            return False
        cur = self._course_gate_dist()
        return cur is not None and cur < VISION_FINAL_APPROACH_M

    def _vision_flythrough(self, nx: float, ny: float, r_frac: float) -> bool:
        """Detect gate pass from vision when gate fills frame (r_frac stays large)."""
        if self._peak_r_frac < 0.08 or not self._advancing:
            return False
        if not self._vision_pass_allowed():
            return False

        # Gate 0 — strict pass, but allow fill-frame and exit signatures
        if self._gates_passed == 0:
            if self._peak_r_frac < 0.12:
                return False
            odometry = self.data.get("odometry")
            if abs(nx) < 0.35 and r_frac < self._peak_r_frac * 0.55:
                return True
            if self._peak_r_frac > 0.22 and r_frac < 0.04:
                return True
            if (
                odometry is not None
                and self._peak_r_frac > 0.14
                and self._gate_is_behind(odometry)
            ):
                return True
            return False

        # Classic: gate shrinks while still near center
        if abs(nx) < 0.28 and r_frac < self._peak_r_frac * 0.60:
            return True
        # Gate filled frame — lateral sweep means we flew past center
        if self._peak_r_frac > 0.18 and abs(nx) > 0.38:
            return True
        # Large blob still in view but no longer centered after advance
        if (
            self._peak_r_frac > 0.22
            and abs(nx) > 0.30
            and r_frac > 0.10
        ):
            return True
        # Gate dominated frame then vanished
        if self._peak_r_frac > 0.28 and r_frac < 0.02:
            return True
        # ny flip only for first gate (close approach from below)
        if (
            self._gates_passed == 0
            and self._peak_r_frac > 0.30
            and ny < -0.10
            and r_frac > 0.15
        ):
            return True
        return False

    def _gate_flythrough_detected(self) -> bool:
        """Detect fly-through via path geometry or receding distance."""
        if _time.monotonic() - self._last_pass_mono < MIN_PASS_INTERVAL_S:
            return False

        # Gate 0: vision-primary, but allow odometry fly-through once committed
        if self._course_gate_index == 0:
            if not self._advancing or self._peak_r_frac < 0.14:
                return False

        odometry = self.data.get("odometry")
        gate = self._get_course_gate()
        if odometry is None or gate is None:
            return False

        from simulator.course_path import along_track_m, cross_track_m, segment_for_index

        ox, oy = odometry["x"], odometry["y"]
        cur_dist = self._course_gate_dist(odometry)
        if cur_dist is not None:
            self._telem_min_dist = min(self._telem_min_dist, cur_dist)
            self._segment_min_dist = min(self._segment_min_dist, cur_dist)

        approached = self._segment_min_dist < PATH_PASS_MIN_APPROACH_M
        if not approached or cur_dist is None:
            return False

        idx = self._course_gate_index
        seg = segment_for_index(self.data, idx)
        if seg is not None:
            at = along_track_m(
                (ox, oy),
                (seg["start"][0], seg["start"][1]),
                (seg["end"][0], seg["end"][1]),
            )
            seg_len = float(seg["length_m"])
            xte = abs(
                cross_track_m(
                    (ox, oy),
                    (seg["start"][0], seg["start"][1]),
                    (seg["end"][0], seg["end"][1]),
                )
            )
            if (
                at > seg_len + PAST_GATE_PLANE_M
                and cur_dist < GATE_PASS_SNAP_M
                and xte < GATE_PASS_MAX_XTE_M
            ):
                print(
                    f"[pilot] path fly-through gate {self._gate_id(gate)} "
                    f"dist={cur_dist:.1f}m xte={xte:.1f}m",
                    flush=True,
                )
                return True

        if (
            cur_dist > self._segment_min_dist + 1.5
            and cur_dist < self._segment_min_dist + PASS_RECEDE_MAX_M
            and self._segment_min_dist < GATE_PASS_SNAP_M
            and cur_dist < GATE_PASS_SNAP_M
        ):
            if seg is not None:
                xte = abs(
                    cross_track_m(
                        (ox, oy),
                        (seg["start"][0], seg["start"][1]),
                        (seg["end"][0], seg["end"][1]),
                    )
                )
                if xte > GATE_PASS_MAX_XTE_M:
                    return False
            print(
                f"[pilot] recede fly-through gate {self._gate_id(gate)} "
                f"dist={cur_dist:.1f}m min={self._segment_min_dist:.1f}m",
                flush=True,
            )
            return True

        return False

    def _gate_is_behind(self, odometry: dict) -> bool:  # type: ignore[type-arg]
        """True when drone has passed the gate plane while still on the path."""
        from simulator.course_path import along_track_m, cross_track_m, segment_for_index

        idx = self._course_gate_index
        seg = segment_for_index(self.data, idx)
        if seg is None:
            return False
        pos = (odometry["x"], odometry["y"])
        xte = abs(
            cross_track_m(
                pos,
                (seg["start"][0], seg["start"][1]),
                (seg["end"][0], seg["end"][1]),
            )
        )
        if xte > PATH_REJOIN_XTE_M:
            return False
        at = along_track_m(
            pos,
            (seg["start"][0], seg["start"][1]),
            (seg["end"][0], seg["end"][1]),
        )
        return at > float(seg["length_m"]) + 0.5

    def _path_cross_track(self, odometry: dict) -> tuple[float, dict | None]:  # type: ignore[type-arg]
        from simulator.course_path import cross_track_m, segment_for_index

        seg = segment_for_index(self.data, self._course_gate_index)
        if seg is None:
            return 0.0, None
        xte = cross_track_m(
            (odometry["x"], odometry["y"]),
            (seg["start"][0], seg["start"][1]),
            (seg["end"][0], seg["end"][1]),
        )
        return xte, seg

    def _along_track_frac(
        self, odometry: dict, seg: dict  # type: ignore[type-arg]
    ) -> float:
        from simulator.course_path import along_track_m

        at = along_track_m(
            (odometry["x"], odometry["y"]),
            (seg["start"][0], seg["start"][1]),
            (seg["end"][0], seg["end"][1]),
        )
        return at / max(float(seg["length_m"]), 1.0)

    def _recovery_active(self, odometry: dict) -> bool:  # type: ignore[type-arg]
        """Past gate plane after approaching — need turn-around, not more path pitch."""
        if self._gates_passed == 0:
            return False
        cur = self._course_gate_dist(odometry)
        if cur is None or cur > GATE_OVERSHOOT_RECOVERY_M:
            return False
        if not self._gate_is_behind(odometry):
            return False
        return self._telem_min_dist < 14.0

    def _post_pass_speed_capped(self) -> bool:
        return _time.monotonic() < self._post_pass_speed_cap_until

    def _post_pass_align_phase(self) -> bool:
        return _time.monotonic() < self._post_pass_align_until

    def _sync_sim_gate_progress(self) -> None:
        """Sync pilot gate progress from sim active_gate_index (guarded)."""
        track_gates = self.data.get("track_gates") or []
        if not track_gates:
            return
        sim_idx = int(self.data.get("active_gate_index", 0))
        if (
            self._course_gate_index == 0
            and sim_idx > 0
            and self._peak_r_frac > 0.15
            and self._advancing
        ):
            odometry = self.data.get("odometry")
            if odometry is None:
                return
            cur = self._course_gate_dist(odometry)
            behind = self._gate_is_behind(odometry)
            receding = (
                cur is not None
                and self._telem_min_dist < 6.0
                and cur > self._telem_min_dist + 1.5
            )
            if behind or receding:
                print("[pilot] sim confirmed gate 0 pass", flush=True)
                self._mark_gate_passed()
                self._last_logged_sim_idx = sim_idx
            return
        if sim_idx > self._course_gate_index and sim_idx != self._last_logged_sim_idx:
            self._last_logged_sim_idx = sim_idx
            print(
                f"[pilot] sim active_gate_index={sim_idx} "
                f"(course {self._course_gate_index + 1})",
                flush=True,
            )

    def _cruise_pitch(self) -> float:
        return CRUISE_PITCH_RATE if self._gates_passed == 0 else TELEM_PITCH_RATE

    def _clamp_pitch(self, pitch: float) -> float:
        return _clamp(pitch, -MAX_PITCH_RATE, MAX_PITCH_RATE)

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
        if self._pending_map_sync and track_gates:
            for i in range(min(self._gates_passed, len(track_gates))):
                gid = self._gate_id(track_gates[i])
                if gid:
                    self._completed_gates.add(gid)
            self._course_gate_index = self._gates_passed
            self.data["pilot_course_gate_index"] = self._course_gate_index
            self._pending_map_sync = False
            print(
                f"[pilot] track map synced, course index={self._course_gate_index}",
                flush=True,
            )
        while self._course_gate_index < len(track_gates):
            gate = track_gates[self._course_gate_index]
            gid = self._gate_id(gate)
            if gid and gid in self._completed_gates:
                self._course_gate_index += 1
                continue
            return gate
        return None

    def _course_gate_dist(self, odometry: dict | None = None) -> float | None:  # type: ignore[type-arg]
        if odometry is None:
            odometry = self.data.get("odometry")
        gate = self._get_course_gate()
        if not odometry or gate is None:
            return None
        gx, gy, gz = gate["position_ned"]
        ox, oy, oz = odometry["x"], odometry["y"], odometry.get("z", 0.0)
        return math.sqrt((gx - ox) ** 2 + (gy - oy) ** 2 + (gz - oz) ** 2)

    def _track_course_gate_dist(self) -> float | None:
        dist = self._course_gate_dist()
        if dist is not None:
            self._telem_min_dist = min(self._telem_min_dist, dist)
            self._segment_min_dist = min(self._segment_min_dist, dist)
        return dist

    def _vision_pass_allowed(self) -> bool:
        if self._gates_passed == 0:
            return True
        cur = self._course_gate_dist()
        return cur is not None and cur <= VISION_PASS_MAX_DIST_M

    def _map_only_phase(self) -> bool:
        """Inter-gate transit — map/path steering until real final approach."""
        if self._gates_passed == 0:
            return False
        cur = self._course_gate_dist()
        if cur is None:
            return True
        if cur > VISION_FINAL_APPROACH_M:
            return True
        return self._telem_min_dist > VISION_FINAL_APPROACH_M * 0.65

    def _log_transit_progress(self, gate: dict, odometry: dict) -> None:  # type: ignore[type-arg]
        now = _time.monotonic()
        if now - self._last_progress_log < 4.0:
            return
        self._last_progress_log = now
        dist = self._course_gate_dist(odometry)
        if dist is None:
            return
        course = self._get_course_gate()
        gid = self._gate_id(course) if course else self._gate_id(gate)
        ox, oy, oz = odometry["x"], odometry["y"], odometry.get("z", 0.0)
        xte, _ = self._path_cross_track(odometry)
        print(
            f"[pilot] transit → gate {gid} dist={dist:.1f}m xte={xte:+.1f}m "
            f"pos=({ox:.1f}, {oy:.1f}, {oz:.1f})",
            flush=True,
        )

    def _get_lookahead_gate(self) -> dict | None:  # type: ignore[type-arg]
        track_gates = self.data.get("track_gates") or []
        idx = self._course_gate_index
        if idx + 1 < len(track_gates):
            return track_gates[idx + 1]
        return None

    def _prefetch_upcoming_gate(self) -> dict | None:  # type: ignore[type-arg]
        """Publish next course gate from map before current fly-through."""
        nxt = self._get_lookahead_gate()
        if nxt is None:
            self.data.pop("next_gate", None)
            self.data.pop("upcoming_gate", None)
            return None

        pos = nxt["position_ned"]
        gid = self._gate_id(nxt)
        course_idx = self._course_gate_index + 1
        self.data["next_gate"] = nxt
        self.data["upcoming_gate"] = {
            "gate_id": gid,
            "position_ned": pos,
            "course_index": course_idx,
            "source": "map",
        }
        if course_idx != self._prefetch_logged_idx:
            self._prefetch_logged_idx = course_idx
            print(
                f"[pilot] prefetched gate {gid} at "
                f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                f"before gate {self._course_gate_index}",
                flush=True,
            )
        return nxt

    def _upcoming_gate_ready(self) -> bool:
        """Next gate known (map or vision) before marking current pass."""
        if self._get_lookahead_gate() is None:
            return True
        if self.data.get("next_gate") or self.data.get("upcoming_gate"):
            return True
        detected = self.data.get("upcoming_gate_detected") or {}
        return bool(detected.get("detected"))

    def _in_inter_gate_transit(self, odometry: dict | None = None) -> bool:  # type: ignore[type-arg]
        if self._gates_passed == 0:
            return False
        cur = self._course_gate_dist(odometry)
        return cur is None or cur > TRANSIT_HANDOFF_M

    def _in_final_approach(self, odometry: dict | None = None) -> bool:  # type: ignore[type-arg]
        if self._gates_passed == 0:
            return False
        cur = self._course_gate_dist(odometry)
        return cur is not None and cur < FINAL_APPROACH_M

    def _approach_pitch(
        self,
        bearing_error: float,
        cur_dist: float,
        cruise: float,
        odometry: dict,  # type: ignore[type-arg]
    ) -> float:
        """Pitch through gate opening — don't stop and stare within range."""
        yaw = self._yaw_from_odometry(odometry)
        pitch_err = abs(bearing_error)
        from simulator.course_path import segment_for_index

        seg = segment_for_index(self.data, self._course_gate_index)
        if seg is not None:
            path_err = abs(
                bearing_to_yaw_delta(float(seg["bearing_rad"]), yaw)
            )
            pitch_err = min(pitch_err, path_err)

        behind = self._gate_is_behind(odometry)
        if self._recovery_active(odometry):
            if pitch_err > math.radians(40):
                return self._clamp_pitch(RECOVERY_BRAKE_PITCH)
            base = max(abs(cruise) * 0.35, INTER_GATE_MAX_PITCH)
            align = max(0.15, 1.0 - pitch_err / math.radians(40))
            return self._clamp_pitch(-base * align)

        if self._post_pass_speed_capped() and pitch_err > INTER_GATE_ALIGN_RAD:
            return 0.0

        if (
            cur_dist < 2.0
            and behind
            and cur_dist > self._telem_min_dist + 0.4
        ):
            return 0.0
        if pitch_err > math.radians(75):
            return 0.0
        base = max(abs(cruise) * 0.70, TRANSIT_MAX_PITCH * 1.15)
        if self._post_pass_speed_capped():
            base = min(base, INTER_GATE_MAX_PITCH)
        align = max(0.30, 1.0 - pitch_err / math.radians(75))
        if seg is not None and not behind:
            atf = self._along_track_frac(odometry, seg)
            if atf > PITCH_TAPER_START_FRAC:
                taper = max(0.0, 1.0 - (atf - PITCH_TAPER_START_FRAC) / 0.40)
                base *= taper
        return self._clamp_pitch(-base * align)

    def _approach_z_target(self, oz: float, gz: float, cur_dist: float) -> float:
        if cur_dist > 20.0:
            blend = 0.12
        elif cur_dist > 8.0:
            blend = 0.12 + 0.55 * (20.0 - cur_dist) / 12.0
        else:
            blend = 0.75
        return oz + (gz - oz) * blend

    def _smooth_yaw_rate(self, target: float) -> float:
        self._last_yaw_cmd = 0.82 * self._last_yaw_cmd + 0.18 * target
        return self._last_yaw_cmd

    def _yaw_rate_for_bearing(
        self, bearing_error: float, cur_dist: float, *, aggressive: bool = False
    ) -> float:
        if aggressive:
            return _clamp(INTER_GATE_YAW_GAIN * bearing_error, -0.90, 0.90)
        gain = 0.50 if cur_dist > 18.0 else 0.65
        if self._gates_passed > 0:
            gain = 0.65 if cur_dist > 18.0 else 0.80
        raw = _clamp(gain * bearing_error, -0.65, 0.65)
        return self._smooth_yaw_rate(raw)

    def _direct_bearing_to_course_gate(self, odometry: dict) -> float | None:  # type: ignore[type-arg]
        gate = self._get_course_gate()
        if gate is None:
            return None
        cx, cy, _ = gate["position_ned"]
        return math.atan2(cy - odometry["y"], cx - odometry["x"])

    def _transit_pitch(
        self, bearing_error: float, cur_dist: float, odometry: dict  # type: ignore[type-arg]
    ) -> float:
        cruise = self._cruise_pitch()
        return self._approach_pitch(bearing_error, cur_dist, cruise, odometry)

    def _yaw_from_odometry(self, odometry: dict) -> float:  # type: ignore[type-arg]
        qw = odometry.get("qw", 1.0)
        qx = odometry.get("qx", 0.0)
        qy = odometry.get("qy", 0.0)
        qz = odometry.get("qz", 0.0)
        return math.atan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        )

    def _bearing_for_steer(
        self,
        gate: dict,  # type: ignore[type-arg]
        odometry: dict,  # type: ignore[type-arg]
    ) -> tuple[float, dict | None]:
        """Path bearing far out; smooth blend to gate bearing on approach."""
        from simulator.course_path import segment_for_index

        gx, gy, _ = gate["position_ned"]
        ox, oy = odometry["x"], odometry["y"]
        dist = math.hypot(gx - ox, gy - oy)
        direct = math.atan2(gy - oy, gx - ox)
        seg = segment_for_index(self.data, self._course_gate_index)
        if seg is None:
            return direct, None

        path = float(seg["bearing_rad"])
        if self._gates_passed > 0 and self._recovery_active(odometry):
            if dist < GATE_OVERSHOOT_RECOVERY_M:
                return direct, seg
        if self._gates_passed > 0 and dist > TRANSIT_HANDOFF_M:
            yaw = self._yaw_from_odometry(odometry)
            err = bearing_to_yaw_delta(path, yaw)
            if abs(err) < math.radians(80):
                return path, seg
            return yaw + _clamp(err * 0.22, -0.30, 0.30), seg
        if dist > TRANSIT_HANDOFF_M:
            return path, seg
        atf = self._along_track_frac(odometry, seg)
        if atf > 0.72:
            t = min(1.0, (atf - 0.72) / 0.28)
            err = bearing_to_yaw_delta(direct, path)
            return path + err * t, seg
        if dist > 8.0:
            t = (TRANSIT_HANDOFF_M - dist) / max(TRANSIT_HANDOFF_M - 8.0, 1.0)
            err = bearing_to_yaw_delta(direct, path)
            return path + err * t, seg
        return direct, seg

    def _steering_gate(self, odometry: dict) -> dict | None:  # type: ignore[type-arg]
        """Fly target — current course gate during transit."""
        current = self._get_course_gate()
        if current is None:
            self._path_lookahead = False
            return None

        if self._gates_passed > 0:
            self._path_lookahead = False
            return current

        # Gate 0 — always steer to current gate; no lookahead to gate 1
        self._path_lookahead = False
        return current

    def _on_course_target_changed(self, *, keep_yaw: bool = False) -> None:
        if self._course_gate_index == self._last_target_course_idx:
            return
        in_hover = self._post_gate_time is not None
        self._last_target_course_idx = self._course_gate_index
        self._reset_approach_state(keep_post_gate=in_hover, keep_yaw=keep_yaw)
        gate = self._get_course_gate()
        if gate is not None:
            gid = self._gate_id(gate)
            self._last_gate_id = gid
            print(
                f"[pilot] NEW TARGET gate {gid} "
                f"(course {self._course_gate_index + 1}/"
                f"{len(self.data.get('track_gates') or [])})",
                flush=True,
            )

    def _publish_course_nav(self, gate: dict, odometry: dict) -> None:  # type: ignore[type-arg]
        from simulator.course_path import segment_for_index

        idx = self._course_gate_index
        seg = segment_for_index(self.data, idx + 1 if self._path_lookahead else idx)
        next_gate = self._get_lookahead_gate()
        self.data["next_gate"] = next_gate
        self.data["course_nav"] = {
            "target_gate_id": self._gate_id(gate),
            "lookahead": self._path_lookahead,
            "course_index": idx,
            "segment_to": seg["to_idx"] if seg else None,
            "next_gate_id": self._gate_id(next_gate) if next_gate else None,
        }

    def _vision_usable(self, gate_target: dict) -> bool:  # type: ignore[type-arg]
        if not self._vision_steering_allowed():
            return False
        if (
            self._gates_passed > 0
            and _time.monotonic() < self._vision_cooldown_until
        ):
            return False
        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        # Rear-view gate frame after pass — never chase it
        if self._gates_passed > 0 and abs(nx) > 0.35:
            return False
        # Gate 0 commit: keep tracking through fill-frame / ambiguous frames
        if (
            self._gates_passed == 0
            and self._advancing
            and self._peak_r_frac > 0.14
            and gate_target.get("raw_detected")
            and abs(nx) < 0.72
        ):
            return True
        if abs(nx) > VISION_MAX_NX:
            return False
        # After close approach, ignore off-center blobs (passed gate / wrong contour)
        if self._peak_r_frac > 0.15 and abs(nx) > VISION_CHASE_MAX_NX:
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

    def _get_nav_state(self, *, for_telemetry: bool = False) -> dict:  # type: ignore[type-arg]
        """Nav state for control. Telemetry/map steering always uses odometry."""
        odometry = self.data.get("odometry")
        if for_telemetry and odometry is not None:
            qw = odometry.get("qw", 1.0)
            qx = odometry.get("qx", 0.0)
            qy = odometry.get("qy", 0.0)
            qz = odometry.get("qz", 0.0)
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
            )
            return {
                "source": "odometry",
                "pos": (odometry["x"], odometry["y"], odometry.get("z", 0.0)),
                "quat": (qw, qx, qy, qz),
                "yaw": yaw,
                "vel": (
                    odometry.get("vx", 0.0),
                    odometry.get("vy", 0.0),
                    odometry.get("vz", 0.0),
                ),
                "quality": 1.0,
            }

        vio = self.data.get("vio", {})
        if vio.get("initialized"):
            age = vio.get("vision_age_s")
            quality = float(vio.get("nav_quality", 0.0))
            if quality >= VIO_MIN_NAV_QUALITY and (
                age is None or age < VIO_MAX_NAV_AGE_S
            ):
                pos = vio["pos_ned"]
                quat = vio.get("quat", (1.0, 0.0, 0.0, 0.0))
                qw, qx, qy, qz = quat
                yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
                )
                vel = vio.get("vel_ned", (0.0, 0.0, 0.0))
                return {
                    "source": "vio",
                    "pos": pos,
                    "quat": quat,
                    "yaw": yaw,
                    "vel": vel,
                    "quality": quality,
                }

        odometry = self.data.get("odometry")
        if odometry is not None:
            qw = odometry.get("qw", 1.0)
            qx = odometry.get("qx", 0.0)
            qy = odometry.get("qy", 0.0)
            qz = odometry.get("qz", 0.0)
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
            )
            return {
                "source": "odometry",
                "pos": (odometry["x"], odometry["y"], odometry.get("z", 0.0)),
                "quat": (qw, qx, qy, qz),
                "yaw": yaw,
                "vel": (
                    odometry.get("vx", 0.0),
                    odometry.get("vy", 0.0),
                    odometry.get("vz", 0.0),
                ),
                "quality": 1.0,
            }

        pos_ned = self.data.get("pos_ned")
        if pos_ned is not None and len(pos_ned) >= 3:
            yaw = float(self.data.get("yaw_rad", 0.0))
            return {
                "source": "pos_ned",
                "pos": (pos_ned[0], pos_ned[1], pos_ned[2]),
                "quat": None,
                "yaw": yaw,
                "vel": self.data.get("vel_ned", (0.0, 0.0, 0.0)),
                "quality": 0.5,
            }
        return {"source": "none", "pos": None, "quality": 0.0}

    def _get_position(self, *, for_telemetry: bool = False) -> tuple[float, float, float] | None:
        nav = self._get_nav_state(for_telemetry=for_telemetry)
        pos = nav.get("pos")
        if pos is not None and len(pos) >= 3:
            return (pos[0], pos[1], pos[2])
        return None

    def _is_passed_gate(self, nx: float) -> bool:
        """True if detection bearing points back at a gate we already flew through."""
        drone_pos = self._get_position(for_telemetry=True)
        if drone_pos is None or not self._passed_gate_positions:
            return False

        yaw = self._get_nav_state(for_telemetry=True).get(
            "yaw", self.data.get("yaw_rad", 0.0)
        )
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
        gate = self._steering_gate(odometry)
        if gate is None:
            return None
        self._publish_course_nav(gate, odometry)
        return gate, odometry

    def _reacquire_without_map(self) -> None:
        """Yaw toward visible gate blob; sweep only when blind."""
        odometry = self.data.get("odometry")
        course_gate = self._get_course_gate()
        if self._gates_passed > 0 and odometry is not None and course_gate is not None:
            self._steer_toward_gate(
                course_gate, odometry, pitch=POST_PASS_BRAKE_PITCH
            )
            return

        gate = self.data.get("gate_target") or {}
        drone_pos = self._get_position(for_telemetry=True)
        z_hold = drone_pos[2] if drone_pos else None
        thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_hold)

        nx = gate.get("nx", 0.0)
        if (
            gate.get("raw_detected")
            and abs(nx) < 0.85
            and not self._is_passed_gate(nx)
        ):
            ny = gate.get("ny", 0.0)
            yaw_rate = _clamp(VISION_YAW_GAIN * 0.5 * nx, -1.0, 1.0)
            ny_offset = _clamp(
                ny * VISION_VY_GAIN * 0.5, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
            )
            z_target = (drone_pos[2] if drone_pos else 0.0) + ny_offset
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0, 0.0, yaw_rate, thrust)
            return

        self._do_search()

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
        gate = self.data.get("gate_target") or {}
        if gate.get("raw_detected"):
            pitch = 0.0
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

        self._sync_sim_gate_progress()
        self._prefetch_upcoming_gate()
        self._track_course_gate_dist()
        if self._gate_flythrough_detected():
            self._mark_gate_passed()
            self._mode_str = "transit"

        # Post-pass — map steer toward next gate (no vision look-back)
        if (
            self._gates_passed > 0
            and (
                _time.monotonic() < self._brake_until
                or self._post_pass_align_phase()
            )
            and self.data.get("odometry") is not None
        ):
            gate = self._get_course_gate()
            odometry = self.data["odometry"]
            if gate is not None:
                self._mode_str = "post_pass_brake"
                self._steer_toward_gate(gate, odometry, pitch=POST_PASS_BRAKE_PITCH)
                return

        # Post-gate hover (disabled when POST_GATE_HOVER_S == 0)
        if self._post_gate_time is not None and POST_GATE_HOVER_S > 0:
            elapsed = _time.monotonic() - self._post_gate_time
            if elapsed < POST_GATE_HOVER_S:
                self._mode_str = "post_gate_hover"
                odometry = self.data.get("odometry")
                gate = self._get_course_gate()
                if gate is not None and odometry is not None:
                    self._path_lookahead = False
                    self._publish_course_nav(gate, odometry)
                    pitch = -TRANSIT_MAX_PITCH * 0.5 if elapsed > 0.5 else 0.0
                    self._steer_toward_gate(gate, odometry, pitch=pitch)
                else:
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
        map_only = self._map_only_phase()

        gate_target = self.data.get("gate_target")
        cam = self.data.get("camera")
        vision_ok = False
        if not vision_cooldown and gate_target and cam is not None:
            age = _time.monotonic() - cam.get("received_at", 0)
            if age < VISION_MAX_AGE_S and self._vision_usable(gate_target):
                vision_ok = True

        # Final approach — map pitch through opening
        if self._gates_passed > 0 and telem is not None:
            nearest, odometry = telem
            cur = self._course_gate_dist(odometry)
            if cur is not None and cur < FINAL_APPROACH_M:
                self._advancing = True
                self._on_course_target_changed()
                if self._recovery_active(odometry):
                    self._mode_str = "recovery"
                else:
                    self._mode_str = "approach"
                self._fly_toward_gate_telemetry(nearest, odometry)
                return

        # Inter-gate transit — map steering, no vision
        if self._in_inter_gate_transit() and telem is not None:
            gate, odometry = telem
            self._on_course_target_changed()
            self._log_transit_progress(gate, odometry)
            self._mode_str = "transit"
            self._fly_toward_gate_telemetry(
                gate, odometry, allow_pass_detection=False
            )
            return

        if self._searching and telem is not None:
            self._searching = False
            self._search_start_time = None
            print("[pilot] track map ready, exiting search", flush=True)

        # Post-pass / inter-gate: map steering only until within vision range
        if (vision_cooldown or map_only) and self._gates_passed > 0:
            if telem is not None:
                nearest, odometry = telem
                self._on_course_target_changed()
                self._log_transit_progress(nearest, odometry)
                self._mode_str = "telemetry_path" if self._path_lookahead else "telemetry"
                self._fly_toward_gate_telemetry(nearest, odometry)
                return
            self._mode_str = "post_pass_hover"
            drone_pos = self._get_position()
            self._hover(drone_pos[2] if drone_pos else None)
            return

        # Gates 2+: map only — vision yaws at off-center blobs and causes pass-by
        if self._gates_passed > 0 and telem is not None and not vision_cooldown:
            nearest, odometry = telem
            self._on_course_target_changed()
            self._searching = False
            self._search_start_time = None
            if self._recovery_active(odometry):
                self._mode_str = "recovery"
            else:
                self._mode_str = (
                    "telemetry_path" if self._path_lookahead else "telemetry"
                )
            self._fly_toward_gate_telemetry(nearest, odometry)
            return

        # Gates 2+ without map yet: stay in search, don't chase random vision
        if (
            self._gates_passed > 0
            and telem is None
            and not force_telem
            and not vision_cooldown
            and vision_ok
        ):
            self._mode_str = "reacquire"
            self._reacquire_without_map()
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
            self._on_course_target_changed()
            self._searching = False
            self._search_start_time = None
            self._mode_str = "telemetry"
            self._fly_toward_gate_telemetry(nearest, odometry)
            return

        if not force_telem and vision_ok and not map_only and self._gates_passed == 0:
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
            and self._gates_passed == 0
            and not vision_cooldown
            and telem is None
            and self._vision_usable(gate_target)
        ):
            self._mode_str = "vision_ambiguous"
            self._fly_ambiguous(gate_target)
            return

        if self._searching and telem is None:
            self._mode_str = "reacquire"
            self._reacquire_without_map()
            return

        # Telemetry — only after gate 1 (gate 0 is vision-only)
        if telem is not None and self._gates_passed > 0:
            nearest, odometry = telem
            self._on_course_target_changed()
            if self._searching:
                self._searching = False
                self._search_start_time = None
            self._mode_str = "telemetry"
            self._fly_toward_gate_telemetry(nearest, odometry)
            return

        if self._searching:
            self._mode_str = "reacquire"
            self._reacquire_without_map()
            return

        # Gate 0: vision dropped mid-commit — punch through on map/odometry
        if (
            self._gates_passed == 0
            and self._advancing
            and self._peak_r_frac > 0.12
        ):
            self._mode_str = "gate0_commit"
            self._fly_gate0_commit(gate_target)
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
        if (
            gate.get("raw_detected")
            or gate.get("detected")
            or gate.get("corner_tracking")
        ):
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

    def _gate0_pitch_scale(self, r_frac: float) -> float:
        """Gate 0 — pitch up when close so we fly through, not into, the frame."""
        if r_frac > 0.14 or self._peak_r_frac > 0.18:
            return 0.60
        if r_frac > 0.08:
            return 0.45
        return self._pitch_scale(r_frac)

    def _fly_gate0_commit(self, gate_target: dict | None) -> None:  # type: ignore[type-arg]
        """Forward commit when gate fills frame and vision goes ambiguous."""
        odometry = self.data.get("odometry")
        cruise = self._cruise_pitch()
        pitch = self._clamp_pitch(cruise * 0.55)
        yaw_rate = 0.0
        z_target = None
        if gate_target and odometry is not None:
            nx = gate_target.get("nx", 0.0)
            ny = gate_target.get("ny", 0.0)
            r_frac = gate_target.get("r_frac", 0.0)
            if abs(nx) < 0.55:
                yaw_rate = _clamp(VISION_YAW_GAIN * nx * 0.45, -0.8, 0.8)
            z_now = odometry.get("z", 0.0)
            z_target = z_now + _clamp(
                ny * VISION_VY_GAIN, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
            )
            if self._vision_flythrough(nx, ny, r_frac):
                self._mark_gate_passed()
                pitch = self._clamp_pitch(abs(cruise) * 0.35)
        thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        if odometry is not None:
            yaw_rate = self._yaw_rate_toward_course_gate(odometry)
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
        odometry = self.data.get("odometry")
        course_gate = self._get_course_gate()
        if odometry is not None and course_gate is not None:
            self._publish_course_nav(course_gate, odometry)

        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        r_frac = gate_target.get("r_frac", 0.0)
        gate_conf = min(gate_target.get("gate_confidence", 1.0), 0.55)
        vio_q = float((self.data.get("vio") or {}).get("nav_quality", 1.0))
        if gate_target.get("corner_tracking"):
            gate_conf = min(1.0, gate_conf * 1.15)
        gate_conf *= max(0.5, vio_q)
        cruise = self._cruise_pitch()
        p_scale = (
            self._gate0_pitch_scale(r_frac)
            if self._gates_passed == 0
            else self._pitch_scale(r_frac)
        )

        yaw_rate = _clamp(VISION_YAW_GAIN * nx * gate_conf, -1.2, 1.2)

        ny_offset = _clamp(
            ny * VISION_VY_GAIN, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
        )
        z_now = odometry.get("z", 0.0) if odometry else 0.0

        aligned_x = abs(nx) < 0.22
        z_target = z_now + ny_offset
        eff_conf = gate_conf
        allow_advance = abs(nx) < 0.22

        self._peak_r_frac = max(
            self._peak_r_frac, r_frac if abs(nx) < 0.35 else self._peak_r_frac
        )

        if self._vision_flythrough(nx, ny, r_frac):
            self._mark_gate_passed()
            pitch = self._clamp_pitch(abs(cruise) * 0.3)
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            yaw_rate = (
                self._yaw_rate_toward_course_gate(odometry)
                if odometry is not None
                else 0.0
            )
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
            return

        if allow_advance and not self._advancing and r_frac > 0.05 and abs(nx) < 0.22:
            self._advancing = True
            self._stabilize_start = None
            print("[pilot] ADVANCE → gate aligned, pitching forward", flush=True)

        if self._advancing:
            # ADVANCE — only pitch forward when gate is centered in view
            off_center_limit = 0.32 if self._gates_passed > 0 else 0.85
            bad_vertical = (
                self._gates_passed > 0 and abs(ny) > 0.88 and r_frac < 0.08
            )
            gate0_commit = (
                self._gates_passed == 0
                and (self._peak_r_frac > 0.16 or r_frac > 0.10)
                and abs(nx) < 0.40
            )
            if abs(nx) > off_center_limit or bad_vertical:
                self._advancing = False
                self._stabilize_start = None
                print(
                    f"[pilot] STABILIZE → off-center nx={nx:+.2f}, yaw only",
                    flush=True,
                )
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif gate0_commit:
                pitch = self._clamp_pitch(cruise * 0.55)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif abs(nx) < 0.28 and r_frac >= VISION_PROXIMITY_R_FRAC:
                slow = 0.45 * p_scale
                pitch = self._clamp_pitch(cruise * slow)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif abs(nx) < 0.28:
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
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        else:
            if allow_advance and aligned_x and abs(nx) < 0.15:
                pitch = self._clamp_pitch(cruise * 0.18 * eff_conf)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            else:
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)

            if allow_advance and aligned_x:
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

    def _steer_toward_gate(
        self,
        gate: dict,  # type: ignore[type-arg]
        odometry: dict,  # type: ignore[type-arg]
        *,
        pitch: float = 0.0,
    ) -> None:
        """Yaw (and optional pitch) toward gate along course path."""
        from simulator.course_path import cross_track_m, segment_for_index

        gx, gy, gz = gate["position_ned"]
        ox, oy = odometry["x"], odometry["y"]
        yaw = self._yaw_from_odometry(odometry)
        seg = segment_for_index(self.data, self._course_gate_index)
        direct = self._course_steering_bearing(odometry)
        if self._gates_passed > 0 and direct is not None:
            bearing = direct
            aggressive_yaw = True
        else:
            bearing, seg = self._bearing_for_steer(gate, odometry)
            aggressive_yaw = False
        bearing_error = bearing_to_yaw_delta(bearing, yaw)
        if seg is not None and not aggressive_yaw:
            xte = cross_track_m(
                (ox, oy),
                (seg["start"][0], seg["start"][1]),
                (seg["end"][0], seg["end"][1]),
            )
            bearing_error -= _clamp(xte * PATH_XTE_YAW_GAIN * 0.4, -0.15, 0.15)

        cur_dist = math.hypot(gx - ox, gy - oy)
        current = self._get_course_gate()
        if current is not None:
            cx, cy, cz = current["position_ned"]
            oz = odometry.get("z", 0.0)
            cur_dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2 + (cz - oz) ** 2)
        yaw_rate = self._yaw_rate_for_bearing(
            bearing_error, cur_dist, aggressive=aggressive_yaw
        )
        if aggressive_yaw:
            self._last_yaw_cmd = yaw_rate
        z_target = self._approach_z_target(
            odometry.get("z", 0.0), gz, cur_dist
        )
        thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    def _fly_inter_gate(
        self,
        gate: dict,  # type: ignore[type-arg]
        odometry: dict,  # type: ignore[type-arg]
        *,
        allow_pass_detection: bool = True,
    ) -> None:
        """Gates 1+ — steer to course gate center, pitch only when aligned."""
        current = self._get_course_gate()
        if current is None:
            self._hover()
            return

        if _time.monotonic() < self._brake_until:
            self._steer_toward_gate(current, odometry, pitch=POST_PASS_BRAKE_PITCH)
            return

        ox, oy, oz = odometry["x"], odometry["y"], odometry.get("z", 0.0)
        cx, cy, cz = current["position_ned"]
        cur_dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2 + (cz - oz) ** 2)
        self._telem_min_dist = min(self._telem_min_dist, cur_dist)
        self._segment_min_dist = min(self._segment_min_dist, cur_dist)

        # Blend path bearing when far, direct bearing when close.
        yaw = self._yaw_from_odometry(odometry)
        steer_bearing = self._course_steering_bearing(odometry)
        if steer_bearing is None:
            steer_bearing = math.atan2(cy - oy, cx - ox)

        bearing_error = bearing_to_yaw_delta(steer_bearing, yaw)
        err_abs = abs(bearing_error)
        aligned = err_abs < math.radians(15)

        yaw_rate = self._smooth_yaw_rate(
            _clamp(0.70 * bearing_error, -0.70, 0.70)
        )
        z_target = self._approach_z_target(oz, cz, cur_dist)
        thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        cruise = self._cruise_pitch()
        pitch = 0.0

        # Must turn toward gate before pitching forward.
        if self._post_pass_align_phase() or not aligned:
            if err_abs > math.radians(10):
                pitch = POST_PASS_BRAKE_PITCH
        elif cur_dist < INTER_GATE_FLYTHROUGH_M and aligned:
            pitch = self._clamp_pitch(-max(abs(cruise) * 0.65, 0.035))
        elif aligned and cur_dist < 30.0:
            pitch = self._clamp_pitch(-INTER_GATE_MAX_PITCH)

        if allow_pass_detection and self._gate_flythrough_detected():
            self._mark_gate_passed()
            pitch = self._clamp_pitch(abs(cruise) * 0.2)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    def _fly_toward_gate_telemetry(
        self,
        gate: dict,  # type: ignore[type-arg]
        odometry: dict,  # type: ignore[type-arg]
        *,
        allow_pass_detection: bool = True,
    ) -> None:
        from simulator.course_path import cross_track_m

        if self._gates_passed > 0:
            self._fly_inter_gate(gate, odometry, allow_pass_detection=allow_pass_detection)
            return

        if _time.monotonic() < self._brake_until:
            self._steer_toward_gate(gate, odometry, pitch=POST_PASS_BRAKE_PITCH)
            return

        ox, oy, oz = odometry["x"], odometry["y"], odometry.get("z", 0.0)

        gx, gy, gz = gate["position_ned"]
        cruise = self._cruise_pitch()

        dx = gx - ox
        dy = gy - oy
        dist = math.sqrt(dx * dx + dy * dy)

        current = self._get_course_gate()
        cur_dist = dist
        if current is not None:
            cx, cy, cz = current["position_ned"]
            cur_dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2 + (cz - oz) ** 2)
        self._telem_min_dist = min(self._telem_min_dist, cur_dist)
        self._segment_min_dist = min(self._segment_min_dist, cur_dist)

        z_target = self._approach_z_target(oz, gz, cur_dist)

        if cur_dist < 6.0:
            cruise = cruise * max(0.5, cur_dist / 6.0)
        elif cur_dist < 12.0:
            cruise = cruise * 0.65

        yaw = self._yaw_from_odometry(odometry)
        bearing_to_gate, seg = self._bearing_for_steer(gate, odometry)
        bearing_error = bearing_to_yaw_delta(bearing_to_gate, yaw)

        if seg is not None:
            xte = cross_track_m(
                (ox, oy),
                (seg["start"][0], seg["start"][1]),
                (seg["end"][0], seg["end"][1]),
            )
            bearing_error -= _clamp(xte * PATH_XTE_YAW_GAIN * 0.4, -0.15, 0.15)

        yaw_rate = self._yaw_rate_for_bearing(bearing_error, cur_dist)
        if self._recovery_active(odometry):
            yaw_rate = _clamp(0.75 * bearing_error, -0.75, 0.75)
        thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        if self._recovery_active(odometry):
            thrust = self._altitude_thrust(HOVER_THRUST - 0.04, z_target=z_target)
        pitch = 0.0

        # Far transit — path follow
        if self._in_inter_gate_transit(odometry):
            if self._recovery_active(odometry) and not self._overshoot_logged:
                self._overshoot_logged = True
                print(
                    f"[pilot] OVERSHOOT gate {self._gate_id(gate)} "
                    f"dist={cur_dist:.1f}m min={self._telem_min_dist:.1f}m — braking",
                    flush=True,
                )
            pitch = self._transit_pitch(bearing_error, cur_dist, odometry)
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
            return

        # Final approach (< 22m) — drive through opening
        if self._in_final_approach(odometry):
            self._advancing = True
            if self._recovery_active(odometry) and not self._overshoot_logged:
                self._overshoot_logged = True
                print(
                    f"[pilot] OVERSHOOT gate {self._gate_id(gate)} "
                    f"dist={cur_dist:.1f}m min={self._telem_min_dist:.1f}m — braking",
                    flush=True,
                )
            pitch_bearing = bearing_error
            if seg is not None:
                path_err = bearing_to_yaw_delta(float(seg["bearing_rad"]), yaw)
                if abs(path_err) < abs(bearing_error):
                    pitch_bearing = path_err
            pitch = self._approach_pitch(pitch_bearing, cur_dist, cruise, odometry)
            if allow_pass_detection and (
                self._telem_min_dist < 5.0
                and cur_dist < 6.0
                and self._segment_min_dist < 7.0
                and (
                    self._gate_is_behind(odometry)
                    or cur_dist > self._telem_min_dist + 1.0
                )
            ):
                self._mark_gate_passed()
                pitch = self._clamp_pitch(abs(cruise) * 0.25)
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
            return

        if allow_pass_detection and cur_dist < 8.0:
            if (
                self._advancing
                and self._telem_min_dist < 3.0
                and cur_dist < 2.0
            ):
                self._mark_gate_passed()
                pitch = self._clamp_pitch(abs(cruise) * 0.25)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
                self.controller.set_control_mode("attitude")
                self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
                return

            if (
                self._advancing
                and self._telem_min_dist < 2.5
                and cur_dist > max(self._telem_min_dist + 1.5, 3.0)
            ):
                self._mark_gate_passed()
                pitch = self._clamp_pitch(abs(cruise) * 0.25)
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
                self.controller.set_control_mode("attitude")
                self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)
                return

        # Normalized horizontal alignment: 0 = perfectly aligned, 1 = 180 deg off
        nx_telemetry = bearing_error / math.pi  # [-1, 1]
        # Normalized vertical alignment
        ny_telemetry = _clamp((gz - oz) / 5.0, -1, 1)

        # Inter-gate: don't block forward motion on altitude while still far out
        if cur_dist > 10.0:
            centered = abs(bearing_error) < 0.15
        else:
            centered = abs(bearing_error) < 0.12 and abs(ny_telemetry) < 0.25

        if (
            self._gates_passed > 0
            and cur_dist < 8.0
            and not self._advancing
            and abs(bearing_error) < 0.15
        ):
            self._advancing = True
            self._stabilize_start = None

        if self._advancing:
            if abs(bearing_error) > 0.35:
                self._advancing = False
                self._stabilize_start = None
                print(
                    f"[pilot] STABILIZE → bearing err {math.degrees(bearing_error):.0f}°, yaw only",
                    flush=True,
                )
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif dist < TELEMETRY_PROXIMITY_M and not self._advancing:
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            elif abs(bearing_error) < 0.25:
                threats = self._obstacle_threats(0.0)
                if threats:
                    yaw_rate, pitch, thrust = self._avoid_obstacles(
                        yaw_rate, gz, HOVER_THRUST, 0.0
                    )
                else:
                    alignment = max(0.0, 1.0 - abs(nx_telemetry))
                    pitch = self._clamp_pitch(cruise * (0.25 + 0.40 * alignment))
                    thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            else:
                pitch = 0.0
                thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
        else:
            # STABILIZE — align bearing; pitch forward once bearing ok (alt follows)
            thrust = self._altitude_thrust(HOVER_THRUST, z_target=z_target)
            if centered:
                if self._stabilize_start is None:
                    self._stabilize_start = _time.monotonic()
                elif _time.monotonic() - self._stabilize_start >= STABILIZE_HOLD_S:
                    gate_vis = self.data.get("gate_target") or {}
                    vis_nx = abs(gate_vis.get("nx", 0.0))
                    if (
                        cur_dist <= 10.0
                        and vis_nx > 0.35
                        and gate_vis.get("raw_detected")
                    ):
                        self._stabilize_start = None
                    else:
                        self._advancing = True
                        self._stabilize_start = None
                        print(
                            "[pilot] ADVANCE → bearing aligned, pitching forward",
                            flush=True,
                        )
                pitch = 0.0
            elif abs(bearing_error) < 0.25 and cur_dist > 10.0:
                # Cruise toward gate while altitude PID catches up
                pitch = self._clamp_pitch(cruise * 0.35)
            else:
                pitch = 0.0
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
