"""Pilot — known-waypoint racer using the proven baseline control structure.

The sim has NO onboard position/velocity controller (velocity-NED setpoints are
ignored). It actuates only on set_attitude_target body-rate + thrust fields, and —
as the color baseline proved — it stays stable ONLY when those commands are SMALL
and bounded (~0.2). Large/amplified attitude commands flip the drone.

Fixed-heading translation (small, bounded commands — large ones flip the drone):
  - hold a fixed DOWN-TRACK heading (yaw = approach direction), so the drone never
    slews sideways into a gate edge while chasing the center,
  - ROLL to strafe out lateral offset to the gate line,
  - PITCH as a speed controller (accelerate if slow, BRAKE if fast — the sim has no
    drag, so without braking it coasts and overshoots),
  - thrust = altitude PID to the gate height,
  - sequence gates from `active_gate_index`.

Called at ~250 Hz by controller.update(). Color baseline lives on branch
`baseline/color-detection`.
"""

from __future__ import annotations

import math
import time

from simulator.config import ROUND1_GATES
from simulator.transforms import ned_velocity_to_body

CONTROL_DT_S = 1 / 250

# Velocity is ESTIMATED from position differences — the sim's odometry velocity is in an
# unreliable/inverted frame and cannot be trusted for speed or altitude control.
VEL_EST_DT_S = 0.03  # recompute velocity every ~30 ms (smoother than per-250 Hz tick)
VEL_EST_EMA = 0.4  # EMA smoothing factor on the position-derivative
VEL_EST_MAX_JUMP_M = 3.0  # position deltas larger than this = teleport/reset; ignore

# Carrot beyond the gate center along the track direction -> keep flying THROUGH.
LOOKAHEAD_M = 3.0

# Forward drive — a speed controller on PITCH (negative = nose down = accelerate,
# positive = nose up = brake). Active braking is essential: the sim has ~no drag, so
# without it the drone coasts at whatever top speed it reached and overshoots gates.
# Body-frame velocity controller: command a desired velocity toward the carrot, and tilt
# to drive the velocity ERROR to zero. This brakes leftover velocity in ANY direction
# (robust to messy starts), not just when aligned.
CRUISE_SPEED_MPS = 2.5  # max target horizontal speed — slow + precise for Round 1
APPROACH_SLOWDOWN_K = 0.4  # target speed = K * dist-to-gate (slow down as we near a gate)
MIN_SPEED_MPS = 0.8  # don't crawl to a stop at the gate — keep enough to pass through
KP_VEL_TILT = 0.05  # velocity error (m/s) -> tilt (rad)
MAX_FWD_PITCH = 0.12  # cap on forward (accelerate) tilt
MAX_BRAKE_PITCH = 0.25  # cap on nose-up (brake) tilt

# Yaw — only to hold the fixed down-track heading (gentle; not for chasing the gate).
KP_YAW = 0.8
YAW_DEADBAND_RAD = math.radians(4)
MAX_YAW_RATE = 1.5

# Roll — strafe (body-right) to track desired lateral velocity.
MAX_ROLL = 0.12

# Altitude PID (thrust). TRIM = true hover (0.5). D uses the jump-GUARDED velocity estimate
# (vel[2]), so teleport spikes can't launch the drone anymore. Without D the altitude is an
# undamped oscillator (no drag) and bobs ±1 m, which stalls/destabilizes the run.
ALTITUDE_TRIM = 0.5
KP_Z = 0.18
KI_Z = 0.01
KD_Z = 0.10
HOVER_THRUST = 0.5


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Pilot:
    """Yaw-steered, speed-regulated waypoint pilot on known gates."""

    def __init__(self, controller, data):  # type: ignore[type-arg]
        self.controller = controller
        self.data = data
        self._z_integral = 0.0
        self._last_z_target: float | None = None
        self._last_index: int | None = None
        self._race_done = False
        self._used_fallback = False
        self._last_pos: tuple[float, float, float] | None = None
        self._last_vel_t: float = 0.0
        self._vel_est: tuple[float, float, float] = (0.0, 0.0, 0.0)
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0.0, 0.0, 0.0, HOVER_THRUST)
        print("[pilot] waypoint-yaw-steer init; waiting for armed + track data", flush=True)

    # ------------------------------------------------------------------
    def _level_hover(self, z_target: float | None = None) -> None:
        odometry = self.data.get("odometry")
        if odometry is not None and z_target is not None:
            thrust = self._altitude_thrust(odometry["z"], self._vel_est[2], z_target)
        else:
            thrust = HOVER_THRUST
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0.0, 0.0, 0.0, thrust)

    def _gate_center(self, gate):  # type: ignore[no-untyped-def]
        p = gate["position_ned"]
        return (p[0], p[1], p[2])

    def _approach_dir(self, gates, idx, drone_pos):  # type: ignore[no-untyped-def]
        """Down-track through-direction for gate `idx`.

        Derived from gate-to-gate geometry (NOT gate-minus-drone), so it always points
        forward toward the next gate and never flips backward when the drone overshoots —
        that flip was what made the drone yaw ~180 deg ("scanning backward") after a gate.
        """
        center = self._gate_center(gates[idx])
        if idx + 1 < len(gates):
            nxt = self._gate_center(gates[idx + 1])  # exit toward the next gate
            d = (nxt[0] - center[0], nxt[1] - center[1], nxt[2] - center[2])
        elif idx > 0:
            prev = self._gate_center(gates[idx - 1])  # last gate: entry from previous
            d = (center[0] - prev[0], center[1] - prev[1], center[2] - prev[2])
        else:
            d = (center[0] - drone_pos[0], center[1] - drone_pos[1], center[2] - drone_pos[2])
        n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if n < 1e-6:
            return (-1.0, 0.0, 0.0)
        return (d[0] / n, d[1] / n, d[2] / n)

    def _estimate_velocity(self, pos):  # type: ignore[no-untyped-def]
        """NED velocity from position differences (the odometry velocity is unreliable)."""
        now = time.monotonic()
        if self._last_pos is None:
            self._last_pos = pos
            self._last_vel_t = now
            return self._vel_est
        dt = now - self._last_vel_t
        if dt >= VEL_EST_DT_S:
            delta = tuple(pos[i] - self._last_pos[i] for i in range(3))
            jump = math.sqrt(sum(d * d for d in delta))
            if jump > VEL_EST_MAX_JUMP_M:
                # Teleport/reset — don't let it spike the estimate; reseat and decay to 0.
                self._vel_est = (0.0, 0.0, 0.0)
            else:
                raw = tuple(delta[i] / dt for i in range(3))
                self._vel_est = tuple(
                    VEL_EST_EMA * raw[i] + (1.0 - VEL_EST_EMA) * self._vel_est[i]
                    for i in range(3)
                )
            self._last_pos = pos
            self._last_vel_t = now
        return self._vel_est

    def _altitude_thrust(self, z: float, vz_est: float, z_target: float) -> float:
        if self._last_z_target is not None and abs(z_target - self._last_z_target) > 2.0:
            self._z_integral = 0.0
        self._last_z_target = z_target
        error = z - z_target  # NED z down: z>target => too low => need more thrust
        self._z_integral = _clamp(self._z_integral + error * CONTROL_DT_S, -0.5, 0.5)
        # +KD*vz_est: estimated vz is true NED-down rate, so this damps vertical motion.
        raw = ALTITUDE_TRIM + KP_Z * error + KI_Z * self._z_integral + KD_Z * vz_est
        return _clamp(raw, 0.0, 1.0)

    # ------------------------------------------------------------------
    def tick(self) -> None:
        armed = self.data.get("armed", False)
        odometry = self.data.get("odometry")
        gates = self.data.get("track_gates")
        idx = self.data.get("active_gate_index")

        if not armed or odometry is None:
            self._level_hover()
            return

        # Fallback to the known Round-1 track if the live gate table wasn't received. The sim
        # broadcasts it only once at race start, so a pilot started mid-race never gets it —
        # but race_status (active_gate_index) keeps streaming, so idx tells us the race is on.
        if not gates and idx is not None:
            gates = ROUND1_GATES
            if not self._used_fallback:
                self._used_fallback = True
                print("[pilot] no live track data — using known Round-1 gates", flush=True)

        if not gates or idx is None:
            self._level_hover()  # connected, race not started
            return

        if idx >= len(gates):
            if not self._race_done:
                self._race_done = True
                print(f"[pilot] RACE COMPLETE — all {len(gates)} gates", flush=True)
            self._level_hover(odometry["z"])
            return

        if idx != self._last_index:
            print(f"[pilot] TARGET gate {idx}/{len(gates) - 1}", flush=True)
            self._last_index = idx

        pos = (odometry["x"], odometry["y"], odometry["z"])
        vel = self._estimate_velocity(pos)  # NED velocity from position (reliable)
        yaw = self.data.get("yaw_rad", 0.0)

        # Carrot beyond the gate along the track direction.
        center = self._gate_center(gates[idx])
        approach = self._approach_dir(gates, idx, pos)
        carrot = (
            center[0] + approach[0] * LOOKAHEAD_M,
            center[1] + approach[1] * LOOKAHEAD_M,
            center[2] + approach[2] * LOOKAHEAD_M,
        )

        ex, ey = carrot[0] - pos[0], carrot[1] - pos[1]

        # Yaw: DON'T. The drone starts facing down-track (-X, toward every gate) and the
        # body-frame velocity controller below translates in any direction via roll+pitch,
        # so no rotation is needed. Commanding yaw actively spun the drone around — the
        # sim's yaw-rate response is inverted/positive-feedback, so any heading error grew
        # until it faced backwards. Holding heading keeps the camera forward and is stable.
        yaw_rate = 0.0

        # Desired horizontal velocity: toward the carrot, at a speed that tapers down with
        # distance to the gate so we cross slowly and precisely.
        dist_to_gate = math.hypot(center[0] - pos[0], center[1] - pos[1])
        target_speed = _clamp(
            APPROACH_SLOWDOWN_K * dist_to_gate, MIN_SPEED_MPS, CRUISE_SPEED_MPS
        )
        eh = math.hypot(ex, ey)
        if eh > 1e-6:
            vdes_n = ex / eh * target_speed
            vdes_e = ey / eh * target_speed
        else:
            vdes_n = vdes_e = 0.0

        # Velocity error -> body frame -> tilt. Drives velocity to the desired vector, which
        # brakes any leftover velocity (forward OR lateral) regardless of heading.
        verr_fwd, verr_right = ned_velocity_to_body(vdes_n - vel[0], vdes_e - vel[1], yaw)
        pitch = _clamp(-KP_VEL_TILT * verr_fwd, -MAX_FWD_PITCH, MAX_BRAKE_PITCH)
        roll = _clamp(KP_VEL_TILT * verr_right, -MAX_ROLL, MAX_ROLL)

        # Thrust: altitude PID toward the gate altitude (be AT gate height when crossing).
        thrust = self._altitude_thrust(pos[2], vel[2], center[2])

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(roll, pitch, yaw_rate, thrust)
