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
import os
import time

import numpy as np

from simulator.config import ROUND1_GATES
from simulator.rl_core import (
    ABS_MAX_TARGET_SPEED_MPS,
    RL_DT_S,
    GuidanceResidual,
    action_to_residual,
    build_observation,
    gate_normal,
    identity_residual,
    is_flyaway,
)
from simulator.transforms import ned_velocity_to_body

CONTROL_DT_S = 1 / 250

# Path to a trained residual policy. If present at startup, the pilot self-loads it and
# applies the learned guidance residual (P3 deploy). Absent/broken -> bare pilot.
POLICY_PATH = os.environ.get("RL_POLICY_PATH", "policy.pt")

# Velocity is ESTIMATED from position differences — the sim's odometry velocity is in an
# unreliable/inverted frame and cannot be trusted for speed or altitude control.
VEL_EST_DT_S = 0.03  # recompute velocity every ~30 ms (smoother than per-250 Hz tick)
VEL_EST_EMA = 0.4  # EMA smoothing factor on the position-derivative
VEL_EST_MAX_JUMP_M = 3.0  # position deltas larger than this = teleport/reset; ignore

# Carrot beyond the gate center along the track direction -> keep flying THROUGH.
LOOKAHEAD_M = 3.0
# Home to the EXACT gate center on final approach: shrink the look-ahead to 0 as we close in,
# so the carrot doesn't aim past the gate toward the next one and cut the corner (which made
# us clip the gate frame laterally). Beyond this range, full look-ahead for a smooth line.
HOMING_DIST_M = 6.0

# Forward drive — a speed controller on PITCH (negative = nose down = accelerate,
# positive = nose up = brake). Active braking is essential: the sim has ~no drag, so
# without it the drone coasts at whatever top speed it reached and overshoots gates.
# Body-frame velocity controller: command a desired velocity toward the carrot, and tilt
# to drive the velocity ERROR to zero. This brakes leftover velocity in ANY direction
# (robust to messy starts), not just when aligned.
CRUISE_SPEED_MPS = 2.5  # max target horizontal speed — slow + precise for Round 1
APPROACH_SLOWDOWN_K = (
    0.4  # target speed = K * dist-to-gate (slow down as we near a gate)
)
MIN_SPEED_MPS = 0.8  # don't crawl to a stop at the gate — keep enough to pass through
KP_VEL_TILT = 0.05  # velocity error (m/s) -> tilt (rad)
MAX_FWD_PITCH = 0.12  # cap on forward (accelerate) tilt
MAX_BRAKE_PITCH = 0.25  # cap on nose-up (brake) tilt

# Yaw — ACTIVELY hold the down-track heading (P0.5 fix for the gate-2 orbit). Leaving yaw
# free (yaw_rate=0) clears g0-1 but at g2 the heading slowly drifts from roll/pitch coupling,
# the body frame rotates, and the velocity controller spirals into an orbit. We hold yaw at
# the gate-to-gate down-track bearing. The sim's yaw-rate response is INVERTED (commanding
# +rate yaws the nose the other way — methodology bug #7), so the correcting command uses
# YAW_HOLD_SIGN = -1. Gentle gain + deadband keep it near the proven baseline.
# NOTE: sign is unverified in sim — validate in P0.5 (`make sim`); set False to revert to the
# known-good yaw_rate=0 baseline if it regresses.
YAW_HOLD_ENABLED = True
YAW_HOLD_SIGN = -1.0
KP_YAW = 0.5
YAW_DEADBAND_RAD = math.radians(4)
MAX_YAW_RATE = 1.0

# Roll — strafe (body-right) to track desired lateral velocity.
MAX_ROLL = 0.12

# Altitude PID (thrust). TRIM = true hover (0.5). D uses the jump-GUARDED velocity estimate
# (vel[2]), so teleport spikes can't launch the drone anymore. Without D the altitude is an
# undamped oscillator (no drag) and bobs ±1 m, which stalls/destabilizes the run.
ALTITUDE_TRIM = 0.5
KP_Z = 0.18
KI_Z = (
    0.03  # was 0.01 — too weak; left a ~1.3 m steady-state altitude error -> gate clips
)
KD_Z = (
    0.05  # was 0.10 — over-damped the descent (braked it ~1.3 m above descending gates)
)
HOVER_THRUST = 0.5

# Vertical-alignment gating: don't fly horizontally INTO a gate we're not level with. When
# the altitude error exceeds the gate aperture, slow the horizontal approach so altitude can
# catch up — otherwise we arrive above/below the hoop and clip its frame (every gate after
# gate 0 descends, so this bit hard).
VERT_ALIGN_TOL_M = (
    2.5  # only slow when WELL beyond the aperture (~1.36 m) + normal ~1.1 m offset
)
VERT_ALIGN_SLOW = 0.35  # horizontal-speed multiplier while badly off altitude


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _perp_right_horizontal(appr) -> tuple[float, float]:
    """Horizontal NED unit vector pointing to body-right of the down-track direction.

    Right of forward (n, e) in the NED horizontal plane is (-e, n).
    """
    n, e = appr[0], appr[1]
    mag = math.hypot(n, e)
    if mag < 1e-6:
        return (0.0, 0.0)
    return (-e / mag, n / mag)


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

        # --- residual-RL guidance layer ---
        # The current guidance residual laid over the pilot. Identity = bare pilot.
        # When `external_residual` is True (the RL Gym env drives training), the env owns
        # `self.residual` and the pilot will NOT self-run a deployed policy.
        self.residual: GuidanceResidual = identity_residual()
        self.external_residual: bool = False
        self.last_action = np.zeros(3, dtype=np.float32)
        self._policy = None
        self._policy_last_t: float = 0.0
        self._policy_failed: bool = False
        self._flyaway_logged: bool = False

        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0.0, 0.0, 0.0, HOVER_THRUST)
        print(
            "[pilot] waypoint-yaw-steer init; waiting for armed + track data",
            flush=True,
        )
        self._maybe_load_policy()

    # ------------------------------------------------------------------
    def _maybe_load_policy(self) -> None:
        """Auto-load a trained residual policy for deploy (P3). Lazy torch import so
        `make sim` works with no RL deps when there's no policy. Any failure -> bare pilot."""
        if self.external_residual or not os.path.exists(POLICY_PATH):
            return
        try:
            from simulator.policy_runtime import PolicyRuntime

            self._policy = PolicyRuntime.load(POLICY_PATH)
            print(
                f"[pilot] loaded residual policy '{POLICY_PATH}' (deploy mode)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — never let a bad model break flight
            self._policy = None
            print(f"[pilot] no residual policy ({e}); flying bare pilot", flush=True)

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
            d = (
                center[0] - drone_pos[0],
                center[1] - drone_pos[1],
                center[2] - drone_pos[2],
            )
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
        if (
            self._last_z_target is not None
            and abs(z_target - self._last_z_target) > 2.0
        ):
            self._z_integral = 0.0
        self._last_z_target = z_target
        error = z - z_target  # NED z down: z>target => too low => need more thrust
        self._z_integral = _clamp(self._z_integral + error * CONTROL_DT_S, -1.0, 1.0)
        # +KD*vz_est: estimated vz is true NED-down rate, so this damps vertical motion.
        raw = ALTITUDE_TRIM + KP_Z * error + KI_Z * self._z_integral + KD_Z * vz_est
        return _clamp(raw, 0.0, 1.0)

    def _yaw_hold_rate(self, approach, yaw: float) -> float:
        """Body yaw-rate command to hold the down-track bearing (P0.5 gate-2 fix)."""
        if not YAW_HOLD_ENABLED:
            return 0.0
        desired = math.atan2(approach[1], approach[0])  # NED bearing of the through-dir
        err = _wrap_pi(desired - yaw)
        if abs(err) < YAW_DEADBAND_RAD:
            return 0.0
        return _clamp(YAW_HOLD_SIGN * KP_YAW * err, -MAX_YAW_RATE, MAX_YAW_RATE)

    def _maybe_run_policy(self, pos, vel, yaw: float, gates, idx: int) -> None:
        """Deploy: run the loaded residual policy at the RL cadence to refresh
        self.residual. No-op during training (env owns the residual) or with no policy.
        A watchdog disables the policy and reverts to the bare pilot on any fault."""
        if self._policy is None or self.external_residual:
            return
        now = time.monotonic()
        if now - self._policy_last_t < RL_DT_S:
            return
        self._policy_last_t = now
        try:
            att = self.data.get("attitude") or {}
            roll = float(att.get("roll", 0.0))
            pitch = float(att.get("pitch", 0.0))
            obs = build_observation(
                pos, vel, roll, pitch, yaw, gates, idx, self.residual, self.last_action
            )
            action = np.asarray(self._policy.act(obs), dtype=np.float32).reshape(-1)
            if action.shape[0] != 3 or not np.all(np.isfinite(action)):
                raise ValueError("policy produced an invalid action")
            self.last_action = action
            self.residual = action_to_residual(action)
        except Exception as e:  # noqa: BLE001 — a model fault must never DNF the race
            self._policy = None
            self.residual = identity_residual()
            if not self._policy_failed:
                self._policy_failed = True
                print(
                    f"[pilot] residual policy disabled ({e}); reverting to bare pilot",
                    flush=True,
                )

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
                print(
                    "[pilot] no live track data — using known Round-1 gates", flush=True
                )

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

        # Deploy: refresh the learned guidance residual at the RL cadence (no-op in
        # training, where the env owns self.residual, or when no policy is loaded).
        self._maybe_run_policy(pos, vel, yaw, gates, idx)
        res = self.residual

        center = self._gate_center(gates[idx])

        # Fly-away guard: if we're implausibly far from the target gate, the run has gone
        # unstable (or the frame is bad). Arrest to a level hover instead of commanding
        # toward a far carrot — a runaway is what corrupts the sim's coordinate frame.
        if is_flyaway(pos, center):
            if not self._flyaway_logged:
                self._flyaway_logged = True
                print(
                    f"[pilot] FLY-AWAY GUARD: far from gate {idx} (pos={tuple(round(p, 1) for p in pos)});"
                    " arresting to hover",
                    flush=True,
                )
            self.controller.set_control_mode("attitude")
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, HOVER_THRUST)
            return
        self._flyaway_logged = False

        # Look-through along the gate's OWN normal (perpendicular to its plane), not toward
        # the next gate. Crossing along the normal keeps the carrot's y,z locked on the exact
        # gate center, so every gate is threaded through its middle instead of cutting the
        # corner toward the next gate (which clipped the frame laterally). approach (gate-to-
        # gate) is kept only to orient the normal's sign and hold yaw down-track.
        approach = self._approach_dir(gates, idx, pos)
        normal = gate_normal(gates[idx], travel_hint=approach)
        perp_n, perp_e = _perp_right_horizontal(normal)

        # Home to the EXACT center on final approach: shrink the look-ahead to 0 as we close
        # in, so we cross dead-center.
        dist3 = math.sqrt(
            (center[0] - pos[0]) ** 2
            + (center[1] - pos[1]) ** 2
            + (center[2] - pos[2]) ** 2
        )
        homing = _clamp(dist3 / HOMING_DIST_M, 0.0, 1.0)
        eff_lookahead = res.lookahead_m * homing
        carrot = (
            center[0] + normal[0] * eff_lookahead + perp_n * res.lateral_offset_m,
            center[1] + normal[1] * eff_lookahead + perp_e * res.lateral_offset_m,
            center[2] + normal[2] * eff_lookahead,
        )

        ex, ey = carrot[0] - pos[0], carrot[1] - pos[1]

        # Yaw: actively hold the down-track bearing (P0.5). Bare-pilot baseline was
        # yaw_rate=0; the active hold stops the gate-2 orbit. Uses the inverted sign
        # (sim bug #7) and a deadband so small errors don't thrash.
        yaw_rate = self._yaw_hold_rate(approach, yaw)

        # Desired horizontal velocity: toward the carrot, at a speed that tapers down with
        # distance to the gate so we cross slowly and precisely. The residual speed_mult
        # lets the policy push faster on straights or brake harder for tight gates; an
        # absolute ceiling caps it regardless.
        dist_to_gate = math.hypot(center[0] - pos[0], center[1] - pos[1])
        target_speed = _clamp(
            APPROACH_SLOWDOWN_K * dist_to_gate, MIN_SPEED_MPS, CRUISE_SPEED_MPS
        )
        target_speed = _clamp(
            target_speed * res.speed_mult, 0.0, ABS_MAX_TARGET_SPEED_MPS
        )

        # Vertical-alignment gating: only when BADLY off the gate altitude (well beyond the
        # aperture) do we slow the horizontal approach, so we descend/climb INTO the hoop
        # instead of clipping its frame. Tolerance must exceed the drone's normal ~1.1 m
        # altitude offset (and the ~1.36 m aperture) or it stalls on every gate.
        if abs(pos[2] - center[2]) > VERT_ALIGN_TOL_M:
            target_speed *= VERT_ALIGN_SLOW
        eh = math.hypot(ex, ey)
        if eh > 1e-6:
            vdes_n = ex / eh * target_speed
            vdes_e = ey / eh * target_speed
        else:
            vdes_n = vdes_e = 0.0

        # Velocity error -> body frame -> tilt. Drives velocity to the desired vector, which
        # brakes any leftover velocity (forward OR lateral) regardless of heading.
        verr_fwd, verr_right = ned_velocity_to_body(
            vdes_n - vel[0], vdes_e - vel[1], yaw
        )
        pitch = _clamp(-KP_VEL_TILT * verr_fwd, -MAX_FWD_PITCH, MAX_BRAKE_PITCH)
        roll = _clamp(KP_VEL_TILT * verr_right, -MAX_ROLL, MAX_ROLL)

        # Thrust: altitude PID toward the gate altitude (be AT gate height when crossing).
        thrust = self._altitude_thrust(pos[2], vel[2], center[2])

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(roll, pitch, yaw_rate, thrust)
