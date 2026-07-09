"""Keyboard-piloted manual flight — explore the map by hand.

Controls (see CONTROLS_HINT):
  W / S : fly forward / backward
  A / D : strafe left / right
  Q / E : yaw left / right       (turn in place)
  SPACE : climb
  X     : descend
  R / F : increase / decrease flight speed (tap to step)
  - / = : trim hover thrust down / up (find the takeoff/hover point)

Input is captured by a focused Tk window (see simulator/manual_ui.py) so the
simulator window can't steal the keyboard. Horizontal motion is a cascade: the
outer loop holds a *target speed* (default 5 km/h, R/F) by leaning into the error
against the measured odometry velocity; the inner loop commands body angular rates
to reach that lean. The sim commands body rates (attitude ignored on the wire) and
does not auto-level, so this self-leveling is required.

Signs, hover thrust and gains are the constants below — tune them live with the
HUD (see the plan's verification steps). The `- / =` keys trim hover thrust in
flight so you can find the value that actually holds altitude.
"""

from __future__ import annotations

import math
import time

from simulator.controller import CONTROL_HZ

# --------------------------------------------------------------------------------------
# Tunables (start conservative; refine in-sim with the HUD)
# --------------------------------------------------------------------------------------
KMH_PER_MPS = 3.6

DEFAULT_SPEED_KMH = 5.0  # target horizontal speed at startup
SPEED_STEP_KMH = 1.0  # change per R (up) / F (down) tap
SPEED_MIN_KMH = 1.0
SPEED_MAX_KMH = 30.0

K_VEL = 0.15  # target lean (rad) per m/s of speed error
MAX_LEAN = 0.15  # rad, cap on the lean the velocity loop may command

YAW_RATE = 0.5  # rad/s while Q/E held (fly2_course's YAW_CLIP)

# Altitude hold — PD on NED z toward a latched target. Constants are the
# flight-proven values from rl/fly2_course.py (Flight-Automation branch); hover
# thrust for this sim was measured there at 0.27 (thrust-accel ~36 m/s^2).
HOVER_T = 0.27  # thrust that holds hover at zero error (and no-telemetry fallback)
KP_Z = 0.025  # thrust per metre of altitude error
KD_Z = 0.030  # thrust per m/s of vertical-speed error (damping)
THRUST_MIN = 0.18
THRUST_MAX = 0.5
CLIMB_RATE_MPS = 2.0  # how fast SPACE raises the altitude setpoint
DESCEND_RATE_MPS = 2.0  # how fast X lowers the altitude setpoint
CONTROL_DT_S = 1.0 / CONTROL_HZ  # control period for setpoint slewing

HOVER_TRIM_STEP = 0.02  # thrust change per -/= tap
HOVER_TRIM_LIMIT = 0.35  # max +/- trim around hover

K_ATT = 0.6  # body-rate per radian of attitude error (fly2_course)
RATE_CLIP = 0.30  # max commanded body rate (rad/s)

# Command-rate signs vs the ODOMETRY attitude convention, measured live by
# fly2_course: pitch normal; roll and yaw inverted. Valid because _attitude()
# prefers the sim's odometry quaternion. Flip one if an axis flies backwards.
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0

STATUS_LOG_INTERVAL_S = 1.0

CONTROLS_HINT = (
    "Manual flight: [W/S] fwd/back  [A/D] left/right  [Q/E] turn  "
    "[SPACE] up  [X] down  [R/F] speed +/-  [-/=] hover trim"
)

# every key the pilot reads; the Tk input window maintains these as held/not-held
_ALL_KEYS = ("w", "a", "s", "d", "q", "e", "space", "x", "r", "f", "minus", "equal")


def _default_is_pressed():
    """Return keyboard.is_pressed, or a no-op that reports nothing pressed."""
    try:
        import keyboard
    except (ImportError, OSError):  # pragma: no cover - e.g. non-root linux
        print(
            "[manual] 'keyboard' unavailable — no input will be read. "
            "Install it (and on Linux run as root).",
            flush=True,
        )
        return lambda _key: False
    return keyboard.is_pressed


def _quat_to_euler(q):
    """(w, x, y, z) quaternion -> (roll, pitch, yaw) in radians (aerospace)."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class ManualControl:
    """Keyboard state -> attitude-rate + thrust commands, with velocity hold + alt-hold."""

    def __init__(self, controller, data, is_pressed=None):
        self.controller = controller
        self.data = data
        self._is_pressed = (
            is_pressed if is_pressed is not None else _default_is_pressed()
        )
        self.speed_mps = DEFAULT_SPEED_KMH / KMH_PER_MPS  # target horizontal speed
        self.hover_trim = 0.0  # live thrust trim added to hover
        self.last_cmd = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "thrust": 0.0}
        self._edge = dict.fromkeys(("r", "f", "minus", "equal"), False)
        self._z_target = None  # latched altitude setpoint (NED z, down-positive)
        self._last_status_log = 0.0
        self._start_t = time.monotonic()
        self._input_seen = False  # have we ever read a keypress?
        print(CONTROLS_HINT, flush=True)
        print(f"[manual] speed = {self.speed_kmh:.1f} km/h", flush=True)

    @property
    def speed_kmh(self):
        return self.speed_mps * KMH_PER_MPS

    # --- telemetry helpers -------------------------------------------------
    def _attitude(self):
        """(roll, pitch, yaw) rad, preferring odometry; None if unavailable."""
        odo = self.data.get("odometry")
        if odo is not None and "q" in odo:
            return _quat_to_euler(odo["q"])
        att = self.data.get("attitude")
        if att is not None:
            return att["roll"], att["pitch"], att["yaw"]
        return None

    def _altitude(self):
        """(z, vz) in NED (down-positive); (None, None) if unavailable."""
        src = self.data.get("odometry") or self.data.get("local_position_ned")
        if src is None:
            return None, None
        return src.get("z"), src.get("vz")

    def _velocity_ned(self):
        """(vx, vy) horizontal velocity in NED, or None if unavailable."""
        src = self.data.get("odometry") or self.data.get("local_position_ned")
        if src is None:
            return None
        vx, vy = src.get("vx"), src.get("vy")
        if vx is None or vy is None:
            return None
        return vx, vy

    # --- discrete (tap) inputs: speed + hover trim -------------------------
    def _rising(self, keys, name):
        """True on the tick a key transitions from up to down (edge-triggered)."""
        down = keys[name]
        edge = down and not self._edge[name]
        self._edge[name] = down
        return edge

    def _update_discrete(self, keys):
        step = SPEED_STEP_KMH / KMH_PER_MPS
        lo, hi = SPEED_MIN_KMH / KMH_PER_MPS, SPEED_MAX_KMH / KMH_PER_MPS
        if self._rising(keys, "r"):
            self.speed_mps = min(self.speed_mps + step, hi)
            print(f"[manual] speed = {self.speed_kmh:.1f} km/h", flush=True)
        if self._rising(keys, "f"):
            self.speed_mps = max(self.speed_mps - step, lo)
            print(f"[manual] speed = {self.speed_kmh:.1f} km/h", flush=True)
        if self._rising(keys, "equal"):
            self.hover_trim = min(self.hover_trim + HOVER_TRIM_STEP, HOVER_TRIM_LIMIT)
            print(f"[manual] hover trim = {self.hover_trim:+.2f}", flush=True)
        if self._rising(keys, "minus"):
            self.hover_trim = max(self.hover_trim - HOVER_TRIM_STEP, -HOVER_TRIM_LIMIT)
            print(f"[manual] hover trim = {self.hover_trim:+.2f}", flush=True)

    # --- control law -------------------------------------------------------
    def tick(self):
        keys = {k: bool(self._is_pressed(k)) for k in _ALL_KEYS}
        if any(keys.values()):
            self._input_seen = True
        self._update_discrete(keys)

        att = self._attitude()
        roll, pitch, yaw = att if att is not None else (0.0, 0.0, 0.0)

        tgt_pitch, tgt_roll = self._target_lean(keys, yaw)

        roll_cmd = SIGN_ROLL * _clip(K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        pitch_cmd = SIGN_PITCH * _clip(
            K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP
        )

        yaw_cmd = 0.0
        if keys["q"]:
            yaw_cmd -= YAW_RATE
        if keys["e"]:
            yaw_cmd += YAW_RATE
        yaw_cmd *= SIGN_YAW

        thrust = self._thrust_command(keys)

        self.last_cmd = {
            "roll": roll_cmd,
            "pitch": pitch_cmd,
            "yaw": yaw_cmd,
            "thrust": thrust,
        }
        self.controller.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        self._maybe_log(keys, thrust, roll, pitch)

    def _target_lean(self, keys, yaw):
        """Outer velocity loop: desired body speed -> target (pitch, roll) lean."""
        fwd_t = self.speed_mps * (keys["w"] - keys["s"])
        lat_t = self.speed_mps * (keys["d"] - keys["a"])

        vel = self._velocity_ned()
        if vel is not None:
            vx, vy = vel
            fwd_m = vx * math.cos(yaw) + vy * math.sin(yaw)
            lat_m = -vx * math.sin(yaw) + vy * math.cos(yaw)
        else:
            fwd_m = lat_m = 0.0

        # forward speed error -> nose-down (negative) pitch; right speed -> right roll
        tgt_pitch = -_clip(K_VEL * (fwd_t - fwd_m), -MAX_LEAN, MAX_LEAN)
        tgt_roll = _clip(K_VEL * (lat_t - lat_m), -MAX_LEAN, MAX_LEAN)
        return tgt_pitch, tgt_roll

    def _thrust_command(self, keys):
        z, vz = self._altitude()

        # SPACE/X move the altitude setpoint up/down; otherwise it stays latched
        # (hold). NED z is down-positive, so climbing means a more-negative target.
        if z is not None and self._z_target is None:
            self._z_target = z
        vz_des = 0.0
        if keys["space"] and self._z_target is not None:
            self._z_target -= CLIMB_RATE_MPS * CONTROL_DT_S
            vz_des = -CLIMB_RATE_MPS
        if keys["x"] and self._z_target is not None:
            self._z_target += DESCEND_RATE_MPS * CONTROL_DT_S
            vz_des = DESCEND_RATE_MPS

        return self._altitude_thrust(z, vz, vz_des)

    def _altitude_thrust(self, z, vz, vz_des):
        """PD on NED z toward the latched setpoint; open-loop fallback if blind.

        vz_des is the wanted vertical rate while SPACE/X are held, so the damping
        term pulls toward the commanded climb/descent instead of fighting it.
        """
        if z is None or self._z_target is None:
            return _clip(HOVER_T + self.hover_trim, THRUST_MIN, THRUST_MAX)
        vz = vz if vz is not None else 0.0
        thrust = (
            HOVER_T
            + self.hover_trim
            + KP_Z * (z - self._z_target)
            + KD_Z * (vz - vz_des)
        )
        return _clip(thrust, THRUST_MIN, THRUST_MAX)

    # --- HUD / logging -----------------------------------------------------
    def status(self):
        """Snapshot for the HUD: setpoints, last command, and telemetry response."""
        att = self._attitude()
        z, vz = self._altitude()
        vel = self._velocity_ned()
        hspeed = math.hypot(*vel) if vel is not None else None
        return {
            "speed_kmh": self.speed_kmh,
            "hover_trim": self.hover_trim,
            "input_seen": self._input_seen,
            "armed": self.data.get("armed"),
            "have_telemetry": att is not None or z is not None,
            "cmd": dict(self.last_cmd),
            "roll_deg": math.degrees(att[0]) if att is not None else None,
            "pitch_deg": math.degrees(att[1]) if att is not None else None,
            "yaw_deg": math.degrees(att[2]) if att is not None else None,
            "alt_m": (-z) if z is not None else None,  # up-positive for display
            "vz_mps": vz,
            "hspeed_kmh": hspeed * KMH_PER_MPS if hspeed is not None else None,
        }

    def _maybe_log(self, keys, thrust, roll, pitch):
        now = time.monotonic()
        if now - self._last_status_log < STATUS_LOG_INTERVAL_S:
            return
        self._last_status_log = now
        labels = {
            "w": "W",
            "a": "A",
            "s": "S",
            "d": "D",
            "q": "Q",
            "e": "E",
            "space": "SPC",
            "x": "X",
        }
        held = " ".join(v for k, v in labels.items() if keys[k]) or "-"
        z, _ = self._altitude()
        z_str = f"{z:.1f}" if z is not None else "n/a"
        print(
            f"[manual] keys={held} spd={self.speed_kmh:.1f}km/h thrust={thrust:.3f} "
            f"z={z_str} roll={math.degrees(roll):+.0f} pitch={math.degrees(pitch):+.0f}",
            flush=True,
        )
