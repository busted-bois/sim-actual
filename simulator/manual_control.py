"""Keyboard-piloted manual flight — explore the map by hand.

Controls (see CONTROLS_HINT):
  W / S : fly forward / backward
  A / D : strafe left / right
  Q / E : yaw left / right       (turn in place)
  R     : climb
  F     : descend
  C     : recover — hold to unwind the commanded tilt back to level
  L     : auto-land (descend and disarm on touchdown; press again to re-arm)
  - / = : trim hover thrust down / up (find the takeoff/hover point)

Input is captured by a focused Tk window (see simulator/manual_ui.py) so the
simulator window can't steal the keyboard. Horizontal motion is a cascade: the
outer loop holds a fixed cruise speed (CRUISE_SPEED_KMH, 5 km/h) by leaning into
the error against the measured odometry velocity; the inner loop commands body
angular rates to reach that lean. The sim commands body rates (attitude ignored on
the wire) and does not auto-level, so this self-leveling is required.

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

CRUISE_SPEED_KMH = 5.0  # fixed target horizontal speed (W/A/S/D lean toward this)

K_VEL = 0.15  # target lean (rad) per m/s of speed error
MAX_LEAN = 0.15  # rad, cap on the lean the velocity loop may command

YAW_RATE = 0.5  # rad/s while Q/E held (fly2_course's YAW_CLIP)

# Altitude hold — PD on NED z toward a latched target. Constants are the
# flight-proven values from rl/experts/fly2_course.py (Flight-Automation branch); hover
# thrust for this sim was measured there at 0.27 (thrust-accel ~36 m/s^2).
HOVER_T = 0.27  # thrust that holds hover at zero error (and no-telemetry fallback)
KP_Z = 0.025  # thrust per metre of altitude error
# Vertical loop. KD_Z is the R/F feedforward (pressing R adds ~KD_Z*CLIMB_RATE of
# thrust) and the vspeed damping. Bumped hard 2026-07-09 because climb/descend read
# as unnoticeable in-sim — but the bigger cause was R/F being gated behind pose
# telemetry (fixed in _thrust_command: they now also work open-loop via
# VSPEED_OPENLOOP_BIAS). Wide thrust clamps give strong up/down authority.
KD_Z = 0.10  # thrust per m/s of vertical-speed error (damping + climb feedforward)
THRUST_MIN = 0.12  # low floor so F (descend) has strong authority
THRUST_MAX = 0.60  # high ceiling so R (climb) has strong authority
CLIMB_RATE_MPS = 4.0  # R climb target speed (m/s)
DESCEND_RATE_MPS = 4.0  # F descend target speed (m/s)
VSPEED_OPENLOOP_BIAS = 0.12  # thrust bias applied to R/F when pose telemetry is absent
CONTROL_DT_S = 1.0 / CONTROL_HZ  # control period for setpoint slewing

# Auto-land (L): descend at a gentle fixed rate (decoupled from the brisker manual
# DESCEND_RATE_MPS so touchdowns stay soft), then disarm once settled on the ground.
# NED z is down-positive, so a real descent shows up as vz > 0.
LAND_DESCEND_MPS = 1.5  # auto-land descent speed (gentler than manual X)
LAND_DESCEND_VZ = 0.30  # m/s downward we must see first (avoids instant touchdown)
LAND_SETTLE_VZ = 0.10  # m/s; |vz| under this after descending = stopped by ground
LAND_SETTLE_TICKS = max(1, int(0.4 * CONTROL_HZ))  # settled this long -> touchdown
LAND_TIMEOUT_S = 6.0  # failsafe / pose-blocked blind descent: disarm after this

HOVER_TRIM_STEP = 0.02  # thrust change per -/= tap
HOVER_TRIM_LIMIT = 0.35  # max +/- trim around hover

# Attitude (inner) loop. K_ATT was raised from fly2_course's conservative 0.6 to
# 3.0 (2026-07-09) so W/A/S/D drive into their lean about as fast as Q/E yaw:
# pressing W now pitches at ~0.45 rad/s instead of ~0.09. RATE_CLIP raised to match
# YAW_RATE so pitch/roll aren't capped below yaw. Well within the loop's discrete
# stability limit (dt*K_ATT << 1 at 90 Hz); if the drone ever wobbles when hovering
# or braking, lower K_ATT toward ~2.0 — it's the single responsiveness knob.
K_ATT = 3.0  # body-rate (rad/s) per radian of attitude error
RATE_CLIP = 0.5  # max commanded pitch/roll body rate (rad/s), == YAW_RATE

# The one-shot arm at client start can be lost (sent before the sim registers
# us) or undone (sim disarms after a crash), so tick() re-sends ARM until the
# sim's HEARTBEAT reports armed (measured live 2026-07-08: ARM is ACKed and
# sticks when repeated).
ARM_RETRY_S = 1.0

# Pose telemetry counts as "blocked" (event/qualification session) when
# HIGHRES_IMU arrived this recently while ODOMETRY/ATTITUDE stay absent.
IMU_FRESH_S = 2.0

# Command-rate signs per axis. fly2_course used SIGN_ROLL = -1, but on THIS live sim
# that reversed A/D strafe AND made C's roll-leveling positive feedback (ran off to
# one side); flipping to +1 (matching pitch) fixed both — corrected 2026-07-09 from
# live A/D + C evidence. Yaw flipped so Q/E turn as the pilot expects. Flip one if an
# axis flies backwards.
SIGN_ROLL = +1.0
SIGN_PITCH = +1.0
SIGN_YAW = +1.0

# C auto-level = command dead-reckoning. The sim has no auto-leveling: it integrates
# our body-rate commands into attitude and never levels on its own. So from a level
# spawn, roll/pitch == the time-integral of the rate commands WE sent (_cmd_roll /
# _cmd_pitch, kept in tick()). While C is held we command the OPPOSITE of that
# integral, unwinding exactly the tilt A/D/W/S built up — sign-correct by
# construction because it reuses the confirmed wire convention (D -> + -> banks
# right), with no sensor sign to guess. (An accelerometer-based estimate was tried
# first and its sign could never be pinned down live; it's gone.) The integral
# resets while disarmed — the drone spawns/respawns level with motors off.

STATUS_LOG_INTERVAL_S = 1.0

CONTROLS_HINT = (
    "Manual flight: [W/S] fwd/back  [A/D] left/right  [Q/E] turn  "
    "[R] up  [F] down  [C] level  [L] auto-land  [-/=] hover trim  (cruise 5 km/h)"
)

# every key the pilot reads; the Tk input window maintains these as held/not-held
_ALL_KEYS = ("w", "a", "s", "d", "q", "e", "r", "f", "c", "l", "minus", "equal")


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
        self.speed_mps = CRUISE_SPEED_KMH / KMH_PER_MPS  # fixed cruise speed
        self.hover_trim = 0.0  # live thrust trim added to hover
        self.last_cmd = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "thrust": 0.0}
        self._edge = dict.fromkeys(("l", "minus", "equal"), False)
        self._z_target = None  # latched altitude setpoint (NED z, down-positive)
        self._next_arm_t = 0.0  # monotonic time of the next allowed arm retry
        self._arm_announced = False
        self._last_status_log = 0.0
        self._start_t = time.monotonic()
        self._input_seen = False  # have we ever read a keypress?
        # auto-land (L) state machine
        self._landing = False  # actively descending to touch down
        self._landed = False  # touched down + disarmed; suppresses re-arm
        self._land_start_t = 0.0
        self._land_saw_descent = False
        self._land_settle_ticks = 0
        self._recovering = False  # C held: leveling out to recover from an awkward tilt
        # Dead-reckoned attitude: integral of every rate command sent since arming
        # (== physical roll/pitch, since the sim applies our rates and never levels).
        self._cmd_roll = 0.0  # rad, wire-command convention (+ = the way D banks)
        self._cmd_pitch = 0.0  # rad
        print(CONTROLS_HINT, flush=True)
        print(f"[manual] cruise speed = {self.speed_kmh:.1f} km/h", flush=True)

    @property
    def speed_kmh(self):
        return self.speed_mps * KMH_PER_MPS

    # --- telemetry helpers -------------------------------------------------
    def _attitude(self):
        """(roll, pitch, yaw) rad from REAL telemetry (odometry, then ATTITUDE); None
        if pose is blocked. Deliberately no estimated fallback here: feeding an
        estimate into the always-on inner loop spun the drone on spawn. The C
        auto-level instead unwinds the dead-reckoned command integral (see tick())."""
        odo = self.data.get("odometry")
        if odo is not None and "q" in odo:
            return _quat_to_euler(odo["q"])
        att = self.data.get("attitude")
        if att is not None:
            return att["roll"], att["pitch"], att["yaw"]
        return None

    def _attitude_source(self):
        """Which real telemetry _attitude() is using: 'odom' | 'att' | None."""
        odo = self.data.get("odometry")
        if odo is not None and "q" in odo:
            return "odom"
        if self.data.get("attitude") is not None:
            return "att"
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
        if self._rising(keys, "l"):
            self._toggle_land()
        if self._rising(keys, "equal"):
            self.hover_trim = min(self.hover_trim + HOVER_TRIM_STEP, HOVER_TRIM_LIMIT)
            print(f"[manual] hover trim = {self.hover_trim:+.2f}", flush=True)
        if self._rising(keys, "minus"):
            self.hover_trim = max(self.hover_trim - HOVER_TRIM_STEP, -HOVER_TRIM_LIMIT)
            print(f"[manual] hover trim = {self.hover_trim:+.2f}", flush=True)

    # --- auto-land (L) -----------------------------------------------------
    def _toggle_land(self):
        if self._landing or self._landed:
            self._cancel_land()
        else:
            self._start_land()

    def _start_land(self):
        self._landing = True
        self._landed = False
        self._land_start_t = time.monotonic()
        self._land_saw_descent = False
        self._land_settle_ticks = 0
        print("[manual] auto-land: descending to touchdown…", flush=True)

    def _cancel_land(self):
        was_landed = self._landed
        self._landing = False
        self._landed = False
        self._land_settle_ticks = 0
        self._z_target = None  # re-latch the hold altitude on the next tick
        if was_landed:
            self._next_arm_t = 0.0  # re-arm promptly
            print("[manual] auto-land canceled — re-arming.", flush=True)
        else:
            print("[manual] auto-land canceled.", flush=True)

    def _touchdown(self):
        self._landing = False
        self._landed = True
        self.controller.disarm()
        print("[manual] touchdown — disarmed. Press L to re-arm.", flush=True)

    def _ensure_armed(self):
        """Re-send ARM (throttled) until the sim's HEARTBEAT reports armed."""
        if self._landed:
            return  # deliberately on the ground — don't fight the auto-land disarm
        if self.data.get("armed") is True:
            return
        now = time.monotonic()
        if now < self._next_arm_t:
            return
        self._next_arm_t = now + ARM_RETRY_S
        if not self._arm_announced:
            self._arm_announced = True
            print(
                f"[manual] arming — retrying every {ARM_RETRY_S:.0f}s until the "
                "sim reports armed",
                flush=True,
            )
        self.controller.arm()

    def _pose_blocked(self):
        """True when IMU streams but real pose (ODOMETRY/ATTITUDE/LOCAL_POS) doesn't
        — event/qualification sessions block it. The IMU-derived attitude estimate
        does NOT count as pose here, so the HUD still flags the block."""
        has_real_pose = (
            self.data.get("odometry") is not None
            or self.data.get("attitude") is not None
            or self.data.get("local_position_ned") is not None
        )
        if has_real_pose:
            return False
        imu_mono = self.data.get("highres_imu_mono")
        return imu_mono is not None and time.monotonic() - imu_mono < IMU_FRESH_S

    # --- control law -------------------------------------------------------
    def tick(self):
        self._ensure_armed()
        keys = {k: bool(self._is_pressed(k)) for k in _ALL_KEYS}
        if any(keys.values()):
            self._input_seen = True
        self._update_discrete(keys)

        # Dead-reckoned attitude re-zeroes while disarmed: motors are off, the drone
        # sits/spawns level, so the command integral must restart from level too.
        if self.data.get("armed") is not True:
            self._cmd_roll = 0.0
            self._cmd_pitch = 0.0

        # Landed: sit disarmed on the ground, motors off, ignore flight inputs.
        if self._landed:
            self.last_cmd = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "thrust": 0.0}
            self.controller.send_attitude_rates(0.0, 0.0, 0.0, 0.0)
            self._maybe_log(keys, 0.0, 0.0, 0.0)
            return

        att = self._attitude()
        roll, pitch, yaw = att if att is not None else (0.0, 0.0, 0.0)

        # C = recover: command the OPPOSITE of the dead-reckoned attitude — literally
        # the negative of the accumulated A/D/W/S roll+pitch commands — until it
        # unwinds to level, ignoring the movement keys. No sensor and no sign flags:
        # _cmd_roll is already in wire convention, so "opposite" is just the minus.
        self._recovering = keys["c"]

        if self._recovering:
            roll_cmd = _clip(-K_ATT * self._cmd_roll, -RATE_CLIP, RATE_CLIP)
            pitch_cmd = _clip(-K_ATT * self._cmd_pitch, -RATE_CLIP, RATE_CLIP)
        else:
            # While auto-landing, self-level and ignore horizontal input.
            if self._landing:
                tgt_pitch = tgt_roll = 0.0
            else:
                tgt_pitch, tgt_roll = self._target_lean(keys, yaw)
            roll_cmd = SIGN_ROLL * _clip(
                K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP
            )
            pitch_cmd = SIGN_PITCH * _clip(
                K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP
            )

        yaw_cmd = 0.0
        if not self._landing and not self._recovering:
            if keys["q"]:
                yaw_cmd -= YAW_RATE
            if keys["e"]:
                yaw_cmd += YAW_RATE
            yaw_cmd *= SIGN_YAW

        thrust = self._thrust_command(keys)

        # Dead-reckon: fold the rates we are sending into the attitude integral.
        # The sim applies exactly these rates and never self-levels, so this stays
        # equal to the physical roll/pitch (from the level, disarmed start).
        self._cmd_roll += roll_cmd * CONTROL_DT_S
        self._cmd_pitch += pitch_cmd * CONTROL_DT_S

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

        if self._landing:
            return self._land_thrust(z, vz)

        # R/F command a vertical speed. Set unconditionally (even with no pose
        # telemetry) so climb/descend always respond — _altitude_thrust falls back
        # to an open-loop bias when blind. NED z is down-positive, so climb = -vz_des.
        vz_des = 0.0
        if keys["r"]:
            vz_des -= CLIMB_RATE_MPS
        if keys["f"]:
            vz_des += DESCEND_RATE_MPS

        # When we have telemetry, latch a hold altitude and slew it while R/F are
        # held, so releasing holds the altitude we climbed to.
        if z is not None and self._z_target is None:
            self._z_target = z
        if self._z_target is not None:
            self._z_target += vz_des * CONTROL_DT_S

        return self._altitude_thrust(z, vz, vz_des)

    def _land_thrust(self, z, vz):
        """Auto-land: slew the setpoint down at the gentle landing rate, then check
        for touchdown. Returns the commanded thrust."""
        if z is not None and self._z_target is None:
            self._z_target = z
        if self._z_target is not None:
            self._z_target += LAND_DESCEND_MPS * CONTROL_DT_S
        self._check_touchdown(vz)
        return self._altitude_thrust(z, vz, LAND_DESCEND_MPS)

    def _check_touchdown(self, vz):
        """Disarm once the drone has descended and then stopped moving.

        NED z is down-positive, so a descent shows up as vz > 0. Wait to see a real
        descent, then a sustained near-zero vz (the ground stopping us). The
        wall-clock timeout is the failsafe and the pose-blocked (no vz) path.
        """
        if time.monotonic() - self._land_start_t >= LAND_TIMEOUT_S:
            self._touchdown()
            return
        if vz is None:
            return  # no velocity telemetry — rely on the timeout above
        if vz > LAND_DESCEND_VZ:
            self._land_saw_descent = True
        if self._land_saw_descent and abs(vz) < LAND_SETTLE_VZ:
            self._land_settle_ticks += 1
            if self._land_settle_ticks >= LAND_SETTLE_TICKS:
                self._touchdown()
        else:
            self._land_settle_ticks = 0

    def _altitude_thrust(self, z, vz, vz_des):
        """PD on NED z toward the latched setpoint; open-loop climb/descend if blind.

        vz_des is the wanted vertical rate while R/F are held, so the damping term
        pulls toward the commanded climb/descent instead of fighting it. With no pose
        telemetry (event session blocks it) the PD can't run, so R/F instead apply a
        fixed thrust bias in the commanded direction — uncapped (no vspeed feedback
        to hold a rate) but responsive, so climb/descend still work.
        """
        if z is None or self._z_target is None:
            bias = 0.0
            if vz_des < 0.0:  # climb
                bias = VSPEED_OPENLOOP_BIAS
            elif vz_des > 0.0:  # descend
                bias = -VSPEED_OPENLOOP_BIAS
            return _clip(HOVER_T + self.hover_trim + bias, THRUST_MIN, THRUST_MAX)
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
        # Display attitude: real telemetry if present, else the dead-reckoned command
        # integral — exactly what the C auto-level unwinds, so the HUD shows what C
        # will act on (should read ~0 at a level hover, and track A/D/W/S banks).
        if att is not None:
            disp_roll, disp_pitch, disp_yaw = att
            disp_src = self._attitude_source()
        else:
            disp_roll, disp_pitch, disp_yaw, disp_src = (
                self._cmd_roll,
                self._cmd_pitch,
                None,  # yaw isn't dead-reckoned (C doesn't touch heading)
                "cmd",
            )
        z, vz = self._altitude()
        vel = self._velocity_ned()
        hspeed = math.hypot(*vel) if vel is not None else None
        return {
            "speed_kmh": self.speed_kmh,
            "hover_trim": self.hover_trim,
            "mode": (
                "LANDED"
                if self._landed
                else "LANDING"
                if self._landing
                else "RECOVER"
                if self._recovering
                else "FLY"
            ),
            "input_seen": self._input_seen,
            "armed": self.data.get("armed"),
            "have_telemetry": att is not None or z is not None,
            "pose_blocked": self._pose_blocked(),
            "att_source": disp_src,  # 'odom' | 'att' | 'cmd' (dead-reckoned display)
            "cmd": dict(self.last_cmd),
            "roll_deg": math.degrees(disp_roll) if disp_roll is not None else None,
            "pitch_deg": math.degrees(disp_pitch) if disp_pitch is not None else None,
            "yaw_deg": math.degrees(disp_yaw) if disp_yaw is not None else None,
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
            "r": "R",
            "f": "F",
            "c": "C",
            "l": "L",
        }
        held = " ".join(v for k, v in labels.items() if keys[k]) or "-"
        z, _ = self._altitude()
        z_str = f"{z:.1f}" if z is not None else "n/a"
        print(
            f"[manual] keys={held} spd={self.speed_kmh:.1f}km/h thrust={thrust:.3f} "
            f"z={z_str} roll={math.degrees(roll):+.0f} pitch={math.degrees(pitch):+.0f}",
            flush=True,
        )
