import math

try:
    import keyboard
except (ImportError, OSError):  # pragma: no cover - e.g. non-root linux
    keyboard = None

from simulator.pilot import HOVER_THRUST

MANUAL_TOGGLE_KEY = "n"

HORIZONTAL_SPEED_M_S = 3.0
VERTICAL_SPEED_M_S = 2.0
YAW_RATE_RAD_S = 1.5
PITCH_RATE_PER_M_S = 0.08
ROLL_RATE_PER_M_S = 0.08
CONTROL_DT_S = 1.0 / 250.0

CLIMB_KEYS = ("q", "r", "page up")
DESCEND_KEYS = ("e", "f", "page down")

CONTROLS_HINT = (
    "Manual controls: [n] toggle manual mode (default OFF) | "
    "[w/a/s/d] north/west/south/east | [q/e or r/f or pgup/pgdn] up/down | "
    "[up/down] forward/back along camera | [left/right] turn camera"
)


class ManualControl:
    """Keyboard flight via attitude rates — same control path as autopilot hover."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self.active = False
        self._toggle_was_down = False
        self._hold_z = None
        self._z_offset = 0.0
        print(CONTROLS_HINT, flush=True)
        if keyboard is None:
            print(
                "Manual control unavailable: install keyboard + run terminal as admin on Windows.",
                flush=True,
            )

    def tick(self):
        """Poll keyboard. Returns True while manual mode owns the drone."""
        if keyboard is None:
            return False
        self._poll_toggle()
        if not self.active:
            return False
        self._apply_movement()
        return True

    def _poll_toggle(self):
        toggle_down = keyboard.is_pressed(MANUAL_TOGGLE_KEY)
        if toggle_down and not self._toggle_was_down:
            self.active = not self.active
            if self.active:
                self._capture_altitude_hold()
            else:
                self._hold_z = None
                self._z_offset = 0.0
            state = "ON" if self.active else "OFF"
            hold = (
                f", hold z={self._hold_z:.1f}m"
                if self.active and self._hold_z is not None
                else ""
            )
            print(f"Manual control: {state}{hold}", flush=True)
        self._toggle_was_down = toggle_down

    def _capture_altitude_hold(self):
        alt = self.data.get("odometry") or self.data.get("local_position_ned")
        self._hold_z = float(alt["z"]) if alt is not None else None
        self._z_offset = 0.0

    def _camera_yaw(self):
        attitude = self.data.get("attitude")
        if attitude is None:
            return 0.0
        return float(attitude["yaw"])

    def _any_pressed(self, keys):
        return any(keyboard.is_pressed(key) for key in keys)

    def _apply_movement(self):
        vx_n = 0.0  # desired NED north m/s
        vy_n = 0.0  # desired NED east m/s
        yaw_rate = 0.0

        if keyboard.is_pressed("w"):
            vx_n += HORIZONTAL_SPEED_M_S
        if keyboard.is_pressed("s"):
            vx_n -= HORIZONTAL_SPEED_M_S
        if keyboard.is_pressed("d"):
            vy_n += HORIZONTAL_SPEED_M_S
        if keyboard.is_pressed("a"):
            vy_n -= HORIZONTAL_SPEED_M_S

        if self._any_pressed(CLIMB_KEYS):
            self._z_offset -= VERTICAL_SPEED_M_S * CONTROL_DT_S
        if self._any_pressed(DESCEND_KEYS):
            self._z_offset += VERTICAL_SPEED_M_S * CONTROL_DT_S

        yaw = self._camera_yaw()
        if keyboard.is_pressed("up"):
            vx_n += HORIZONTAL_SPEED_M_S * math.cos(yaw)
            vy_n += HORIZONTAL_SPEED_M_S * math.sin(yaw)
        if keyboard.is_pressed("down"):
            vx_n -= HORIZONTAL_SPEED_M_S * math.cos(yaw)
            vy_n -= HORIZONTAL_SPEED_M_S * math.sin(yaw)

        if keyboard.is_pressed("right"):
            yaw_rate += YAW_RATE_RAD_S
        if keyboard.is_pressed("left"):
            yaw_rate -= YAW_RATE_RAD_S

        # Body-frame rates from world-frame velocity intent.
        forward = vx_n * math.cos(yaw) + vy_n * math.sin(yaw)
        lateral = -vx_n * math.sin(yaw) + vy_n * math.cos(yaw)
        pitch_rate = -PITCH_RATE_PER_M_S * forward
        roll_rate = ROLL_RATE_PER_M_S * lateral

        z_target = None if self._hold_z is None else self._hold_z + self._z_offset
        thrust = self.controller.pilot._altitude_thrust(HOVER_THRUST, z_target=z_target)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            thrust=thrust,
        )
