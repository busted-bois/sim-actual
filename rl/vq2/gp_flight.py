"""Fly the RL policy through the PROVEN auto_gp.py flight stack.

`auto_gp.py` (make control-flight) flies via `setup_components` -> a `Controller`
whose GP pilot handles the whole lifecycle (arm, WAIT_FOR_START -> GO, re-arm on
sim reset) and streams commands on the attitude-quaternion wire at CONTROL_HZ.

For closed-loop RL the POLICY must steer, not the GP guidance. The pilot always
emits its command via `controller.set_attitude_quat_deg(...)`, so we intercept
that ONE call: once the env has pushed an action, we substitute the policy's
command for the pilot's. The pilot's arming/robust control loop keeps running;
only the steering is replaced.

Drop-in for the env's use of `SimInterface`: exposes `.data`,
`send_attitude_quat_deg`, `arm`, `reset_sim`, `close`. A background thread runs
the auto_gp control loop (`controller.update()`), so the command is streamed to
the sim continuously; the env just sets the target each decision tick.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from pymavlink import mavutil

# Final clamp on (base + residual) roll/pitch command sent to the sim, so a
# residual can never drive the attitude command into a flip.
MAX_CMD_DEG = 25.0


class GPFlightInterface:
    def __init__(self, ip: str = "127.0.0.1", mav_port: int = 14550):
        os.environ["AUTO_PILOT"] = "gp"             # FORCE the GP pilot (proven lifecycle)
        os.environ.setdefault("GP_DISPLAY", "0")
        from simulator.setup import setup_components

        self.data: dict = {"_quiet_vision": True}
        self._boot_ms = int(time.time() * 1000)
        comps = setup_components(self.data, self._boot_ms, ip, mav_port)
        self._comps = comps
        self.controller = comps["controller"]
        self._conn = self.controller.sim_conn

        # --- RESIDUAL override: the policy ADDS a small correction to the GP
        # pilot's base command (not replace). Base = proven stability; residual =
        # the policy's nudge. Zero residual => identical to the pure GP flight.
        self._residual = (0.0, 0.0, 0.0, 0.0)
        self._rl_active = False
        self._orig_set = self.controller.set_attitude_quat_deg

        def _clamp(v, lo, hi):
            return lo if v < lo else hi if v > hi else v

        def _override(roll_deg, pitch_deg, yaw_deg, thrust):
            if self._rl_active:
                dr, dp, dy, dth = self._residual
                roll_deg = _clamp(roll_deg + dr, -MAX_CMD_DEG, MAX_CMD_DEG)
                pitch_deg = _clamp(pitch_deg + dp, -MAX_CMD_DEG, MAX_CMD_DEG)
                yaw_deg = yaw_deg + dy
                thrust = _clamp(thrust + dth, 0.0, 1.0)
            return self._orig_set(roll_deg, pitch_deg, yaw_deg, thrust)

        self.controller.set_attitude_quat_deg = _override

        # --- background flight loop (the auto_gp.py loop) ---
        # ALL MAVLink wire sends happen on THIS thread. arm()/reset_sim() from the
        # env thread enqueue here (pymavlink is not thread-safe -> concurrent sends
        # from two threads corrupt packets). deque append/popleft are atomic in CPython.
        self._cmd_queue = deque()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # controller.update() SELF-PACES via time.sleep(1/control_hz) (=60 Hz for
        # the GP pilot), exactly like auto_gp.py's main loop -- so NO extra sleep
        # here (a second sleep detuned the pilot to ~36 Hz). Sleep only if a tick
        # bailed early, mirroring auto_gp.py.
        import traceback
        last_tb = 0.0
        while self._running:
            # drain env-thread commands (arm / reset teleport) HERE so every wire
            # send is serialized on this one thread.
            while self._cmd_queue:
                try:
                    self._cmd_queue.popleft()()
                except Exception:
                    traceback.print_exc()
            try:
                self.controller.update()
            except Exception:
                now = time.monotonic()
                if now - last_tb >= 5.0:
                    traceback.print_exc()
                    last_tb = now
                time.sleep(1.0 / 90.0)

    # ---- SimInterface-compatible API used by the env / reset harness --------
    def send_residual_deg(self, d_roll, d_pitch, d_yaw, d_thrust):
        """Set the policy's residual correction; the background loop ADDS it to the
        GP base command every tick. Zero residual => pure GP flight."""
        self._residual = (float(d_roll), float(d_pitch), float(d_yaw), float(d_thrust))
        self._rl_active = True

    def arm(self):
        self._cmd_queue.append(self.controller.arm)   # runs on the bg thread

    def reset_sim(self):
        # Teleport to spawn (cmd 31000), sent on the bg thread. The GP pilot re-arms
        # on the sim-clock reset; stop overriding so it holds cleanly until the
        # policy steps again.
        self._rl_active = False
        self._residual = (0.0, 0.0, 0.0, 0.0)

        def _teleport():
            self._conn.mav.command_long_send(
                self._conn.target_system, self._conn.target_component,
                31000, 0, 0, 0, 0, 0, 0, 0, 0,
            )

        self._cmd_queue.append(_teleport)

    def flying(self) -> bool:
        pilot = getattr(self.controller, "pilot", None)
        phase = getattr(pilot, "phase", None)
        return str(getattr(phase, "name", "")) in ("FLYING", "BACKOFF")

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        self.controller.set_attitude_quat_deg = self._orig_set
        for k in ("ts_loop", "mavlink_rx", "vision_rx"):
            try:
                self._comps[k].get_thread_for_join().join(timeout=2.0)
            except Exception:
                pass


# re-export so callers can build MAVLink messages if needed.
__all__ = ["GPFlightInterface", "mavutil"]
