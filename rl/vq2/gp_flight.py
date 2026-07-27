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

from pymavlink import mavutil


class GPFlightInterface:
    def __init__(self, ip: str = "127.0.0.1", mav_port: int = 14550):
        os.environ.setdefault("AUTO_PILOT", "gp")   # GP pilot = the proven lifecycle
        os.environ.setdefault("GP_DISPLAY", "0")
        from simulator.controller import CONTROL_HZ
        from simulator.setup import setup_components

        self.data: dict = {"_quiet_vision": True}
        self._boot_ms = int(time.time() * 1000)
        comps = setup_components(self.data, self._boot_ms, ip, mav_port)
        self._comps = comps
        self.controller = comps["controller"]
        self._conn = self.controller.sim_conn
        self._hz = CONTROL_HZ

        # --- guidance override: replace the pilot's steering with the policy's ---
        self._rl_cmd = (0.0, 0.0, 0.0, 0.0)
        self._rl_active = False
        self._orig_set = self.controller.set_attitude_quat_deg

        def _override(roll_deg, pitch_deg, yaw_deg, thrust):
            if self._rl_active:
                roll_deg, pitch_deg, yaw_deg, thrust = self._rl_cmd
            return self._orig_set(roll_deg, pitch_deg, yaw_deg, thrust)

        self.controller.set_attitude_quat_deg = _override

        # --- background flight loop (the auto_gp.py loop) ---
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import traceback
        last_tb = 0.0
        dt = 1.0 / self._hz
        while self._running:
            try:
                self.controller.update()      # pilot lifecycle + send (overridden steering)
            except Exception:
                now = time.monotonic()
                if now - last_tb >= 5.0:
                    traceback.print_exc()
                    last_tb = now
            time.sleep(dt)

    # ---- SimInterface-compatible API used by the env / reset harness --------
    def send_attitude_quat_deg(self, roll_deg, pitch_deg, yaw_deg, thrust):
        """Set the policy's command; the background loop streams it at CONTROL_HZ.
        Also force attitude_quat mode so the command actually goes on the wire."""
        self._rl_cmd = (float(roll_deg), float(pitch_deg), float(yaw_deg), float(thrust))
        self._rl_active = True
        self.controller.set_control_mode("attitude_quat")

    def arm(self):
        self.controller.arm()

    def reset_sim(self):
        # Teleport to spawn (cmd 31000). The GP pilot re-arms on the sim-clock
        # reset; stop overriding so it holds cleanly until the policy steps again.
        self._rl_active = False
        self._conn.mav.command_long_send(
            self._conn.target_system, self._conn.target_component,
            31000, 0, 0, 0, 0, 0, 0, 0, 0,
        )

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
