"""Focused Tk control window for manual flight.

Owns the keyboard so the simulator window can't steal it (that is why SPACE was
resetting the sim and WASD did nothing). While THIS window is focused, key events
are delivered to us and the sim receives nothing. A live HUD shows input, the
commands we send, and the telemetry response so you can see whether the drone is
actually reacting.

Run via `make manual` (manual.py). Keep this window focused; watch the sim window
beside it.
"""

from __future__ import annotations

import threading
import time

from simulator.controller import CONTROL_HZ
from simulator.manual_control import _ALL_KEYS, ManualControl

_LEGEND = (
    "W/S fwd/back    A/D left/right    Q/E turn\n"
    "SPACE climb    X descend    R/F speed +/-    - / = hover trim\n"
    "Keep THIS window focused.   Esc or close to quit."
)


def _norm(keysym: str) -> str:
    """tkinter keysym -> our key name ('w', 'minus', 'equal', ...)."""
    return keysym.lower() if len(keysym) == 1 else keysym


def _fmt(v, unit="", places=1):
    if isinstance(v, (int, float)):
        return f"{v:.{places}f}{unit}"
    return "n/a"


def run_manual_ui(controller, data):
    """Blocking: open the control window and fly until it is closed."""
    import tkinter as tk

    held = {k: False for k in _ALL_KEYS}
    stop = threading.Event()
    pilot = ManualControl(controller, data, is_pressed=lambda k: held.get(k, False))

    def control_loop():
        dt = 1.0 / CONTROL_HZ
        print(f"[manual] control loop running at {CONTROL_HZ} Hz", flush=True)
        while not stop.is_set():
            try:
                pilot.tick()
            except Exception as exc:  # keep flying loop alive; surface once
                print(f"[manual] control error: {exc}", flush=True)
            time.sleep(dt)

    root = tk.Tk()
    root.title("Drone Manual Control")
    root.configure(bg="#101418")
    root.geometry("460x340")
    root.attributes("-topmost", True)

    def on_press(e):
        n = _norm(e.keysym)
        if n in held:
            held[n] = True

    def on_release(e):
        n = _norm(e.keysym)
        if n in held:
            held[n] = False

    def quit_ui(*_):
        stop.set()
        root.after(60, root.destroy)

    root.bind("<KeyPress>", on_press)
    root.bind("<KeyRelease>", on_release)
    root.bind("<Escape>", quit_ui)
    root.protocol("WM_DELETE_WINDOW", quit_ui)

    tk.Label(
        root,
        text=_LEGEND,
        fg="#7fd88f",
        bg="#101418",
        justify="left",
        font=("Consolas", 10),
    ).pack(anchor="w", padx=12, pady=(12, 8))
    hud = tk.Label(
        root,
        text="",
        fg="#e6e6e6",
        bg="#101418",
        justify="left",
        font=("Consolas", 11),
    )
    hud.pack(anchor="w", padx=12)

    warned_blocked = [False]

    def refresh():
        if stop.is_set():
            return
        s = pilot.status()
        cmd = s["cmd"]
        held_str = " ".join(k.upper() for k in _ALL_KEYS if held.get(k)) or "-"
        if s["have_telemetry"]:
            tel = "yes"
        elif s["pose_blocked"]:
            tel = "IMU only (pose BLOCKED)"
            if not warned_blocked[0]:
                warned_blocked[0] = True
                print(
                    "[manual] sim streams IMU but no pose telemetry — this is an "
                    "event/qualification session, which blocks ODOMETRY/ATTITUDE. "
                    "Start a TRAINING session for manual flight.",
                    flush=True,
                )
        else:
            tel = "NO  <-- no data from sim!"
        hint = (
            "\n!! event session blocks pose — start a TRAINING session"
            if s["pose_blocked"]
            else ""
        )
        armed = {True: "yes", False: "no", None: "?"}[s["armed"]]
        hud.config(
            text=(
                f"input held : {held_str}\n"
                f"armed      : {armed}     telemetry : {tel}\n"
                f"speed set  : {s['speed_kmh']:.1f} km/h     hover trim : {s['hover_trim']:+.2f}\n"
                f"altitude   : {_fmt(s['alt_m'], ' m')}     vspeed : {_fmt(s['vz_mps'], ' m/s')}\n"
                f"hor. speed : {_fmt(s['hspeed_kmh'], ' km/h')}\n"
                f"attitude   : roll {_fmt(s['roll_deg'], '', 0)}  "
                f"pitch {_fmt(s['pitch_deg'], '', 0)}  yaw {_fmt(s['yaw_deg'], '', 0)}\n"
                f"cmd sent   : roll {cmd['roll']:+.2f}  pitch {cmd['pitch']:+.2f}  "
                f"yaw {cmd['yaw']:+.2f}  thr {cmd['thrust']:.2f}"
                f"{hint}"
            )
        )
        root.after(60, refresh)

    threading.Thread(target=control_loop, daemon=True).start()
    print("[manual] control window open — click it to focus, then fly.", flush=True)
    refresh()
    root.mainloop()
    stop.set()
