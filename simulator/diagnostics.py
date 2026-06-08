"""Telemetry probe — read-only observer of shared_data for Phase 1 unknowns.

Does NOT touch flight logic. Runs as a daemon thread, started from main.py.
Answers: (Q1) does active_gate_index increment on pass, (Q2) gate orientation
quaternions, (Q3) odometry drift vs known gate positions over a lap.

Writes to stdout ([diag] lines) and to diagnostics.log in cwd (overwritten per run).
"""

from __future__ import annotations

import math
import threading
import time

_LOG_PATH = "diagnostics.log"
_SAMPLE_HZ = 2.0


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


class TelemetryProbe:
    def __init__(self, data: dict) -> None:  # type: ignore[type-arg]
        self.data = data
        self.is_running = True
        self._gates_dumped = False
        self._last_index: int | None = None
        self._fh = open(_LOG_PATH, "w", buffering=1)  # line-buffered
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _emit(self, line: str) -> None:
        stamp = f"{time.monotonic():9.2f}"
        msg = f"[diag {stamp}] {line}"
        print(msg, flush=True)
        self._fh.write(msg + "\n")

    def _dump_gates_once(self) -> None:
        gates = self.data.get("track_gates")
        if not gates or self._gates_dumped:
            return
        self._gates_dumped = True
        self._emit(f"GATE TABLE — {len(gates)} gates")
        for i, g in enumerate(gates):
            pos = g.get("position_ned")
            orient = g.get("orientation_ned")
            w = g.get("width")
            h = g.get("height")
            self._emit(
                f"  gate[{i}] pos={pos} orient_wxyz={orient} w={w} h={h}"
            )

    def _check_index_transition(self) -> None:
        idx = self.data.get("active_gate_index")
        if idx is None or idx == self._last_index:
            return
        pos = self._pos()
        gate_pos = self._active_gate_pos(idx)
        d = _dist(pos, gate_pos) if (pos and gate_pos) else None
        self._emit(
            f"ACTIVE_GATE_INDEX {self._last_index} -> {idx}  "
            f"odom_pos={_fmt(pos)} dist_to_new_gate={_fmt_num(d)}"
        )
        self._last_index = idx

    def _pos(self) -> tuple[float, float, float] | None:
        odo = self.data.get("odometry")
        if odo is not None:
            return (odo["x"], odo["y"], odo["z"])
        return None

    def _active_gate_pos(self, idx: int) -> tuple[float, float, float] | None:
        gates = self.data.get("track_gates")
        if gates and 0 <= idx < len(gates):
            return tuple(gates[idx]["position_ned"])  # type: ignore[return-value]
        return None

    def _loop(self) -> None:
        period = 1.0 / _SAMPLE_HZ
        while self.is_running:
            try:
                self._dump_gates_once()
                self._check_index_transition()

                idx = self.data.get("active_gate_index")
                pos = self._pos()
                odo = self.data.get("odometry")
                if pos is not None and idx is not None:
                    gpos = self._active_gate_pos(idx)
                    d = _dist(pos, gpos) if gpos else None
                    vel = (odo.get("vx"), odo.get("vy"), odo.get("vz")) if odo else None
                    yaw = self.data.get("yaw_rad")
                    armed = self.data.get("armed", False)
                    self._emit(
                        f"armed={int(armed)} idx={idx} pos={_fmt(pos)} vel={_fmt(vel)} "
                        f"yaw={_fmt_num(yaw)} dist_active={_fmt_num(d)}"
                    )
            except Exception as e:  # never let the probe crash the run
                self._emit(f"probe error: {e}")
            time.sleep(period)

    def stop(self) -> None:
        self.is_running = False
        try:
            self._fh.close()
        except Exception:
            pass


def _fmt(t) -> str:  # type: ignore[no-untyped-def]
    if t is None:
        return "None"
    return "(" + ", ".join(_fmt_num(v) for v in t) + ")"


def _fmt_num(v) -> str:  # type: ignore[no-untyped-def]
    if v is None:
        return "None"
    return f"{v:+.2f}"
