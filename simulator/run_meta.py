"""Sidecar describing what a run actually did.

A published recording used to travel alone, so telling a gate-4 run from a
gate-1 crash meant pulling 65 MB and watching it. Nothing recorded the outcome
either: race_monitor only computes predicates, lap_log writes on success only,
and the OUTCOME= lines go to stdout and vanish. This writes one small JSON next
to the mp4 -- runs/videos/vision_<RUN_ID>.json -- so the archive can be read
without downloading any video.

The whole dict is rewritten on every event, via a temp file and os.replace, so
a run killed with Ctrl+C still leaves a parseable sidecar. It is ~1 KB and
events arrive about once a minute, so rewriting costs nothing.

NOTHING HERE MAY RAISE. This runs inside the flight process, on the same
"telemetry must never ground the pilot" contract as gp_pilot._open_log: every
public function swallows everything. A broken sidecar is worth less than a
lost run.
"""

import atexit
import functools
import json
import os
import subprocess
import sys
import time

from simulator.run_id import RUN_ID

_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "videos")
_PATH = os.path.join(_DIR, f"vision_{RUN_ID}.json")

_state = {
    "schema": 1,
    "run_id": RUN_ID,
    # Filled at finalize; declared here so the JSON reads top-down.
    "target": None,
    "entry": None,
    "pilot": None,
    "git": None,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ended_utc": None,
    "video": None,
    "attempts": [],
}
_finalized = False


def _guard(fn):
    """Never let sidecar bookkeeping escape into the flight loop."""

    @functools.wraps(fn)
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:  # noqa: BLE001 - deliberate: see module docstring
            return None

    return wrapped


def _describe_process():
    """Best-effort guess at which Makefile target is flying, so a passive
    `make view` session is distinguishable from a real flight. Recorded
    alongside the raw entry point, which is the unambiguous truth."""
    entry = os.path.basename(sys.argv[0] or "") or "?"
    pilot = os.environ.get("AUTO_PILOT", "").strip().lower()
    auto = os.environ.get("AUTO_FLIGHT", "").strip().lower() in ("1", "true", "yes")
    if not pilot:
        pilot = "ibvs" if auto else ""
    target = {
        "auto_gp.py": "control-flight",
        "auto.py": "auto",
        "auto_rl.py": "rl-flight",
        "main.py": "sim",
        "manual.py": "manual",
    }.get(entry)
    if target is None:
        mod = " ".join(sys.argv[:2])
        if "vision_view" in mod:
            target = "view"
        elif "fly2" in mod:
            target = "fly-vision" if "--mode vision" in " ".join(sys.argv) else "fly"
    return entry, pilot or None, target


def _git():
    """Which code flew this. One subprocess at finalize -- never at import, so
    nothing is added to a flight process's startup."""
    root = os.path.dirname(os.path.dirname(__file__))

    def run(*args):
        return subprocess.run(
            args, cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()

    return {
        "sha": run("git", "rev-parse", "--short", "HEAD") or None,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _totals():
    gates = [a.get("gates_passed") for a in _state["attempts"] if a.get("gates_passed") is not None]
    laps = [a.get("lap_s") for a in _state["attempts"] if a.get("lap_s") is not None]
    return {
        "attempts": len(_state["attempts"]),
        "best_gates": max(gates) if gates else None,
        "best_lap_s": min(laps) if laps else None,
    }


def _write():
    os.makedirs(_DIR, exist_ok=True)
    _state["totals"] = _totals()
    tmp = _PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_state, fh, indent=1, sort_keys=False)
    os.replace(tmp, _PATH)  # atomic: a Ctrl+C never leaves a half-written file


def _current():
    return _state["attempts"][-1] if _state["attempts"] else None


@_guard
def note_attempt_start(n, telemetry_path):
    """A flight log opened -- one attempt begins."""
    _state["attempts"].append(
        {
            "n": n,
            "telemetry": os.path.basename(telemetry_path) if telemetry_path else None,
            "t_start_unix": round(time.time(), 3),
            "t_end_unix": None,
            "gates_passed": None,
            "outcome": None,
            "lap_s": None,
            "end_reason": None,
        }
    )
    _write()


@_guard
def note_attempt_end(gates=None, reason=None):
    """That log closed. `gates` is the pilot's running gate count."""
    cur = _current()
    if cur is None:
        return
    cur["t_end_unix"] = round(time.time(), 3)
    if gates is not None:
        cur["gates_passed"] = gates
    if reason is not None:
        cur["end_reason"] = reason
    _write()


@_guard
def note_outcome(outcome, attempt=None, lap_s=None, active=None):
    """The retry loop judged an attempt. Note this fires on the AUTO_FLIGHT
    path, which flies IBVS and writes no CSV -- so there may be no attempt to
    attach to, and we start one."""
    cur = _current()
    if cur is None or (attempt is not None and cur.get("n") not in (None, attempt)):
        _state["attempts"].append(
            {"n": attempt, "telemetry": None, "t_start_unix": None, "t_end_unix": None,
             "gates_passed": None, "outcome": None, "lap_s": None, "end_reason": None}
        )
        cur = _current()
    cur["outcome"] = outcome
    if lap_s is not None:
        cur["lap_s"] = round(float(lap_s), 2)
    if active is not None and cur.get("gates_passed") is None:
        cur["gates_passed"] = active
    _write()


@_guard
def note_video(stats):
    """What display recorded: real frame rate, size, and the epoch origin of
    the burned-in t= overlay. `stats` is simulator.display.stats()."""
    if not stats:
        return
    path = stats.pop("path", None)
    if path and os.path.exists(path):
        stats["bytes"] = os.path.getsize(path)
    _state["video"] = stats
    _write()


@_guard
def finalize():
    """Stamp the end and record which commit flew. Idempotent -- called from
    display.close() and again via atexit."""
    global _finalized
    if _finalized:
        return
    # Nothing was recorded and no attempt was flown -- importing this module
    # must not litter runs/videos/ with sidecars for processes that never flew.
    if not _state["attempts"] and not _state["video"]:
        return
    _finalized = True
    _state["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry, pilot, target = _describe_process()
    _state["entry"], _state["pilot"], _state["target"] = entry, pilot, target
    _state["git"] = _git()
    _write()


# A run killed before close() still gets its ending stamped.
atexit.register(finalize)
