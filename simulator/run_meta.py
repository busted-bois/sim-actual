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
import hashlib
import json
import os
import subprocess
import sys
import time

from simulator.run_id import RUN_ID

_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "videos")
_PATH = os.path.join(_DIR, f"vision_{RUN_ID}.json")

_state = {
    "schema": 2,
    "run_id": RUN_ID,
    # Filled at finalize; declared here so the JSON reads top-down.
    "target": None,
    "entry": None,
    "pilot": None,
    "git": None,
    "config": None,
    "track": None,
    "log_columns": None,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ended_utc": None,
    "video": None,
    "attempts": [],
}
_finalized = False

# Every tuning knob on this branch is an env var (docs/blueline-method.md
# "Switches"), and none of them were recorded -- so two runs at the same sha
# were indistinguishable and A/B was impossible. Captured by PREFIX, not by a
# hand-written list: there are already 26+ names and a list drifts out of date
# the same way the CSV header did. Only vars actually set are captured, which
# is right -- the defaults live in code, and sha+diff_sha pin that.
_CONFIG_PREFIXES = ("GP_", "BL_", "GATE", "AUTO_", "SIM_RESET", "RACE_", "SKIP_")


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
    nothing is added to a flight process's startup.

    `dirty` alone is useless for A/B: every recorded run so far flew dirty, so
    the sha identified nothing. The three fields below make a dirty tree
    identifiable without storing the diff itself:

      diff_sha   hash of the tracked-file diff.
      untracked  a brand-new module is invisible to `git diff HEAD`, and a new
                 module is exactly the thing you would be A/B-ing.
      submodule  vendor/AndurilGP is a submodule; its dirt does not appear in
                 the parent's diff at all. Without this, two runs against
                 different vendor trees hash identically -- silently poisoning
                 the exact comparison these fields exist to protect.
    """
    root = os.path.dirname(os.path.dirname(__file__))

    def run(*args):
        return subprocess.run(
            args, cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()

    porcelain = run("git", "status", "--porcelain")
    diff = run("git", "diff", "HEAD")
    return {
        "sha": run("git", "rev-parse", "--short", "HEAD") or None,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(porcelain),
        "diff_sha": hashlib.sha1(diff.encode("utf-8", "replace")).hexdigest()[:12]
        if diff
        else None,
        "untracked": sorted(
            ln[3:] for ln in porcelain.splitlines() if ln.startswith("?? ")
        )
        or None,
        "submodule": run("git", "rev-parse", "HEAD:vendor/AndurilGP")[:12] or None,
    }


def _config():
    """Env knobs actually set for this run, plus the raw argv."""
    return {
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(_CONFIG_PREFIXES)
        },
        "argv": list(sys.argv),
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
def note_track(gates):
    """Which course was flown. `gates` is data["track_gates"].

    Two runs on different courses are not comparable, and nothing otherwise
    stops an aggregate pass averaging across them. Gate 0 pins the layout
    without storing the whole map."""
    if not gates:
        return
    pos = (gates[0] or {}).get("position_ned")
    _state["track"] = {
        "n_gates": len(gates),
        "gate0_ned": [round(float(v), 2) for v in tuple(pos)[:3]]
        if pos is not None and len(tuple(pos)) >= 3
        else None,
    }
    _write()


@_guard
def note_log_columns(columns):
    """The flight-log header. Recorded so an aggregate pass can reject a
    legacy schema without opening a single CSV -- the header IS the version,
    which also catches column REORDERING that a version int would not."""
    _state["log_columns"] = list(columns)
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
    # Config first: it cannot fail, while _git() shells out. Guarded
    # separately because @_guard wraps this whole function -- a git that is
    # missing, slow, or unreachable used to discard every other field with it.
    _state["config"] = _config()
    try:
        _state["git"] = _git()
    except Exception:  # noqa: BLE001 - see module docstring
        _state["git"] = None
    _write()


# A run killed before close() still gets its ending stamped.
atexit.register(finalize)
