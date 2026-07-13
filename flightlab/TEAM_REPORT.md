# Vertical harness — team report

**Branch:** `harness-vertical` (from `origin/main`)  
**Status:** Package complete. Live 3× green iteration **blocked** — sim exits / no MAVLink until human joins TRAINING.

## How to finish live iteration

1. Start sim (`AIGP_3379/FlightSim.exe` or shipping binary).
2. Join a **TRAINING** session (not event/qualification).
3. `uv sync`
4. `uv run python -m flightlab.run_vertical --method pd`
5. One change per run: `pd` → gains → `pid` → `pid_tilt` → `pid_tilt_filt`
6. Stop at **3 consecutive** full-suite PASS. Keep every `runs/vertical_*`.

## Delivered layout

```
flightlab/
  bus.py state.py safety.py metrics.py maneuvers.py protocol.py
  controllers/{leveler,pd,pid,pid_tilt,pid_tilt_filt}.py
  run_vertical.py  README.md
```

- Command rate: **90 Hz**; hover default **0.27**; clamps 0.12–0.60
- MAVLINK20 before pymavlink; GCS heartbeat ~1 Hz; re-arm ~1 s
- Blow-up kill: level+hover 2 s → disarm → reset 31000
- CLI: `--method`, `--only`, `--list`; `make vertical`
- Logs: `runs/vertical_<utc>/log.jsonl` + `report.md`

## Numbers (fill after live greens)

| Item | Value |
|------|-------|
| True hover thrust | *(V1 thrust_mean)* |
| Final method + gains | *(TBD)* |
| Max climb / descend (m/s) | *(V3)* |
| V2 overshoot / settle | *(V2)* |
| V1 jitter (alt/vz/thrust std, r/p p2p) | *(V1)* |
| Blow-ups | *(log paths)* |

## Probe note (this session)

- `udpin:0.0.0.0:14550` with no TRAINING → no vehicle heartbeat / empty msg counts.
- Bus now fails fast if `wait_heartbeat` returns None / `target_system==0`.

## Do not commit yet

Awaiting human review per Spec A.
