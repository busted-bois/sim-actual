# AI-GP Round 1 

[![demo](./demo.gif)]

| config | gates | time | collisions |
|---|---:|---:|---:|
| `measured` | `6/6` | `24.xx s` | `0` |

## Run

`fly.py` listens for simulator telemetry, arms the drone, waits for a fresh race
start, then sends rate/thrust commands through MAVLink.

The simulator runs on Windows. The pilot can run on another machine if a UDP
relay forwards the simulator's localhost MAVLink stream:

```cmd
python relay.py --target-ip <PILOT_IP>
```

Then start the pilot:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 fly.py
```

Start the simulator run after the pilot is waiting.

## Files

- `fly.py`
- `relay.py`
- `src/aigp_pilot/raceconfig.py`
- `src/aigp_pilot/course.py`
- `src/aigp_pilot/raceline.py`
- `src/aigp_pilot/control.py`
- `src/aigp_pilot/parsers.py`

## Test

```bash
python3 -m pytest -q
```
