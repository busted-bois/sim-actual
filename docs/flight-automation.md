# Flight automation

Continuous overnight retry loop via `make auto`. Normal `make sim` is unchanged.

## Commands

| Command | Behavior |
|---------|----------|
| `make auto` | VQ2 R2 overnight automation — vision navigator (YOLO+PnP), continuous in-session retry |
| `make sim` | Standard sim — vision pilot, single run, no auto-retry |
| `make fly-vision` | Manual single run of the same vision navigator via `rl.fly2 --mode vision` |
| `make capture-gates` | Optional: save `rl/data/gate_map.json` (only used to seed the EKF pose fallback) |

## Flight controller

`make auto` uses the **vision navigator** (`simulator/vision_nav_pilot.py` wrapping
`simulator/vision_nav.py`) for **AI-GP VIRTUAL QUALIFIER R2** (sim v1.0.3379+):

- **TRAINING** — recommended for overnight practice runs
- **SUBMISSION** — same map, same client steps

The navigator flies purely from detections — no gate map, no hardcoded coordinates:

- **YOLO11n-pose gate detector** (`simulator/gate_pose.py`, weights `simulator/models/gate_pose.pt`) on its own thread
- **PnP gate pose** (`simulator/gate_pnp.py`) — body-frame gate position per detection
- **Persistent gate map** built in-flight (`VisionGuidance`): confirmed gates (3+ hits, deduped), closing-speed-regulated approach (max 1.5 m/s), PD lateral control, yaw-scan when no gate in view
- **Pose**: sim odometry when the session provides it, else the ESKF estimate (`simulator/vq2_pose.py`); the active source is printed as `[vnav] pose source: ...`
- `race_status` (`active_gate_index`, GO timing) for retry outcomes

`make sim` still uses the vision `Pilot`; only `make auto` swaps to the navigator.

## Overnight workflow

1. Launch **FlightSim v1.0.3379** and enter **AI-GP VIRTUAL QUALIFIER R2 — TRAINING** (or SUBMISSION).
2. Run **`make auto`** — should reach `[AUTO] overnight automation on` within seconds (no 30s odometry wait).
4. When prompted on attempt 1, click **Race** in FlightSim.
5. Automation flies, retries on gate timeout or course complete, and loops until you stop it.
6. After each run the client sends MAVLink reset and waits for a fresh countdown in the same session.

## Cancel automation

Press **Ctrl+C** in the terminal running `make auto`:

→ first Ctrl+C cancels automation and drops into normal sim control (250 Hz loop, no retry).
→ **press Ctrl+C again to exit** the process entirely.

You should see `[AUTO] cancel requested (Ctrl+C) — press Ctrl+C again to exit` then `[AUTO] cancelled — resuming normal sim mode`.

## Stdout markers

| Marker | Meaning |
|--------|---------|
| `VQ2 mode — skipping odometry wait` | Startup using IMU/vision, not odometry |
| `Connect OK: imu=... vision=...` | MAVLink sensors ready |
| `[AUTO] overnight automation on — vision navigator (YOLO+PnP)` | Continuous auto-flight active |
| `[vnav] vision navigator pilot ready` | VisionNavPilot armed |
| `[gate_pose] YOLO loaded on cpu/cuda` | Gate detector model up |
| `[vnav] pose source: odometry` / `EKF` | Which pose estimate is flying |
| `[vnav] GO d=… passed=N map=M` | Navigator chasing a confirmed gate |
| `[vnav] SCAN n=… map=…` | No confirmed gate ahead — yaw-scanning |
| `[vnav] PASSED gN` | Navigator counted a gate pass |
| `[AUTO] cancel requested (Ctrl+C)` | User cancelled |
| `[AUTO] cancelled — resuming normal sim mode` | Automation stopped; normal sim continues |
| `Preflight OK: vision streaming` | Inside SUBMISSION/TRAINING session |
| `Click Race in FlightSim...` | Attempt 1 — waiting for you to start race |
| `[RACE] waiting for fresh race_start after reset...` | Attempt 2+ — waiting for sim restart countdown |
| `Race go!` | Countdown finished |
| `[vision] GATE acquired` / `[vision] GATE lost` | Gate edge logs (`make auto` only) |
| `[RACE] GATE1=pass active=1` | First gate cleared |
| `[RACE] GATE_ADVANCE active=N` | Sim advanced to next gate |
| `[RACE] gate_progress_watch ...` | Post-gate1 progress (every 5s) |
| `[RACE] OUTCOME=gate1_fail attempt=N retrying` | Gate 1 missed; sim reset + retry |
| `[RACE] OUTCOME=gate_stall attempt=N active=X retrying` | No gate advance in 15s; sim reset + retry |
| `[RACE] OUTCOME=success attempt=N lap=Xs best=Ys — restarting` | Lap complete; starting next attempt |
| `[RACE] reset sent — click Restart Race...` | Fallback if sim doesn't auto-restart countdown |

## Lap time files

On each course complete (`OUTCOME=success`), lap times are appended to:

| File | Contents |
|------|----------|
| `rl/data/auto_laps.jsonl` | One JSON line per lap: `ts`, `attempt`, `lap_s`, `best_lap_s` |
| `rl/data/best_lap.txt` | Latest best lap (single line, quick read) |

Stdout still prints `[RACE] OUTCOME=success attempt=N lap=Xs best=Ys`.

## Fail / success logic

- **Gate 1 fail:** `active_gate_index` stays `< 1` for `GATE1_TIMEOUT_S` (default 15s) with `pilot.gates_passed == 0`.
- **Gate stall:** after gate 1, no `active_gate_index` advance for `GATE_PROGRESS_TIMEOUT_S` (default 15s).
- **Success:** `race_finish_time_ns >= 0` or `active_gate_index >= gate_count` — logs lap time and **restarts** (does not exit).
- **Best lap:** tracked in stdout and `rl/data/best_lap.txt` across attempts.

## Env tunables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTO_FLIGHT` | unset | Set to `1` by `make auto` |
| `AUTO_FLIGHT_DEBUG` | unset | Vision miss logs |
| `GATE1_TIMEOUT_S` | `15` (`make auto` sets `30`) | Max seconds after GO to pass gate 1 |
| `GATE1_HARD_TIMEOUT_S` | `60` | Hard gate-1 retry cap even if the pilot counted a pass the sim didn't |
| `GATE_PROGRESS_TIMEOUT_S` | `15` (`make auto` sets `25`) | Max seconds without gate advance after gate 1 |
| `GATE1_WATCH_INTERVAL_S` | `5` | Seconds between progress watch logs |
| `SIM_RESET_WAIT_S` | `5` | Pause after MAVLink sim reset |

## Troubleshooting

### `ERROR: rl/data/gate_map.json missing or empty`

Run `make capture-gates` in VQ2 TRAINING, click **Race** while it listens, then retry `make auto`.

### `ERROR: UDP 14550 already in use by another process`

A stale `make auto`/`make sim` client still holds the MAVLink port. Free it and retry:

```
make free-port
```

### Stuck on `waiting for vision`

Enter **SUBMISSION** or **TRAINING** flight session (not the environment picker menu).

### Stuck after `Click Race...` (attempt 1)

Click **Race** in FlightSim. Wait for countdown and `Race go!`.

### Drone hovers but never moves forward

Confirm `Race go!` printed and `[vision] GATE` lines appear. Without vision updates the EKF cannot correct xy drift.

### No `[vision] GATE` lines

VQ2 R2 gates may be desaturated — detection uses widened HSV + low-sat fallback. Train GateNet (`make train-gatenet`) if needed.

## Notes

- Sim gates are 0-based; "gate 1" = `active_gate_index >= 1`.
- MAVLink reset (31000) resets drone pose; sim usually auto-restarts countdown in the same TRAINING session.
- No menu navigation — stay in TRAINING for the whole overnight run.
- VQ1 sessions are no longer available; use VQ2 R2 only.
