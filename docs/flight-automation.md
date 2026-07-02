# Flight automation

Continuous overnight retry loop via `make auto`. Normal `make sim` is unchanged.

## Commands

| Command | Behavior |
|---------|----------|
| `make auto` | VQ2 R2 overnight automation — main `fly2` course pilot, continuous in-session retry |
| `make sim` | Standard sim — vision pilot, single run, no auto-retry |
| `make capture-gates` | One-time: save `rl/data/gate_map.json` from a VQ2 race start (required for `make auto`) |

## Flight controller

`make auto` uses the **main-branch `fly2` course controller** (`rl/fly2_course.py`) for **AI-GP VIRTUAL QUALIFIER R2** (sim v1.0.3379+):

- **TRAINING** — recommended for overnight practice runs
- **SUBMISSION** — same map, same client steps

VQ2 blocks sim odometry and live gate-map poses. Automation uses:

- Captured **`rl/data/gate_map.json`** (from `make capture-gates`)
- **ESKF pose** — IMU predict + vision bearing/range fusion (`simulator/vq2_pose.py`)
- FPV vision (UDP :5600) for gate detection and pose updates
- `HIGHRES_IMU` (`pressure_alt` for altitude)
- `race_status` (`active_gate_index`, GO timing)

`make sim` still uses the vision `Pilot`; only `make auto` swaps to fly2.

## Overnight workflow

1. Launch **FlightSim v1.0.3379** and enter **AI-GP VIRTUAL QUALIFIER R2 — TRAINING** (or SUBMISSION).
2. Run **`make capture-gates`** once if `rl/data/gate_map.json` is missing — click **Race** while it listens.
3. Run **`make auto`** — should reach `[AUTO] overnight automation on` within seconds (no 30s odometry wait).
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
| `[AUTO] overnight automation on — main fly2 course pilot` | Continuous auto-flight active |
| `[fly2] main course pilot ready` | Fly2CoursePilot armed |
| `[fly2] ACTIVE GATE -> N` | Sim advanced target gate |
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
