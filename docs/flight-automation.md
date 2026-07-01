# Flight automation

Continuous overnight retry loop via `make auto`. Normal `make sim` is unchanged.

## Commands

| Command | Behavior |
|---------|----------|
| `make auto` | VQ2 R2 overnight automation — same `Pilot` as `make sim`, continuous in-session retry |
| `make sim` | Standard sim — vision pilot, single run, no auto-retry |

## Flight controller

`make auto` uses the **vision pilot** for **AI-GP VIRTUAL QUALIFIER R2** (sim v1.0.3379+):

- **TRAINING** — recommended for overnight practice runs
- **SUBMISSION** — same map, same client steps

Competitive VQ2 blocks odometry and gate-map poses. Automation uses:

- FPV vision (UDP :5600)
- `HIGHRES_IMU` (`pressure_alt` for altitude hold)
- `race_status` (`active_gate_index`, GO timing)

After GO the pilot uses the same vision + attitude control as `make sim` (search mode only after passing a gate).

## Overnight workflow

1. Launch **FlightSim v1.0.3379** and enter **AI-GP VIRTUAL QUALIFIER R2 — TRAINING** (or SUBMISSION).
2. Run **`make auto`** — should reach `[AUTO] overnight automation on` within seconds (no 30s odometry wait).
3. When prompted on attempt 1, click **Race** in FlightSim.
4. Automation flies, retries on gate timeout or course complete, and loops until you stop it.
5. After each run the client sends MAVLink reset and waits for a fresh countdown in the same session.

## Cancel automation

Press **Ctrl+C** in the terminal running `make auto`:

→ cancels automation and drops into normal sim control (250 Hz loop, no retry).

You should see `[AUTO] cancel requested (Ctrl+C)` then `[AUTO] cancelled — resuming normal sim mode`.

## Stdout markers

| Marker | Meaning |
|--------|---------|
| `VQ2 mode — skipping odometry wait` | Startup using IMU/vision, not odometry |
| `Connect OK: imu=... vision=...` | MAVLink sensors ready |
| `[AUTO] overnight automation on — ...` | Continuous auto-flight active |
| `[AUTO] cancel requested (Ctrl+C)` | User cancelled |
| `[AUTO] cancelled — resuming normal sim mode` | Automation stopped; normal sim continues |
| `Preflight OK: vision streaming` | Inside SUBMISSION/TRAINING session |
| `Click Race in FlightSim...` | Attempt 1 — waiting for you to start race |
| `[RACE] waiting for fresh race_start after reset...` | Attempt 2+ — waiting for sim restart countdown |
| `Race go!` | Countdown finished |
| `[vision] GATE cx=...` | Gate detected |
| `[RACE] GATE1=pass active=1` | First gate cleared |
| `[RACE] GATE_ADVANCE active=N` | Sim advanced to next gate |
| `[RACE] gate_progress_watch ...` | Post-gate1 progress (every 5s) |
| `[RACE] OUTCOME=gate1_fail attempt=N retrying` | Gate 1 missed; sim reset + retry |
| `[RACE] OUTCOME=gate_stall attempt=N active=X retrying` | No gate advance in 15s; sim reset + retry |
| `[RACE] OUTCOME=success attempt=N lap=Xs best=Ys — restarting` | Lap complete; starting next attempt |
| `[RACE] reset sent — click Restart Race...` | Fallback if sim doesn't auto-restart countdown |

## Fail / success logic

- **Gate 1 fail:** `active_gate_index` stays `< 1` for `GATE1_TIMEOUT_S` (default 15s) with `pilot.gates_passed == 0`.
- **Gate stall:** after gate 1, no `active_gate_index` advance for `GATE_PROGRESS_TIMEOUT_S` (default 15s).
- **Success:** `race_finish_time_ns >= 0` or `active_gate_index >= gate_count` — logs lap time and **restarts** (does not exit).
- **Best lap:** tracked in stdout across attempts (`best=Xs` on each success).

## Env tunables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTO_FLIGHT` | unset | Set to `1` by `make auto` |
| `AUTO_FLIGHT_DEBUG` | unset | Pilot mode + vision miss logs |
| `GATE1_TIMEOUT_S` | `15` | Max seconds after GO to pass gate 1 |
| `GATE_PROGRESS_TIMEOUT_S` | `15` | Max seconds without gate advance after gate 1 |
| `GATE1_WATCH_INTERVAL_S` | `5` | Seconds between progress watch logs |
| `SIM_RESET_WAIT_S` | `5` | Pause after MAVLink sim reset |

## Troubleshooting

### Stuck at `Waiting for telemetry...`

You are on an old build. VQ2 `make auto` should print `VQ2 mode — skipping odometry wait` instead.

### Stuck on `waiting for vision`

Enter **SUBMISSION** or **TRAINING** flight session (not the environment picker menu).

### Stuck after `Click Race...` (attempt 1)

Click **Race** in FlightSim. Wait for countdown and `Race go!`.

### Stuck on `waiting for fresh race_start` (attempt 2+)

The sim may need a manual **Restart Race** click after reset. If the countdown doesn't start within 30s, click it in FlightSim.

### Drone hovers but never moves forward

Confirm `Race go!` printed and a gate is visible in frame. Enable `AUTO_FLIGHT_DEBUG=1` if needed. Train GateNet (`make train-gatenet`) if HSV never detects gates.

### No `[vision] GATE` lines

VQ2 R2 gates may be desaturated — detection uses widened HSV + low-sat fallback. Train GateNet (`make train-gatenet`) if needed.

## Notes

- Sim gates are 0-based; "gate 1" = `active_gate_index >= 1`.
- MAVLink reset (31000) resets drone pose; sim usually auto-restarts countdown in the same TRAINING session.
- No menu navigation — stay in TRAINING for the whole overnight run.
