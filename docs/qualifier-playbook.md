# Qualifier Run Playbook

Step-by-step guide for running the autonomous pilot against FlightSim.exe.

## Prerequisites

- Windows 10/11 with NVIDIA GPU (see [Instructions.md](Instructions.md))
- FlightSim.exe installed (from `AIGP_X.zip`, not in this repo)
- `uv` and `make` installed (`choco install make` on Windows)
- Repo deps installed: `make install`

## Startup Order

1. **Launch FlightSim.exe** and log in with your simulator account.
2. **Start a qualifier / flight session** — drone must be in an active session, not the main menu.
3. **Run the pilot** from this repo:

```bash
make sim
```

Optional collision auto-reset after hold:

```bash
make sim COLLISION_RESET=1
# or: AUTO_RESET_ON_COLLISION=1 make sim
```

## What Happens

| Phase | Pilot behavior | Console signal |
|-------|----------------|----------------|
| Connect | Waits for MAVLink heartbeat (60s timeout) | `Connected to system: …` |
| Track load | Waits for `track_gates` telemetry | `Preflight OK: track_gates loaded` |
| Arm | Arms drone automatically | `Arming drone…` |
| Countdown | Hovers until race GO | `Race go! … branch=countdown` or `scheduled` |
| Race | Pursuit planner + vision/tracking steering | Velocity/attitude commands at 50 Hz |
| Finish | Hovers when `race_finish_time_ns >= 0` | `Race finished! last_gate_race_time=…` |
| Restart | Re-arms after sim reboot, waits for new GO | `Armed for session at sim_boot=…` |

## Race GO Timing

- First run: `race_start_boot_time_ms` is the scheduled GO instant.
- After sim restart: GO = `race_start_boot_time_ms + RACE_COUNTDOWN_MS` (default 3s).
- Override countdown: `RACE_COUNTDOWN_MS=5000 make sim`

Calibrate GO timing with:

```bash
make race-timing-probe
```

## Diagnostics

| Command | When to use |
|---------|-------------|
| `make mavlink-probe` | No heartbeat / no telemetry |
| `make vision-smoke` | Gate detection not working |
| `make tracking-smoke` | IMU tracker drift or health issues |
| `make preflight` | UDP port 5600 blocked |
| `make test` | Verify code before live run |
| `make validate-log` | Analyze tracking CSV after a live run |

## Troubleshooting

### `No MAVLink heartbeat received`

- Confirm FlightSim session is active (not paused at menus).
- Stop other pilots on port 14550: `netstat -ano | findstr 14550`
- Run `make mavlink-probe` while session is active.

### `Vision preflight failed`

- Another process holds UDP 5600.
- Stop other `make sim` instances and retry.

### Drone hovers but never flies

- Race GO not reached — check countdown with `make race-timing-probe`.
- `track_gates` empty — wait for track data or restart session.

### Drone flies but misses gates

- Run `make vision-smoke` to verify gate detection.
- Check `logs/tracking_state_*.csv` for tracking health and drift.
- Tune gains in `simulator/flight_config.py`.

## Tracking CSV

When `LocalTracker` logs, files land in `logs/tracking_state_<timestamp>.csv` (gitignored).

Columns include pose (x/y/z), velocity, attitude, IMU sample count, and vision correction events. Use these to diagnose drift between vision corrections.

## Definition of Done (VQ Round 1)

- [ ] Full lap without manual intervention
- [ ] Race GO handled on first run and after sim restart
- [ ] Collision recovery (hold or auto-reset)
- [ ] Clean hover on race finish
- [ ] `make test` passes
