# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

> AGENTS.md (imported above) holds the working agreement: be terse, never commit unprompted, never self-attribute in git, use `uv`/`make`, plans are multi-phase + concise, and all new sim functionality must work from a bare `make sim` with NO new CLI args.

## What this is

Autonomous drone-racing pilot for the AI Grand Prix. Connects over MAVLink (pymavlink) + a UDP camera stream to an external `FlightSim.exe` (gitignored, Windows-only, not in repo). No GPS / absolute coordinates at race time — the pilot must fly on vision + relative odometry.

## Commands

```bash
make           # uv sync (install deps)
make check     # ruff check --fix . && ruff format .  (run before finishing changes)
make sim       # uv run main.py — connects to the running simulator and flies
```

- Run anything Python via `uv run ...`, never bare `python`/`python3`.
- No test suite exists; `make check` (ruff lint+format) is the only gate. CI runs ruff (`.github/workflows/ruff.yml`).
- On Windows, `make` needs `choco install make`; uses PowerShell.
- `make sim` only works while the simulator app is running and listening — it blocks on `wait_heartbeat()` otherwise.

## Architecture

`main.py` builds the components (`simulator/setup.py`), arms, then spins `controller.update()` in a tight loop forever. Everything is glued together by one mutable `shared_data` dict — RX threads write it, the pilot reads it. No locks; single-writer-per-key by convention.

**Data in (background threads, write `shared_data`):**
- `mavlink_rx.py` — parses MAVLink: `ODOMETRY`/`LOCAL_POSITION_NED` → `odometry`/`pos_ned`/`vel_ned`/`yaw_rad`; `HEARTBEAT` → `armed`; `COLLISION` → `collision`; `ENCAPSULATED_DATA`/`DATA_TRANSMISSION_HANDSHAKE` → reassembled multi-chunk track gate list → `track_gates` (gate positions in NED) + `race_status`.
- `vision_rx.py` — reassembles chunked JPEG frames over UDP:5600, decodes, runs `gate_detector.detect_gate`, writes `gate_target` (`detected`/`nx`/`ny`/`r_frac`, normalized image coords) + `obstacles` + `camera.received_at`.
- `timesync.py` — periodic MAVLink TIMESYNC pings.

**Decision (`pilot.py`):** `Pilot.tick()` runs every control cycle (~250 Hz) with a priority cascade: collision-hold → post-gate hover → **live vision** (`_fly_toward_gate_vision`) → search sweep → **telemetry fallback** (`_fly_toward_gate_telemetry`, nearest gate by NED distance) → hover. Tracks flown-through gates in `_completed_gates`/`_passed_gate_positions` to avoid re-targeting them. Altitude is a PID on odometry Z (`_altitude_thrust`).

**Actuation (`controller.py`):** the pilot calls `set_control_mode(...)` + `set_attitude_rates(...)`; `Controller.update()` sends the MAVLink command for the current mode (`attitude` = `SET_ATTITUDE_TARGET` rates+thrust, the mode the pilot uses) at `CONTROL_HZ = 250`.

### Control-channel reality (important — learned, not obvious from code)
Only **attitude-rate + thrust** commands actually move the drone. Velocity setpoints (`set_velocity_ned` / `position` mode) are ignored by the sim. Reported velocity frame is unreliable, and a "fly-away" corrupts the gate frame — recover with a full sim restart, not in-code fixups. Keep commanded rates small.

### Coordinate notes
NED frame (Z is down, so altitude targets are negative, e.g. `Z_TARGET_NED = -5.0`). Vision `nx`/`ny` are image-normalized to `[-1, 1]` from center; `r_frac` is gate-area / frame-area, used as the proximity proxy for fly-through detection. `transforms.py` has the quaternion/yaw/bearing/body↔NED helpers.

## RL racing pipeline (`rl/`)

8-module learned pipeline, separate from the live `simulator/` deploy path. Shared contract in `rl/spec.py` (intrinsics `fx=fy=320,cx=320,cy=180` → 640×360, 20° up cam tilt, 1.5 m gate, frames, projection, **24-D `OBS_LAYOUT`**, action scaling). **Action space = attitude-rate+thrust** (4-D normalized) — chosen because this sim ignores velocity setpoints (see control-channel note above); the policy can therefore actuate live.

Modules: `sim_interface.py` (1: live telemetry+camera+gate-map) → `dataset.py` (2: frames + PnP-projected masks, auto-labeled, no manual labels) → `gatenet.py` (3: U-Net → `data/gatenet.pt`) → `pnp.py` (4: mask corners → solvePnP pose) → `ekf.py` (5: error-state EKF, IMU predict + vision/odom update) → `observation.py` (6: 24-D gate-relative vector) → `env.py` (7: Gymnasium env over an **internal** quad physics model + 3-stage curriculum — does NOT touch the live sim) → `train_ppo.py` (8: SB3 PPO 3×64 MLP → `data/policy.pt`) → `deploy.py` (live runner). `control.py` is a geometric expert (env validation + deploy fallback).

Every module has an offline `--selftest` (run all: `make rl-test`) that validates without the live sim. Makefile: `make capture|dataset|train-gatenet|train-ppo|fly-policy|rl-test`. Live sim is used ONLY for Modules 1–2 (data) and final eval; RL trains against the internal model. `make sim` still runs the original HSV pilot (untouched); the RL policy flies via `make fly-policy`.

### Orphaned modules
`search.py`, `gate_estimator.py`, `state_machine.py` are legacy scaffolding — not wired into the live `setup → controller → pilot` path (only `state_machine` imports `gate_estimator`, and nothing imports `state_machine`). Don't assume they run; prefer extending `pilot.py`.
