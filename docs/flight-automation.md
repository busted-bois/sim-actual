# Flight automation

Opt-in retry loop via `make auto`. Normal `make sim` is unchanged.

## Commands

| Command | Behavior |
|---------|----------|
| `make auto` | Automation mode — vision pilot + gate-1 retry until complete |
| `make sim` | Standard sim — vision pilot, single run, no auto-retry |

## Flight controller

`make auto` uses the **same vision-first pilot as `make sim`** (`simulator/pilot.py`): camera gate detection with telemetry fallback. Preflight uses the `auto` camera preview window for countdown ROI; flight uses live vision pixels. Odometry-only fly2 course control is **not** used for `make auto`.

VisionRX still receives camera frames in the background; `[vision] GATE` logs are throttled to once per 2s during `make auto` to keep terminal readable.

## Workflow

1. **Open FlightSim** and enter a flight session (MAVLink must connect — client waits up to 30s for heartbeat).
2. Run **`make auto`** — terminal shows `waiting for fresh track_gates`.
3. Click **Race** in FlightSim (or **Restart Race** if you already raced and missed the burst).
4. Wait for `[RACE] fresh race_start=...` → `[RACE] countdown latched` → `Race go!` → `[RACE] visual GO` → `Arming drone...` → `[RACE] controls enabled`.

The sim sends the gate map **once** per race start. If you clicked Race before the client was ready, use **Restart Race** — do not click Race again on an in-progress countdown.

The client **cannot** click Race for you — there is no MAVLink command for that.

## Cancel automation

While `make auto` is running:

- **Ctrl+C** or **Ctrl+any letter** — press **twice** within 3s to confirm

→ cancels automation and drops into normal sim control (250 Hz loop, no retry). Process keeps running.

First press prints `[AUTO] press Ctrl+C or Ctrl+letter again to exit automation`.

## Agent workflow

1. Run `make auto` once (FlightSim open).
2. Start Race in sim when prompted.
3. Watch terminal stdout for outcome markers.
4. Stop when you see `[RACE] OUTCOME=success` (process exits 0).

## Stdout markers

| Marker | Meaning |
|--------|---------|
| `[AUTO] automation on — ...` | Auto-flight mode active |
| `[AUTO] cancelled — resuming normal sim mode` | User cancelled; normal loop |
| `[RACE] attempt=N waiting for track_gates...` | New attempt starting |
| `[RACE] waiting... odometry=... race_start=... track_gates=...` | Preflight status every 5s while waiting for track burst |
| `[RACE] fresh race_start=... track_sim_boot=...` | race_start accepted for this track burst |
| `[RACE] countdown latched branch=... go_boot=...` | Two-phase GO: witnessed active countdown |
| `[RACE] visual GO (countdown cleared)` | On-screen countdown seen then cleared |
| `[RACE] visual GO fallback ...` | No countdown on camera; mavlink GO + margin used |
| `[FLIGHT_DBG] ...` | Preflight/control debug (`make auto` enables via `AUTO_FLIGHT_DEBUG=1`) |
| `[FLIGHT_DBG] milestone preflight_start/track_ok/...` | Attempt milestones with `armed`, `vel_ned`, `go_boot` |
| `[FLIGHT_DBG] safe_hold sent thrust=0` | First mavlink zero-thrust hold during preflight |
| `[FLIGHT_DBG] safe_tick armed=... vel_ned=...` | Throttled preflight safety loop (every 500ms) |
| `[FLIGHT_DBG] fresh_race_start reject=...` | Why race_start not accepted yet |
| `[FLIGHT_DBG] race_go_p1 reject_at_go / witnessed=...` | Two-phase GO phase-1 state |
| `[FLIGHT_DBG] race_go_p2 remaining_ms=...` | Countdown progress toward GO |
| `[FLIGHT_DBG] visual_go state idle->saw_countdown` | Vision countdown gate transitions |
| `[FLIGHT_DBG] race_start_chg old->new delta=...` | race_start telemetry changed |
| `[FLIGHT_DBG] track_burst num_gates=N` | Gate map burst (every race) |
| `[FLIGHT_DBG] controls enabled=False/True` | Control loop gate toggled |
| `[AUTO_FLIGHT_DEBUG] race_status ...` | Throttled race telemetry (`AUTO_FLIGHT_DEBUG=1`) |
| `[AUTO_FLIGHT_DEBUG] track burst received num_gates=N` | Gate map burst arrived |
| `[RACE] gate1_watch active=N elapsed=Xs pilot_passed=P` | Gate-1 progress (every 5s) |
| `[RACE] OUTCOME=gate1_fail attempt=N retrying` | Gate 1 missed; sim reset + retry |
| `[RACE] post_reset active=N sim_boot=...` | Telemetry after reset |
| `[RACE] reset sent — click Restart Race...` | May need manual Restart Race |
| `[RACE] GATE1=pass active=1` | First gate cleared; per-gate stall watch begins |
| `[RACE] GATE_ADVANCE active=N` | Sim advanced to next gate |
| `[RACE] gate_progress_watch active=N last_active=M elapsed=Xs` | Post-gate1 progress (every 5s) |
| `[RACE] OUTCOME=gate_stall attempt=N active=X retrying` | No gate advance in 20s; sim reset + retry |
| `[RACE] OUTCOME=success finish_ns=... active=...` | Full course complete |

## Fail / success logic

- **Gate 1 fail:** sim `active_gate_index` stays `< 1` and either:
  - elapsed time exceeds `GATE1_TIMEOUT_S` (default 10s), or
  - elapsed > 8s, pilot hasn't passed visually, and drone crossed gate-0 plane without sim advancing
- **Gate stall (after gate 1):** `active_gate_index` unchanged for `GATE_PROGRESS_TIMEOUT_S` (default 20s) → MAVLink reset + retry
- **GO timing (two-phase mavlink + vision):**
  - Phase 1: must latch `scheduled` or `countdown` branch (rejects instant stale `at_go`)
  - Phase 2: `sim_boot >= go_boot + GO_POST_MARGIN_MS` (default 400ms)
  - Phase 3 (vision): countdown visible in camera ROI, then cleared for N frames
  - Adaptive `go_boot`: scheduled future → `race_start`; countdown/restart → `race_start + 3s`
- **Pre-GO safety:** disarm + **transmit** zero-thrust attitude every 100ms during all preflight waits (`send_safe_hold`)
- **Fresh race_start:** must change from track burst or be scheduled future GO
- **Success:** `race_finish_time_ns >= 0` or `active_gate_index >= gate_count`.

## Env tunables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTO_FLIGHT` | unset | Set to `1` by `make auto` |
| `AUTO_FLIGHT_DEBUG` | `1` (via `auto.py`) | MAVLink race/track + `[FLIGHT_DBG]` preflight logs |
| `AUTO_VISION_PREVIEW` | `1` (via `auto.py`) | Show `auto` camera window with countdown ROI |
| `AUTO_GO_VISION` | `1` (via `auto.py`) | Wait for visual countdown clear before arm |
| `GO_POST_MARGIN_MS` | `400` | Extra ms after telemetry GO before arm |
| `VISUAL_GO_MIN_CLEAR_FRAMES` | `5` | Frames without countdown before visual GO |
| `VISUAL_GO_SEE_TIMEOUT_S` | `8` | Fallback to mavlink-only if no countdown on camera |
| `GATE1_TIMEOUT_S` | `10` | Max seconds after GO to pass gate 1 |
| `GATE1_MIN_ELAPSED_S` | `8` | Min elapsed before early plane-cross fail |
| `GATE1_WATCH_INTERVAL_S` | `5` | Seconds between gate1_watch logs |
| `GATE_PROGRESS_TIMEOUT_S` | `20` | Max seconds without gate advance after gate 1 |
| `CANCEL_CONFIRM_WINDOW_S` | `3` | Seconds to confirm double-cancel |
| `SIM_RESET_WAIT_S` | `5` | Pause after MAVLink sim reset |

## Troubleshooting

### Stuck on `waiting for track_gates`

1. Confirm FlightSim is in a **flight session** (not main menu).
2. Run `make auto` first, then click **Race** or **Restart Race**.
3. If you already raced before the client connected, the one-shot burst was missed — **Restart Race** only.
4. Run `make capture-gates` during Restart Race; should print gates captured.
5. Enable debug: `AUTO_FLIGHT_DEBUG=1 make auto` — look for `[AUTO_FLIGHT_DEBUG] track burst` and `race_status` lines.

### Drone flies before countdown

1. Confirm heartbeat connected (`Connected to system: 1`) **before** clicking Race.
2. Watch the `auto` preview window — green ROI + `countdown` label during 3-2-1.
3. Terminal must show `[RACE] countdown latched` **before** `Race go!` (instant `at_go` = stale telemetry).
4. Paste log from `[FLIGHT_DBG] milestone preflight_start` through `controls_enabled` (~2s after).
5. **Interpretation:** `armed=True` + rising `vel_ned` before `controls_enabled` → sim thrust or safe_hold not winning; check `[FLIGHT_DBG] safe_hold` / `safe_tick`. `visual GO fallback` before 3-2-1 → camera ROI missing countdown.
6. Set `AUTO_GO_VISION=0` only to debug mavlink GO in isolation.

### `Race go!` never prints

- Usually a latch/timing issue (fixed in preflight latch retry).
- Ensure heartbeat connected before clicking Race.
- After gate-1 reset, use **Restart Race** and wait for fresh track + countdown.

### `make fly` fails but `make auto` works

- `make fly` uses saved `rl/data/gate_map.json` — run `make capture-gates` during Restart Race to refresh.
- Try `make fly` with `--flipz` on climb courses.

## Notes

- Sim gates are 0-based; "gate 1" = `active_gate_index >= 1`.
- MAVLink reset (31000) resets drone pose, not the Race UI — use **Restart Race** if track/countdown doesn't reload.
