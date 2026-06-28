# Flight automation

Opt-in retry loop via `make auto`. Normal `make sim` is unchanged.

## Commands

| Command | Behavior |
|---------|----------|
| `make auto` | Automation mode — fly2 speed controller (ks/improve_speed) + retry until complete |
| `make sim` | Standard sim — vision pilot, single run, no auto-retry |

## Flight controller

`make auto` uses the **fly2 odometry course controller** from `ks/improve_speed` (~1:04 lap target): speed 4.0 m/s, faster yaw, cross-track roll. Uses live `track_gates` from the sim (no saved `gate_map.json` required). `make sim` still uses the vision pilot.

## Workflow

1. **Open FlightSim** and enter a flight session (MAVLink must connect — client waits up to 30s for heartbeat).
2. Run **`make auto`** — terminal shows `waiting for fresh track_gates`.
3. Click **Race** in FlightSim (or **Restart Race** if you already raced and missed the burst).
4. Wait for `Race go!` then `[fly2] loaded N gates` — drone should accelerate toward gate 1.

The sim sends the gate map **once** per race start. If you clicked Race before the client was ready, use **Restart Race** — do not click Race again on an in-progress countdown.

The client **cannot** click Race for you — there is no MAVLink command for that.

## Cancel automation

While `make auto` is running:

- **Ctrl+C**, or
- **Ctrl+any letter**

→ cancels automation and drops into normal sim control (250 Hz loop, no retry). Process keeps running.

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
| `[AUTO_FLIGHT_DEBUG] race_status ...` | Throttled race telemetry (`AUTO_FLIGHT_DEBUG=1`) |
| `[AUTO_FLIGHT_DEBUG] track burst received num_gates=N` | Gate map burst arrived |
| `[RACE] gate1_watch active=N elapsed=Xs pilot_passed=P` | Gate-1 progress (every 5s) |
| `[RACE] OUTCOME=gate1_fail attempt=N retrying` | Gate 1 missed; sim reset + retry |
| `[RACE] post_reset active=N sim_boot=...` | Telemetry after reset |
| `[RACE] reset sent — click Restart Race...` | May need manual Restart Race |
| `[RACE] GATE1=pass active=1` | First gate cleared; run continues to finish |
| `[RACE] OUTCOME=success finish_ns=... active=...` | Full course complete |

## Fail / success logic

- **Gate 1 fail:** sim `active_gate_index` stays `< 1` and either:
  - elapsed time exceeds `GATE1_TIMEOUT_S` (default 10s), or
  - elapsed > 8s, pilot hasn't passed visually, and drone crossed gate-0 plane without sim advancing
- **Success:** `race_finish_time_ns >= 0` or `active_gate_index >= gate_count`.
- **After gate 1:** no auto-retry on later gates; run continues until success or manual cancel.

## Env tunables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTO_FLIGHT` | unset | Set to `1` by `make auto` |
| `AUTO_FLIGHT_DEBUG` | unset | Set to `1` for MAVLink race/track diagnostics |
| `GATE1_TIMEOUT_S` | `10` | Max seconds after GO to pass gate 1 |
| `GATE1_MIN_ELAPSED_S` | `8` | Min elapsed before early plane-cross fail |
| `GATE1_WATCH_INTERVAL_S` | `5` | Seconds between gate1_watch logs |
| `SIM_RESET_WAIT_S` | `5` | Pause after MAVLink sim reset |

## Troubleshooting

### Stuck on `waiting for track_gates`

1. Confirm FlightSim is in a **flight session** (not main menu).
2. Run `make auto` first, then click **Race** or **Restart Race**.
3. If you already raced before the client connected, the one-shot burst was missed — **Restart Race** only.
4. Run `make capture-gates` during Restart Race; should print gates captured.
5. Enable debug: `AUTO_FLIGHT_DEBUG=1 make auto` — look for `[AUTO_FLIGHT_DEBUG] track burst` and `race_status` lines.

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
