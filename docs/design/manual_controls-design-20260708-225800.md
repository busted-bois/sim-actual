# Manual Flight Debug Plan — Movement Keys Not Working

**Branch:** `manual_controls`  
**Status:** DRAFT  
**Date:** 2026-07-08  
**Symptom:** `make manual` opens Tk HUD; drone does not respond to WASD/QE/SPACE/X. HUD shows `telemetry : NO`, `armed : no`, `cmd sent` near-zero even when flying is attempted.

---

## Copy-paste prompt for implementation agent

```
You are debugging manual keyboard flight on branch `manual_controls` in Real Drone Sim.

SYMPTOM (confirmed screenshot 2026-07-08):
- Tk window "Drone Manual Control" opens and stays focused
- HUD: armed=no, telemetry=NO, input may show keys OR stay "-"
- cmd sent: roll/pitch/yaw ~0, thrust ~0.2 (hover fallback)
- Sim shows ACRO mode, race timer running (~7s), drone at 0 km/h
- User bumped speed to 15 km/h via R — discrete input may work

GOAL: Holding W in focused Tk window → non-zero pitch cmd in HUD → sim drone moves.

DO NOT touch unrelated RL/auto code. Scope: manual.py, simulator/setup.py, simulator/controller.py,
simulator/manual_control.py, simulator/manual_ui.py, simulator/mavlink_rx.py, scripts/mavlink_probe.py,
tests/test_manual_control.py.

Follow docs/design/manual_controls-design-20260708-225800.md phase by phase.
Fix root cause(s), add regression tests, verify with make manual + make probe.
Use uv run (not python). No new CLI args in main.py.
```

---

## What we know

### Observed failure signature

| HUD field | Value | Meaning |
|-----------|-------|---------|
| `telemetry` | NO | No ODOMETRY, ATTITUDE, or LOCAL_POSITION_NED in `shared_data` |
| `armed` | no | `mavlink_rx.on_heartbeat` never saw `MAV_MODE_FLAG_SAFETY_ARMED` |
| `cmd sent` | roll/pitch/yaw 0 | Either no movement keys held, or control law output zero |
| `thrust` | 0.2 | Open-loop hover fallback (`HOVER_T=0.27` clipped?) — blind mode |

### Architecture (data path)

```
FlightSim.exe ──UDP 14550──► setup_components (udpin 0.0.0.0:14550)
                              ├─ GCS heartbeat thread (1 Hz) ──► sim registers client
                              ├─ MAVLinkRX thread ──► shared_data {armed, odometry, attitude}
                              └─ Controller.send_attitude_rates ──► SET_ATTITUDE_TARGET

manual.py → arm() once → run_manual_ui → ManualControl.tick @ 90 Hz
```

### Recent change (commit `7646e87`)

`CONTROL_HZ` dropped 250 → 90 (sim ignores >100 Hz setpoint streams). Fixes one known failure mode but user still broken — **telemetry/arming is the stronger signal**.

### Code facts

- `manual.py` sets `MAVLINK20=1` before imports (ODOMETRY id 331 requires MAVLink 2).
- `setup.py` blocks up to 60s for first HEARTBEAT; raises if none. User got past this (UI opened).
- `controller.arm()` sends `MAV_CMD_COMPONENT_ARM_DISARM` once — **no retry, no armed wait**.
- `ManualControl._target_lean`: with W held and no velocity telemetry, `fwd_m=0` → **should still command non-zero pitch**. Zero cmd with W held ⇒ **input not reaching `held` dict** OR screenshot taken with no keys down.
- Tk captures keys only when **this window focused**; sim must not have focus.
- `on_heartbeat` sets `armed` from **any** HEARTBEAT — risk of reading own GCS heartbeat echo without SAFETY_ARMED bit.

---

## PREMISES (validate before fixing)

1. **Telemetry must flow before closed-loop flight is trustworthy** — agree?
2. **Sim ignores offboard setpoints when disarmed or client not registered** — agree?
3. **Zero pitch/roll cmd with W held means input bug, not MAVLink bug** — agree?
4. **Single stale process on UDP 14550 can steal sim traffic** — agree?

---

## Root-cause hypotheses (ranked)

### H1 — MAVLink link broken after connect (HIGH)

Sim connected at startup then stopped streaming pose. HUD `telemetry NO` + `armed no`.

**Check:** `make probe` while sim session active. Expect `ODOMETRY` or `ATTITUDE` counts > 0 every 2s.

**If only HEARTBEAT:** GCS registration lost, wrong port, stale client, or sim mode blocks pose.

### H2 — Stale UDP 14550 client (HIGH)

Prior `make manual` / `make sim` still bound. Sim routes to old process.

**Check:** `netstat -ano | findstr 14550` (Windows). Kill stale PIDs. Restart sim + manual.

### H3 — Arm never took (MED)

`arm()` sent once at boot; sim may need race-started + delay + retry.

**Check:** Log `shared_data["armed"]` transitions in mavlink_rx. Probe HEARTBEAT `base_mode`.

### H4 — Tk input not captured (MED)

`input held : -` while user presses W ⇒ keysym mismatch or focus loss.

**Check:** Press W — HUD must show `input held : W`. If not, fix `manual_ui._norm` / bindings.

### H5 — Command rate still wrong (LOW after 7646e87)

Verify console prints `[manual] control loop running at 90 Hz`. Unit test already guards ≤100.

### H6 — VQ2 / event-block mode (MED if in VQ2)

Some sim builds withhold ODOMETRY in competition modes. User screenshot looks like active race in hangar — confirm Training vs VQ2.

---

## Phase 1 — Reproduce & bisect (no code)

**Prereq:** FlightSim.exe running, logged in, **race started** (timer counting).

| Step | Command | Pass criteria |
|------|---------|---------------|
| 1.1 | `uv run python -m unittest tests.test_manual_control -v` | All green |
| 1.2 | Kill stale clients on 14550 | Only one listener when manual runs |
| 1.3 | `make probe` (30s, sim active) | `ODOMETRY` or `ATTITUDE` present |
| 1.4 | `make manual` — focus Tk, hold W 3s | `input held : W` AND `cmd pitch` ≠ 0 |
| 1.5 | Same, watch sim | Drone pitch / speed change |

**Decision tree:**

```
probe has pose?
  NO  → fix link (Phase 2)
  YES → manual HUD telemetry still NO?
          YES → shared_data / thread bug (Phase 3)
          NO  → armed?
                  NO  → arm retry (Phase 4)
                  YES → sim ignores setpoints (Phase 5)
```

Record console output for: connect line, `[manual] control loop running at 90 Hz`, any `ConnectionResetError`.

---

## Phase 2 — Fix MAVLink telemetry link

**Files:** `simulator/setup.py`, `scripts/mavlink_probe.py`, maybe `manual.py`

| Task | Detail |
|------|--------|
| 2.1 | Add `scripts/free-mavlink-port.ps1` + `make free-port` if missing on branch |
| 2.2 | Document startup order: kill stale → start sim → start race → `make manual` |
| 2.3 | If probe gets pose but manual doesn't: verify **same** `sim_conn` passed to MAVLinkRX (no second bind) |
| 2.4 | Filter heartbeat in `on_heartbeat`: ignore `MAV_TYPE_GCS` so `armed` reflects vehicle only |
| 2.5 | Re-send `_request_data_streams` after race start if sim only streams pose in-flight |
| 2.6 | Log first ODOMETRY/ATTITUDE receipt once (debug flag or print) |

**Done when:** HUD `telemetry : yes` within 5s of race start.

---

## Phase 3 — Fix Tk input path

**Files:** `simulator/manual_ui.py`

| Task | Detail |
|------|--------|
| 3.1 | Map all keysyms: `w/W`, `space/Space`, `minus/minus/underscore`, `equal/equal/plus` |
| 3.2 | On press, set `held[n]=True` and log once `[manual] key down: W` (temporary) |
| 3.3 | `root.focus_force()` on open + after click |
| 3.4 | Unit test: mock `is_pressed`, assert W → non-zero pitch cmd (may already exist — add if missing) |

**Done when:** W held → HUD `input held : W` and `cmd pitch` noticeably negative.

---

## Phase 4 — Fix arming

**Files:** `simulator/controller.py`, `manual.py`

| Task | Detail |
|------|--------|
| 4.1 | `arm()` → retry 5× @ 200ms until `shared_data["armed"]` or timeout |
| 4.2 | Print `Armed OK` / `Arm timeout` |
| 4.3 | Optional: re-arm when race-start detected in encapsulated race status |

**Done when:** HUD `armed : yes` before piloting.

---

## Phase 5 — Fix setpoint acceptance

**Files:** `simulator/controller.py`, `simulator/manual_control.py`

| Task | Detail |
|------|--------|
| 5.1 | Confirm `CONTROL_HZ == 90` and loop sleep matches |
| 5.2 | Confirm `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` on wire |
| 5.3 | Blind-flight fallback: if no telemetry, command **direct lean** on WASD (bypass velocity loop) so user can fly while telemetry is fixed |
| 5.4 | Tune `HOVER_T` — screenshot thrust 0.2 vs constant 0.27; verify takeoff |

**Done when:** Telemetry yes + W held → sim speed > 0 km/h.

---

## Phase 6 — Regression tests

**File:** `tests/test_manual_control.py`

| Test | Assert |
|------|--------|
| W with no odom | pitch cmd < 0 (direct or lean) |
| heartbeat filter | GCS heartbeat does not set armed=True incorrectly |
| arm retry | mock: armed flips after N heartbeats |

Run: `uv run python -m unittest tests.test_manual_control -v`

---

## Phase 7 — Verification checklist

- [ ] `make probe` → pose telemetry
- [ ] `make manual` → `armed : yes`, `telemetry : yes`
- [ ] W/S/A/D/Q/E/SPACE/X each produce expected cmd sign
- [ ] Drone moves in sim window
- [ ] R/F speed step works (already partially confirmed at 15 km/h)
- [ ] Close Tk → clean exit, port freed
- [ ] Unit tests pass

---

## Files in scope

| File | Role |
|------|------|
| `manual.py` | Entry, bind 0.0.0.0:14550, arm, launch UI |
| `simulator/setup.py` | GCS heartbeat, stream requests, connect |
| `simulator/mavlink_rx.py` | Telemetry → shared_data, armed flag |
| `simulator/controller.py` | SET_ATTITUDE_TARGET, CONTROL_HZ, arm |
| `simulator/manual_control.py` | Control law |
| `simulator/manual_ui.py` | Tk input + HUD |
| `scripts/mavlink_probe.py` | Link diagnostic |
| `tests/test_manual_control.py` | Headless regression |

---

## Out of scope

- Gate transition / RL / auto flight
- Sim graphics / framerate
- New main.py CLI flags

---

## Unresolved questions

1. Training mode or VQ2 when screenshot taken?
2. Does `make probe` show ODOMETRY on your machine right now?
3. With W held, does HUD ever show `input held : W`?
4. Any second terminal still running old `make manual`?

---

## The assignment (for human, before next agent run)

Run **only** these two commands with sim race active and paste output:

```powershell
netstat -ano | findstr 14550
make probe
```

Then `make manual`, hold W 5s in focused Tk window, screenshot HUD. That bisects H1/H2/H4 in one minute.
