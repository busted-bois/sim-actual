# Round 1 Plan — finish all 6 gates

Goal: 6/6 gates, Round 1, reliability > speed. Color detection = archived baseline, untouched.

## Core idea
Sim gives full map (`track_gates`: NED pos + orient quat + w/h of all gates), own pose
(`odometry`), and official next-gate pointer (`active_gate_index`). So Round 1 = known-waypoint
problem. Fly **velocity setpoints** toward known gate positions. Don't fight vision/thrust.

Why current code stalls: raw `pitch_rate` → speed uncontrollable; fragile `r_frac` pass-detection;
nearest-gate guessing instead of official index.

## Phases

### Phase 0 — archive baseline (DONE = baseline preserved)
- [ ] Keep current color-detection pilot intact (branch/tag before edits). No deletion.

### Phase 1 — instrument + resolve unknowns (one sim run) — DONE
- [x] Telemetry probe (`simulator/diagnostics.py`) logs gate table, index transitions, odom @2Hz.
- [x] Q1: `active_gate_index` increments on pass — CONFIRMED (0->1 at gate 0). Use it; delete
      r_frac pass-detection.
- [x] Q2: all 6 gates IDENTICAL orient quat (0.707,0,0,0.707) = gate planes perp to X-axis,
      fly through along -X. Course monotonic in -X, 2.72m square gates. Carrot =
      next-center + lookahead along inter-gate dir; no normal decode needed.
- [x] Q3: odom accurate to ~0.84m at gate 0 (gate=2.72m), same NED frame as gates. Pure-waypoint
      viable. Full-lap drift re-checked once Phase 4 pilot flies all 6.

Course (NED, z+=down): g0(-23.3,-0.4,-0.03) g1(-46.9,-2.5,+5.07) g2(-74.6,+1.2,+13.67)
g3(-111.5,-5.1,+24.57) g4(-135.5,-0.8,+25.36) g5(-159.2,-4.4,+25.97). ~159m, descends ~26m,
steepest g2->g3 (~17deg), lateral within +-5m. Gentle.

### Phase 2 — gate selection from official index
- [ ] Target = `track_gates[active_gate_index]`, not nearest. Delete passed-gate-rejection,
      search-sweep, post-gate-hover, r_frac peak machinery.

### Phase 3 — control-channel probe — DONE (CRITICAL FINDING)
- [x] Velocity-NED setpoints (`set_position_target_local_ned`) DO NOT actuate the drone.
      Armed + 4 m/s cmd → 6cm drift over 30s. Sim has NO onboard pos/vel controller;
      that template path was an unused stub. Only `set_attitude_target` (body-rates+thrust)
      moves the drone (what the color baseline used). We must build the autopilot ourselves.

### Phase 4 — cascaded attitude controller on known waypoints
Guidance -> velocity err -> desired tilt + thrust -> BODY-RATE commands (set_attitude_rates).
- [ ] Outer: carrot = active gate center + approach_dir*lookahead (fly THROUGH). Horizontal
      v_des = clamp(KP_POS*(carrot-pos), CRUISE). a_des = KP_VEL*(v_des - v_meas).
- [ ] Rotate a_des NED->body by yaw; small-angle target_pitch=-a_fwd/g, target_roll=a_right/g.
- [ ] Inner: pitch_rate=KP_ATT*(tgt-pitch), roll_rate likewise; yaw_rate to hold down-track
      (deadband, avoid +-pi wrap thrash that caused the "scanning"). Thrust = altitude PID
      (reuse baseline) to track carrot_z, tilt-compensated.
- [ ] Tune gains over sim runs: start slow CRUISE~3 m/s, conservative tilt cap.

### Phase 5 — robustness + validation
- [ ] Tune speed cap, lookahead, entry-alignment gate, collision backoff.
- [ ] `make sim` end-to-end, 6/6 across several runs. Push speed only after consistent.

## Decisions (resolved by Phase 1)
- Carrot = next gate center + lookahead along inter-gate direction (no quat normal decode).
- `active_gate_index` alone sequences gates; no fallback pass-detector needed.
- Pure-waypoint guidance (no vision correction) for Round 1.
