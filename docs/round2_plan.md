# Round 2 Plan — residual-RL speed layer over the waypoint pilot

Goal: fly Round 2 FASTER than the hand-tuned pilot, without losing the 6/6 finish. R2 keeps
ground-truth (`odometry` + `track_gates`), so this is a CONTROL/speed problem, not perception.

## Core idea
Don't rebuild perception. The pilot already navigates known gates. Wrap it in a Gym env and
train a small PPO policy that outputs SMALL deltas to the pilot's GUIDANCE setpoints
(target-speed mult, lookahead, lateral offset). Inner velocity/tilt/thrust loop + safety
bounds stay untouched. Pilot = stability; RL = the faster, tighter racing line.

Original 8-stage perception+EKF plan is DROPPED (GateNet, Corner/PnP, EKF, dataset-gen) — all
redundant when `odometry`+`track_gates` are handed to us. Libs out: TorchVision, OpenCV,
Albumentations. Libs in: PyTorch, Stable-Baselines3, Gymnasium, pymavlink/MAVSDK, NumPy
(SciPy only if needed for rotations).

## Decisions (resolved via grill)
- Target: speed via RL on ground-truth. Vision NOT used in R2.
- Architecture: residual RL OVER `pilot.py`; inner loop + bounds untouched.
- Action: deltas to guidance setpoints `{target_speed mult, lookahead, lateral offset}` @20 Hz.
  WIDER SYMMETRIC bounds — mult ~[0.4, 2.5], lateral ~±large, lookahead ~±large. Policy can
  brake BELOW pilot schedule to make hard gates, not just speed up.
- Obs ~28–34D: ego (body-vel, roll/pitch, vz, alt-err) + current setpoints + last action +
  NEXT 3 GATES (rel-vector, approach-dir, size). 3 gates = enough to plan a line.
- Reward: completion-gated speed + reliability→speed curriculum. Dense progress + per-gate
  bonus always; TIME reward paid ONLY on full 6/6 finish; large terminal crash/miss penalty.
  Anneal time-weight in AFTER 6/6 is stable.
- Episode start: REVERSE curriculum — start near hardest (g2→g3), expand backward.
- Reset integrity: VERIFY FIRST whether `sim_reset` clears fly-away gate-frame corruption.
- Deploy: gated on ≥20 eval runs (finish-rate ≥ pilot AND median lap-time ↓ ≥10%); runtime
  clamp + watchdog → revert to BARE PILOT on NaN/load-fail/anomaly. Never worse than pilot.

## Build status (2026-06-15)
[x] = code complete + unit-tested (no sim). ⏳SIM = needs the live FlightSim; **BLOCKED**.
All pure logic verified by `make test` (25 passing): `tests/test_rl_core.py`,
`tests/test_pilot_residual.py`, `tests/test_policy_runtime.py` (incl. SB3 export→load→infer).
RL deps live in an optional group (`make rl-sync`); `make sim` needs no ML deps.

**⏳SIM blocker (verified):** the sim is not running and is unreachable — no process, nothing
on UDP 14550, `make verify-reset` returns `NO SIM`. `FlightSim.exe` is a Windows GUI app that
requires logging in with the user's official simulator-account credentials (`Instructions.md`),
which the agent cannot do. The 4 ⏳SIM lines below are turnkey: launch + log in to the sim,
then run the listed `make` target. Sim entry points now fast-fail (15s) instead of hanging.

## Phases

### Phase 0 — VERIFY reset integrity (BLOCKER)
Wide action bounds ⇒ frequent fly-aways ⇒ this decides the whole training harness.
- [x] Probe built: `simulator/verify_reset.py` — arms, flies away, fires `sim_reset`,
      compares the gate table + pose before/after, prints PASS (in-process reset) / FAIL
      (process-restart Contingency).
- [x] ✅ **`make verify-reset` PASS for MILD fly-away (2026-06-15):** flew to (−24.7, 0, −18.4);
      after `sim_reset` gate-frame dev = **0.000 m**, pose → origin, idx → 0. Bonus: reset
      restarts the race and rebroadcasts the gate table.
- [!] ⚠️ **SEVERE fly-away DOES corrupt the frame (2026-06-15):** after a run flew the drone
      to ~1.3 km, `sim_reset` no longer recovered — drone reported pos≈(−227, +384, +1318) right
      after reset and crashed around at altitude. Needed a full FlightSim.exe restart. So the
      in-process reset is fine for normal training crashes, but **wide-bound RL needs corruption
      detection + process-restart** (Contingency) as a backstop, AND action clamps tight enough
      that the drone can't reach the runaway regime.

### Phase 0.5 — pilot traverses all 6 gates (any speed)
Residual premise needs a baseline that already flies the whole track. Today only g0–1 reliable.
- [x] g2 yaw fix implemented: active down-track yaw-hold w/ inverted sign + deadband
      (`pilot.py` `_yaw_hold_rate`, `YAW_HOLD_*`). Reversible via `YAW_HOLD_ENABLED=False`.
- [ ] ⏳SIM **run `make sim`**: confirm yaw-hold sign is correct and the bare pilot gets
      THROUGH all 6 gates; tune `CRUISE`/`APPROACH_SLOWDOWN_K`/`KP_YAW` if needed.

### Phase 1 — Gym env
- [x] `simulator/rl_env.py`: `ResidualRacingEnv` — sim_reset + bare-pilot ferry to the
      reverse-curriculum start gate (`reverse_curriculum_start_gate`); bg control thread.
- [x] Obs builder 32D from `odometry`+`track_gates`, next-3-gates body frame
      (`rl_core.build_observation`, `OBS_DIM=32`) — shape/finite/forward-sign unit-tested.
- [x] Action = guidance-delta @20 Hz held across 250 Hz ticks; wide-symmetric clamps
      (`rl_core.action_to_residual`, applied in `pilot.tick`) — bounds + safety unit-tested.
- [x] Reward: dense progress + per-gate bonus; completion-gated finish time; crash/miss
      (`rl_core.step_reward`/`finish_reward`, env wiring) — unit-tested.
- [x] Termination: plane-crossing outside aperture; `COLLISION` = crash; `active_gate_index`
      = clean pass; timeout (`rl_env.step`, `rl_core` geometry) — geometry unit-tested.

### Phase 2 — PPO reliability phase
- [x] `simulator/rl_train.py`: SB3 PPO, 3×64 MLP, `n_envs=1`, `VecNormalize`, curriculum
      callback (reverse start + time-weight schedule). Export→load→infer SB3-version-checked.
- [ ] ⏳SIM **run `make train`** (after P0): train to 6/6 with time-weight off → `policy.pt`.

### Phase 3 — speed phase + deploy
- [x] Reliability→speed time-weight anneal in the curriculum callback
      (`rl_core.time_weight_schedule`, `RELIABILITY_FRAC`).
- [x] Eval gate built: `make eval` runs ≥20 policy vs bare-pilot episodes, prints finish-rate
      + median time + PASS/FAIL on (finish-rate ≥ pilot AND ≥10% faster) (`rl_train.evaluate`).
- [x] Deploy wired: pilot auto-loads `policy.pt` (lazy torch) + watchdog → bare-pilot fallback
      on NaN/load-fail; `make sim` uses it by default, no flags. Fallback unit-tested.
- [ ] ⏳SIM **run `make eval`** to confirm the deploy gate passes before racing.

## Contingency (decide AFTER P0)
If `sim_reset` corrupts gate frame:
- Process-supervisor relaunch of FlightSim.exe per bad episode (slow → may break real-time PPO), OR
- Surrogate fast-sim pretrain (gym-pybullet-drones / custom NumPy model) then fine-tune on
  FlightSim (sim-to-sim gap on quirky dynamics), OR
- Narrow action bounds to keep the drone recoverable so soft-reset suffices.

## Open / to-confirm
- R2 actually keeps `odometry`+`track_gates` — empirically confirm when R2 sim available; if
  removed, whole plan reverts to the vision/EKF stack.
- Deploy bar (10% time gain) — revisit once baseline lap times known.
- Whether `target_speed` mult <1 ever needed below 0.4 — revisit if hard gates miss.
