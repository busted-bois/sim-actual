# Jacobian / Geo-Control Demo

**Branch:** `feat/hybrid-control-ram-claude`
**File:** `rl/demo_jacobian.py`

---

## What "Jacobian" means here

The Jacobian is NOT a trained matrix. It is the analytic rotation `R(q)^T` already inside
`rl/geo_control.py:velocity_to_action()`. It maps a desired world-frame (NED) velocity error
into the drone's body frame so the right attitude-rate commands can be computed:

```
v_err_fwd = v_err_x*cos(yaw) + v_err_y*sin(yaw)   <- Jacobian col projection, forward axis
v_err_rgt = v_err_x*sin(yaw) - v_err_y*cos(yaw)   <- Jacobian col projection, right axis
```

At yaw=0 (drone faces North): forward=North, right=East -- body axes align with NED.
At any other yaw: the cos/sin terms rotate the projection so the attitude commands
stay correct regardless of heading. That rotation IS the Jacobian structure.

The full inner-loop chain inside `velocity_to_action()`:

```
Step 1  World->Body projection (Jacobian R(q)^T):
          v_err_fwd = v_err_x*cos(yaw) + v_err_y*sin(yaw)
          v_err_rgt = v_err_x*sin(yaw) - v_err_y*cos(yaw)

Step 2  Desired attitude from velocity error:
          target_pitch = -k_lean * v_err_fwd     (forward = nose-down)
          target_roll  =  k_lat  * v_err_rgt     (right   = right roll)

Step 3  Attitude P-controller -> rates:
          pitch_rate = k_att * (target_pitch - current_pitch)
          roll_rate  = -k_att * (target_roll  - current_roll)
          yaw_rate   = -yaw_rate_des              (sign measured on fly2)

Step 4  Tilt-compensated thrust:
          thrust = (hover_thrust + k_vz*vz_err) / (cos(roll)*cos(pitch))
          (banked rotor loses lift; divide to compensate during turns)
```

---

## Architecture: two tiers

```
TIER A -- Pure geometric (no ML, what the demo runs)
---------------------------------------------------
pursuit_velocity()          outer loop: point at gate
      |
      v  v_des_ned (3,) + yaw_rate_des
      |
encode_velocity_action()    normalize to [-1,1]
      |
env.step()
  -> velocity_to_action()   THE JACOBIAN LAYER  (geo_control.py)
  -> roll_rate, pitch_rate, yaw_rate, thrust
  -> internal physics

TIER B -- Hybrid ML (what the branch is building toward)
---------------------------------------------------------
BC/PPO policy               learned outer loop
      |
      v  v_des_ned (3,) + yaw_rate_des  (same interface as Tier A)
      |
velocity_to_action()        SAME JACOBIAN LAYER -- unchanged
      |
MAVLink / internal physics
```

The geo_control Jacobian layer is identical in training (`env.py`) and deployment
(`deploy.py`). That is the whole point: the policy sees the same inner loop it trained
against, so behavior transfers from sim to sim.

---

## How to run

### Tier-A demo (no sim, no ML weights needed)

```bash
# Stage 0: single gate, straight approach
uv run -m rl.demo_jacobian

# Stage 1: two gates, one turn
uv run -m rl.demo_jacobian --stage 1 --episodes 6

# Stage 2: full 6-gate course
uv run -m rl.demo_jacobian --stage 2 --episodes 3

# Save 3-D trajectory plot (matplotlib)
uv run -m rl.demo_jacobian --stage 1 --plot

# Compare modes side by side
uv run -m rl.demo_jacobian --stage 0 --episodes 6 --modes jacobian expert random --plot

# Makefile shortcut (stage 0, 3 episodes, saves plot)
make demo-jacobian

# CI selftest (pass/fail, matches env.py baseline)
uv run -m rl.demo_jacobian --selftest
```

### Tier-B path (needs FlightSim + race running)

```bash
# 1. Record BC demos from fly2
uv run -m rl.fly2 --mode course --log rl/data/demo.jsonl

# 2. BC warm-start
uv run -m rl.bc_dataset --demo rl/data/demo.jsonl

# 3. PPO fine-tune on top of BC weights
uv run -m rl.train_ppo --bc rl/data/demo.jsonl

# 4. Fly on live sim (Tier A -- no ML):
uv run -m rl.fly_jacobian --mode course

# 4. Fly on live sim (Tier B -- trained policy):
uv run -m rl.deploy

# geo_control stays identical in all 4 steps -- only the outer loop changes.
```

---

## Key files

| File | Role |
|------|------|
| `rl/geo_control.py` | `velocity_to_action()` -- the Jacobian inner loop |
| `rl/demo_jacobian.py` | Tier-A demo: gate pursuit -> geo_control -> **internal env** |
| `rl/fly_jacobian.py` | Tier-A demo: same stack on **live FlightSim** via MAVLink |
| `rl/env.py` | Training env: inner loop runs each physics substep |
| `rl/deploy.py` | Live sim: same inner loop with `DEPLOY_GAINS` (hover 0.27) |
| `rl/spec.py` | `scale_velocity_action`, `encode_velocity_action`, shared constants |
| `rl/observation.py` | 24-D gate-relative observation for the ML policy |
| `rl/train_ppo.py` | PPO trainer with optional BC warm-start |
| `rl/fly2.py` | `--log` flag records obs + velocity intent for BC demos |

---

## Gains: internal model vs live sim

| Gain | Internal model (`TRAIN_GAINS`, `DEMO_GAINS`) | Live sim (`DEPLOY_GAINS`, `GeoGains()` defaults) |
|------|----------------------------------------------|--------------------------------------------------|
| `hover_thrust` | 0.5 | 0.27 |
| `k_lat` | 0.2 | 0.11 |
| `roll_max` | 0.22 rad | 0.16 rad |
| `brake_max` | 0.18 rad | 0.07 rad |

Switch `hover_thrust` to 0.27 and tighten the lateral gains when running on FlightSim.
Run `uv run -m rl.dynamics_id` to re-measure if the sim changes.

---

## Demo results (baseline, no ML)

| Stage | Description | Gate pass rate |
|-------|-------------|----------------|
| 0 | 1 gate, straight | 5/6 (83%) |
| 1 | 2 gates, one turn | ~3/6 (50%) -- gets gate 1, misses gate 2 |
| 2 | 6 gates, full course | ~0/18 -- straight pursuit can't corner |

The drop from stage 0 to stage 1 is exactly why the ML outer loop is needed: a hand-coded
pursuit law can't anticipate corners. PPO learns to cut toward the next gate while still
clearing the current one. The Jacobian inner loop stays the same.

---

## What was added in this session

1. **`rl/geo_control.py`** -- `velocity_to_action()`, `GeoGains` dataclass, tilt-compensated
   thrust (`c57adc9`). The Jacobian inner loop.

2. **`rl/demo_jacobian.py`** (new) -- Tier-A demo. Pure geometric controller in
   `GateRacingEnv`: gate pursuit sets `v_des`; geo_control executes it; prints the
   Jacobian breakdown and per-episode stats; optional 3-D matplotlib trajectory plot.

3. **`Makefile`** -- `demo-jacobian` target; `demo_jacobian --selftest` added to `rl-test`.

4. **`docs/HYBRID_CONTROL_SESSION.md`** -- Session handoff doc from Cursor.

5. **`docs/JACOBIAN_DEMO.md`** -- This file.
