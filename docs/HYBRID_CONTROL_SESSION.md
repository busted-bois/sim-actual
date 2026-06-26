# Hybrid Control — Session Handoff (Jun 2026)

Saved from Cursor chat before closing. Use this to resume work.

---

## Project goal

Autonomous drone gate racing (AI Grand Prix). Move from brittle hand-tuned `fly2.py` to **ML + geometric inner loop**: policy chooses *where/how fast to go*; shared `geo_control` converts to attitude rates + thrust.

---

## Key insight: why Kunal's ~1:00 fly2 felt "random"

Not RNG — **marginally stable tuning**:
- Human ENTER timing at countdown varies launch state
- High speed (4 m/s) + weak recovery → one gate clip → never recenters
- No stabilize/recovery mode (unlike `simulator/pilot.py`)
- `lat_align` floor 0.3 still fast when badly off-line

Stable main fly2 (~1:30) vs `ks/improve_speed` (~1:00, ~1 in 3–4 tries).

---

## Architecture (agreed plan)

```
Policy (MLP/PPO)  →  v_des NED (3) + yaw_rate_des (1)  [normalized -1..1]
       ↓
geo_control.velocity_to_action()  →  roll/pitch/yaw rates + thrust
       ↓
MAVLink / internal physics (env.py)
```

**Contract:** same `geo_control` in training (`env.py`) and deploy (`deploy.py`).

**Observation:** 24-D gate-relative vector (`rl/observation.py`) — unchanged.

---

## Branch split (two agents)

| Branch | Owner | Contents |
|--------|-------|----------|
| `feat/hybrid-control-ram-claude` | Claude Code | `geo_control.py`, `fly2 --log` for BC demos |
| `feat/hybrid-control-cursor` | Cursor | `env`, `deploy`, `spec`, `bc_dataset`, `train_ppo`, `policy.py` |

**Integration branch:** `feat/hybrid-control-ram-claude` merged cursor at `de0cd78`.

### Commits (off `main` @ `1d12c7a`)

- `a502018` — `rl/geo_control.py` inner loop
- `1110c35` — `fly2 --log` JSONL BC demos
- `852a62d` — Cursor hybrid stack (env/deploy/train/bc)
- `de0cd78` — merge cursor into ram-claude

---

## Files added/changed

| File | Role |
|------|------|
| `rl/geo_control.py` | `velocity_to_action()`, `GeoGains`, `--selftest` |
| `rl/spec.py` | `scale_velocity_action`, `encode_velocity_action` |
| `rl/env.py` | Hybrid action space + inner loop each substep |
| `rl/deploy.py` | Live sim: `DEPLOY_GAINS` (hover 0.27) |
| `rl/bc_dataset.py` | Train BC from fly2 JSONL |
| `rl/policy.py` | Shared `StandalonePolicy` MLP |
| `rl/train_ppo.py` | `--bc demo.jsonl` warm-start + PPO |
| `rl/fly2.py` | `--log path.jsonl` records obs + v_des intent |
| `Makefile` | `train-bc`, `train-hybrid`, geo/bc in `rl-test` |

---

## Commands

```bash
# Offline self-tests (no sim)
make rl-test
uv run -m rl.geo_control --selftest
uv run -m rl.env --selftest
uv run -m rl.train_ppo --quick
uv run -m rl.deploy --selftest

# Record BC demos (needs FlightSim + race)
uv run -m rl.fly2 --mode course --log rl/data/demo.jsonl

# Train
make train-bc                              # BC only
make train-hybrid                          # BC + PPO (--bc rl/data/demo.jsonl)

# Fly trained policy on live sim
make fly-policy
```

---

## Open gaps / next steps

1. **Record demos** — `demo.jsonl` not created yet (needs live sim run).
2. **Physics alignment** — training env `HOVER_THRUST=0.5` vs live sim `0.27`; run `dynamics_id`, tune `TRAIN_GAINS` / env.
3. **Expert selftest** — stage 1+ velocity expert clears 0 gates in internal env; PPO needs demos + tuning.
4. **Merge to main** — after benchmark flight on sim.
5. **Vision path later** — GateNet + drop odometry for competition rules.

---

## ML path summary (for clean slate)

- Don't need ImageNet policy; need **BC from fly2** + **PPO fine-tune**.
- Jacobian/geometric = `geo_control` (not trained).
- Pretrained useful for **GateNet** vision only (phase 2).

---

## GitHub

- Repo: `busted-bois/sim-actual`
- Cursor branch pushed: `origin/feat/hybrid-control-cursor` @ `852a62d`
- Current integration: `feat/hybrid-control-ram-claude` @ `de0cd78`

---

## Coordination rule

Two agents → **separate branches**, merge on schedule. One owner per file per sprint. Pull before every edit. Claimed split: Claude = `geo_control` + `fly2 --log`; Cursor = env/train/deploy/bc.
