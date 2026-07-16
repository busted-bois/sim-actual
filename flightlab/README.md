# Vertical-axis flight harness

## Human quickstart

1. Start sim, join TRAINING session.
2. `uv sync`
3. `uv run python -m flightlab.run_vertical`

Or: `make vertical`

## CLI

```
uv run python -m flightlab.run_vertical
uv run python -m flightlab.run_vertical --method baro_hold --only V1
uv run python -m flightlab.run_vertical --method pd
uv run python -m flightlab.run_vertical --list
```

Methods: `baro_hold` (default), `pd`, `pid`, `pid_tilt`, `pid_tilt_filt`

### Barometric altitude hold (`baro_hold`)

```
Vz_cmd = kp * (z_up_tgt - zhat_up) - kd * vzhat_up
T_vert = hover + kt * Vz_cmd
thrust = T_vert / (cos(phi)*cos(theta))   # tilt compensation
```

- `zhat` / `vzhat` from VIO/Kalman (`StateEstimator`). Baro fused when `pressure_alt` is finite.
- On many TRAINING builds baro is NaN → zhat rides EKF thrust model.
- Start gains: `kp=1.0`, `kd=0.4`, `kt=0.05`.
- Soft/ESKF: **zero rates when lean≈0** (avoids pitch→70°); soft P toward lean when lean targets set (TILT / gate).
- Tilt compensate full vertical cmd so alt holds when pitching toward a gate. Auto-runs `TILT` (8° lean, sag < 0.3 m).

Logs → `runs/vertical_<utc>/log.jsonl` + `report.md` (includes `zhat`, `vzhat`, `vz_cmd`, `tilt_comp`, `baro_ok`). Exit 0 iff all tests PASS.

## Tests

| # | Test | PASS criteria |
|---|------|---------------|
| V1 | Climb to first-gate pass height, hold 30 s | alt std < 0.15; drift < 0.3; vz std < 0.2; thrust std < 0.02; r/p p2p < 3° |
| V2 | ±5 m steps | overshoot < 15%; settle ±0.25 m in < 2.5 s; bounce decay |
| V3 | vz rate tracking | mean vz within 15% or sat at thrust clamps |
| V4 | Soft land 5 m & 10 m | touchdown \|vz\| < 2; disarm on settle (no timer) |
| TILT | 8° roll lean hold (baro_hold / pid_tilt*) | alt sag < 0.3 m |
