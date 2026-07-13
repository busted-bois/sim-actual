# Vertical-axis flight harness

## Human quickstart

1. Start sim, join TRAINING session.
2. `uv sync`
3. `uv run python -m flightlab.run_vertical`

Or: `make vertical`

## CLI

```
uv run python -m flightlab.run_vertical --method pd
uv run python -m flightlab.run_vertical --method pid --only V1
uv run python -m flightlab.run_vertical --list
```

Methods: `pd`, `pid`, `pid_tilt`, `pid_tilt_filt`

Logs → `runs/vertical_<utc>/log.jsonl` + `report.md`. Exit 0 iff all tests PASS.

## Tests

| # | Test | PASS criteria |
|---|------|---------------|
| V1 | Hover 3 m / 30 s | alt std < 0.15; drift < 0.3; vz std < 0.2; thrust std < 0.02; r/p p2p < 3° |
| V2 | ±5 m steps | overshoot < 15%; settle ±0.25 m in < 2.5 s; bounce decay |
| V3 | vz rate tracking | mean vz within 15% or sat at thrust clamps |
| V4 | Soft land 5 m & 10 m | touchdown \|vz\| < 2; disarm on settle (no timer) |
