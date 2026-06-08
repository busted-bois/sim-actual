# VQ1 Live Validation Report

Generated from local tracking CSV analysis (`make validate-log`). Re-run after each FlightSim session.

## How to validate a live run

1. Start FlightSim qualifier session
2. `make sim` — complete at least one lap
3. `make validate-log` — analyze `logs/tracking_state_*.csv`

## Latest log analysis (baseline)

| Log | Healthy rows | Z range (m) | vz std | Altitude stable |
|-----|--------------|-------------|--------|-----------------|
| tracking_state_1780780979 | 205 | -15.7 .. 52.2 | 37.4 | NO |
| tracking_state_1780886366 | 255 | -338.6 .. 1.4 | 57.3 | NO |
| tracking_state_1780887207 | 732 | -1918.5 .. 0.2 | 24.4 | NO |
| tracking_state_1780887351 | 564 | -1489.4 .. 1.2 | 37.0 | NO |

## Findings

- **Tracking altitude diverges** during IMU propagation — z drifts hundreds of meters negative while vz spikes >50 m/s.
- **Yaw and horizontal tracking** appear functional (drone moves through course in logs).
- **Vision + gate steering** runs but altitude PID fed by tracker vz needs damping/clamping.
- **No written lap-time record yet** — add `last_gate_race_time` from console `Race finished!` line after successful lap.

## Tuning applied (P1-14)

See [simulator/flight_config.py](../simulator/flight_config.py):

- Increased `LOCAL_NED_BLEND` — faster correction from sim NED
- Increased `KD_Z`, reduced `KI_Z` — damp altitude oscillation
- Added `ALTITUDE_VZ_CLAMP` — ignore tracker vz spikes in PID
- Reduced `V_MAX_BODY`, increased `V_TURN_SLOWDOWN` — safer VQ1 speeds

## Definition of Done checklist

- [ ] `make test` — 127 passed
- [ ] `make sim` — connects, arms, race GO
- [ ] Full lap without manual intervention
- [ ] `Race finished!` logged with `last_gate_race_time`
- [ ] `make validate-log` — altitude stable YES on latest log
- [ ] Sim restart → re-arm → second lap

## Next live run goals

1. Record lap time from `Race finished!` console output
2. Count collisions per lap
3. Re-run `make validate-log` until altitude stable passes
