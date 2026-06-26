# RL Research Notes — anduril-sim (AI Grand Prix)

Autonomous drone-racing pilot for AI-GP. Drone flies through a sequence of
gates, vision-only (no GPS/absolute coords in competition; sim currently
exposes ground-truth odometry + track data too). Currently a hand-tuned
heuristic/state-machine pilot — prime candidate for RL replacement.

## Repo layout
- `main.py` — entry point: connects MAVLink, arms drone, runs `controller.update()` loop (250 Hz)
- `simulator/setup.py` — wires up MAVLink conn, RX threads, controller
- `simulator/controller.py` — sends MAVLink commands (attitude/velocity/motor), calls `pilot.tick()` each cycle
- `simulator/pilot.py` — **the policy** (535 lines, heuristic state machine). This is what RL would replace/augment.
- `simulator/state_machine.py` — alternate pure FSM (TAKEOFF/CHASE/ADVANCE/SEARCH) — not currently wired into pilot.py's main logic but defines transition semantics
- `simulator/gate_detector.py` — CV: HSV color threshold + contour detection of orange gates → `GateDetection` (centroid, area, w/h px)
- `simulator/gate_estimator.py` — converts detection → `GateEstimate` (bearing, range, lateral offset) with self-calibrating focal length from track ground truth
- `simulator/vision_rx.py` — UDP (port 5600) receives JPEG camera frames, runs gate_detector + obstacle contour detection, populates `shared_data["gate_target"]`, `shared_data["obstacles"]`
- `simulator/mavlink_rx.py` — MAVLink RX thread: odometry, attitude, collisions, track/gate ground-truth, race status
- `simulator/search.py` — `ExpandingSearch` class for systematic gate-search sweeps (alternate impl, not wired into pilot.py main path)
- `simulator/config.py` — all tunable constants + dataclasses (`GateDetection`, `DroneState`, `TrackGate`)
- `simulator/transforms.py`, `timesync.py` — quaternion/geometry helpers, time sync loop

## Simulator I/O (the "environment")

**Connection**: MAVLink over UDP, `udpin:127.0.0.1:14550` (configurable). Vision frames over separate UDP socket on port 5600.

### Observations (`shared_data` dict, written by RX threads)
- `odometry`: dict with x,y,z (NED position), vx,vy,vz, quaternion (qx,qy,qz,qw), roll/pitch/yaw speeds
- `pos_ned`, `vel_ned`, `yaw_rad`, `yaw_rate` — from LOCAL_POSITION_NED/ODOMETRY/ATTITUDE
- `gate_target`: `{detected: bool, nx, ny, r_frac}` — vision-derived, normalized gate centroid offset (-1..1) and area fraction of frame (0..1). Updated per camera frame.
- `obstacles`: list of `{nx, ny, r_frac}` from non-gate-colored contours in frame
- `track_gates`: ground-truth list of gates `{position_ned, orientation_ned (quat), width, height}` (from ENCAPSULATED track-data packets — likely qualifier-only / debug, real competition won't provide this)
- `active_gate_index`, `race_started`, `race_finish_time_ns`, `race_status` — race progress telemetry
- `collision`: `{id, threat_level, delta}` when collision event fires
- `armed`: bool from heartbeat

### Actions (sent via `controller`)
Three control modes (`controller.set_control_mode(...)`):
1. **"attitude"** (used by current pilot) — `set_attitude_rates(roll_rate, pitch_rate, yaw_rate, thrust)`. All rates in rad/s, thrust in [0,1]. This is the natural RL action space: `Box(4,)` continuous.
2. **"position"** — `set_velocity_ned(vx, vy, vz, yaw_rate)` — velocity setpoints in NED frame
3. **"motor"** — raw motor RPM targets (currently all zeros / unused)

Control loop runs at `CONTROL_HZ = 250` (controller.py); vision frames arrive asynchronously (camera FPS, much slower).

## Current heuristic pilot (simulator/pilot.py)

State held in `Pilot` instance (not RL — hand tuned):
- Modes: hover, vision-guided approach (`_fly_toward_gate_vision`), telemetry-fallback approach (`_fly_toward_gate_telemetry`, uses `track_gates` ground truth), search (`_do_search`, yaw-sweep), post-gate hover, collision-hold
- Phases within approach: STABILIZE (hover + align yaw/altitude until centered for `STABILIZE_HOLD_S`) → ADVANCE (pitch forward) → fly-through detection (gate area shrinking after peak = passed) → POST_GATE_HOVER → SEARCH for next gate
- Altitude held via PID (`_altitude_thrust`: Kp=0.15, Ki=0.01, Kd=0.20, trim=0.55, target z=-5.0 NED i.e. 5m up)
- Gate-passed bookkeeping: `_passed_gate_positions`, `_completed_gates`, `_gates_passed`, with angular/distance rejection (`_is_passed_gate`) to avoid re-targeting a gate just flown through
- Obstacle avoidance: simple "if obstacle within `OBSTACLE_CLEAR_ZONE` of center, yaw away and stop"

Key tunable constants (pilot.py top): `HOVER_THRUST=0.5`, `CRUISE_THRUST=0.55`, `CRUISE_PITCH_RATE=-0.2`, `VISION_YAW_GAIN=40deg`, `VISION_CENTER_DEADBAND=0.30`, `VISION_PROXIMITY_R_FRAC=0.10`, `Z_TARGET_NED=-5.0`, etc.

`simulator/config.py` has a second, partially-overlapping set of constants (for the `state_machine.py` + estimator path), e.g. `PASS_RANGE_M`, `PASS_AREA_FRAC`, `ALTITUDE_TARGET_M=3.0`, `LOST_FRAMES_THRESHOLD=30`, `TAKEOFF_TIMEOUT_S=10.0`. Two parallel control-logic implementations exist — `pilot.py`'s `Pilot.tick()` is the one actually wired into `controller.py`.

## Reward signal candidates (for RL)
From `shared_data`, derivable signals:
- `race_status.active_gate_index` increments = gate passed (sparse reward)
- `race_finish_time_ns` — terminal reward / time-based shaping (minimize race time)
- `collision` events — negative reward / penalty, also triggers a 2s hard hold in current pilot
- `gate_target.r_frac` growth — shaping reward for approaching gate
- `gate_target.nx/ny` magnitude — shaping reward for centering
- Distance-to-next-gate via `track_gates` + `odometry` (ground truth, qualifier-only)
- Episode reset: `controller.send_sim_reset_command()` (MAVLINK_CMD_SIM_RESET = 31000) — useful for RL episode boundaries

## Sensor realism note
Competition rules: **no GPS / absolute coordinates** — only onboard vision. `track_gates` (ground truth gate positions) and `odometry` (full state incl. position) are available in the simulator currently but may not reflect what's available in later qualifier rounds / physical hardware. For RL meant to generalize toward the real competition, prefer observations derivable from `gate_target` (vision), `obstacles` (vision), `attitude`/IMU-like signals (roll/pitch/yaw rates), not raw NED position/track ground truth.

## Integration points for RL
- Drop-in replacement: implement a new class with `.tick()` matching `Pilot`'s interface, swap in `controller.py:136` (`self.pilot = Pilot(self, data)`)
- Action space: 4-dim continuous (roll_rate, pitch_rate, yaw_rate, thrust) via attitude mode — matches current pilot, simplest to swap
- Episode loop would live in `main.py` (currently an infinite `while is_running` loop with no reset/termination logic — would need to add episode termination on collision/finish/timeout + `send_sim_reset_command()`)
- Sim runs as external process (`FlightSim.exe`, gitignored, Windows only) — RL training loop must drive this real-time sim via MAVLink/UDP; no headless/fast-forward mode apparent, so consider wall-clock cost per episode and whether the sim supports speed-up

## Dependencies (pyproject.toml)
numpy, opencv-python, pymavlink, matplotlib, keyboard. No RL libs (gymnasium/stable-baselines/torch) yet — would need to add via `uv add`.
