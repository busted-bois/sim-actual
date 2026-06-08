# Vision Gate Navigation Plan

## Phase 1 - Perception

- [x] Use vision-only gate positions.
- [x] Put CV detection in `simulator/gate_detector.py`.
- [x] Keep `vision_rx.py` to frame ingest plus latest detection/shared data.
- [x] Detect high-contrast `#f3390f` / `rgba(243,57,15)` gate via HSV/color mask first.
- [x] Treat distractor tolerance as secondary.
- [x] Gate shape: largest valid red-orange square ring, plausible area/aspect.
- [x] Accept partial gate only while tracking prior lock.
- [x] Aim at hole center when inner hole detected; fallback orange bbox center.
- [x] Estimate range from apparent gate size.
- [x] Range primary: `2.7m * fx / outer_bbox_width_px`.
- [x] If inner hole found, compare against `1.5m * fx / inner_width_px`.
- [x] Reject or lower confidence if outer/inner range mismatch `>40%`.
- [x] Use gate dimensions: outer `2.7m x 2.7m`, inner `1.5m x 1.5m`, depth `0.26m`.
- [x] Use camera intrinsics: `640x360`, `fx=fy=320`, `cx=320`, `cy=180`, `+20deg` up-tilt, `30Hz`, `90deg` FOV.
- [x] HSV: red wrap ranges `H 0-15` or `170-179`, `S >= 120`, `V >= 80`.
- [x] Apply morphology open/close to mask.
- [x] Contour validation: min area `0.5%` frame, outer aspect `0.75-1.33`, fill ratio `0.15-0.75`.
- [x] Inner hole aspect `0.7-1.4` if found.
- [x] Ring area target about `0.69`, wide tolerance for perspective/clipping.
- [x] Confidence: `0.25 area + 0.2 aspect + 0.2 fill + 0.2 hole + 0.15 temporal_stability`, clamp `0..1`.
- [x] Lock threshold `>=0.55`; strong lock `>=0.75`.
- [x] Add optional lightweight text debug logs via constant.
- [x] Logs include bbox, hole center, confidence, range, command values.
- [x] Log at 5Hz when debug enabled.
- [x] Log state changes immediately: lock, lost, pass, scan.

## Phase 2 - Control

- [x] Command body-frame velocity plus yaw/yaw-rate.
- [x] Put state/control policy in `simulator/gate_navigation.py`.
- [x] Keep `controller.py` to command emission.
- [x] Use vertical velocity from image y-error.
- [x] Camera tilt compensation deferred until sign/behavior verified in sim.
- [x] Adaptive speed: faster when centered/confident/far-mid range; slower when close/off-center/low confidence.
- [x] Optimize reliability before lap time.
- [x] MAVLink: use `SET_POSITION_TARGET_LOCAL_NED` with `MAV_FRAME_LOCAL_NED`, matching `sk/hackathon`.
- [x] Convert body forward/lateral intent to local NED with current yaw; command `vz=0` explicitly.
- [x] Horizontal control: yaw-rate primary, small `vy` assist.
- [x] Errors: `ex=(target_x-cx)/cx`, `ey=(target_y-cy)/cy`.
- [x] `yaw_rate=0.7*ex`, clamp `+-0.6 rad/s`.
- [x] `vy=0.0`; lateral correction disabled until frame response is verified.
- [x] `vz=0.0`; vertical image correction disabled until sign verified.
- [x] Crawl `vx` range `0.05-0.25 m/s`.
- [x] No/weak lock: `0.05 m/s`.
- [x] Strong centered lock and range `>5m`: `0.25 m/s`.
- [x] Near `<2m`: cap `0.12 m/s`; very near `<1m`: `0.08 m/s`.

## Phase 3 - Gate Lifecycle

- [x] Detect pass by vision: gate grows/clips/disappears after near centered lock.
- [x] On loss: short memory coast, then yaw scan.
- [x] Pass candidate: strong/centered lock and range `<0.8m` or outer bbox covers `>60%` frame width/height.
- [x] Declare pass when candidate then gate disappears/clips for `5` frames.
- [x] Lost after `6` consecutive no-detect frames (`0.2s` at 30Hz).
- [x] Coast last command damped for `10` frames (`0.33s`).
- [x] Scan after coast.
- [x] Reset lock history after `30` lost frames (`1s`).
- [x] Scan: `vx=0.05 m/s`, `vy=0`, `vz=0`, `yaw_rate=0.2 rad/s`.
- [x] Scan toward last known horizontal error sign.
- [x] If no history, alternate direction every `2s`.

## Phase 4 - Model Fallback

- [x] Implement color/geometry first.
- [x] Leave ONNX/lightweight model fallback for later.
- [x] Define interface: `GateDetector.detect(frame) -> GateDetection | None`.
- [x] Classical detector implements interface first.
- [x] ONNX later implements same interface; no dependency yet.
- [x] ONNX trigger: color detector loses lock `>10` frames or finds multiple similar candidates.
- [x] Shared data includes `threading.Lock`.
- [x] `vision_rx` writes `latest_detection`, `latest_frame_id`, `latest_vision_time`.
- [x] `controller.update()` reads snapshot under lock, computes command outside lock.
- [x] No blocking CV in control loop.
- [x] Controller treats detection stale after `0.15s`; then loss/coast logic applies.

## Unresolved Questions

None. Next work is sim tuning/verification.
