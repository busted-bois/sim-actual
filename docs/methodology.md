# Methodology: from color detection to known-waypoint navigation

Status: gate 0 + gate 1 cleared reliably (incl. mid-race start). Same approach scales to 6.

## 1. The reframe — perception was never the bottleneck

**Old (color baseline, archived on branch `baseline/color-detection`):**
camera frame -> HSV threshold for the orange gate -> contour -> normalized centroid
(nx, ny) + area -> fly toward the colored blob with pitch/yaw.
Problems: speed uncontrollable, fragile, only ever cleared 1 gate, thrashed on search.

**Key insight:** the sim already hands us a full metric picture over MAVLink:
- `odometry` — our own NED pose (position, velocity, attitude)
- `track_gates` — absolute NED position of every gate
- `active_gate_index` — the official "next gate" pointer

So the unsolved problem is **control**, not perception. We dropped vision entirely and
navigate to known gate positions. Color detection stays archived as a baseline.

## 2. The control architecture

**Discovery:** the sim has NO onboard autopilot. Velocity-NED setpoints
(`set_position_target_local_ned`) are ignored — that template path was a dead stub. The
only channel that actuates the drone is `set_attitude_target` (body roll/pitch/yaw RATES +
thrust). So we build the autopilot ourselves, and commands must stay small/bounded (~0.2)
or the drone flips.

Per control tick (~250 Hz), `simulator/pilot.py`:
1. **Target** = `track_gates[active_gate_index]` (or the Round-1 fallback table).
2. **Carrot** = a point just past the gate along the down-track direction, so we fly
   THROUGH the hoop instead of braking at it.
3. **Velocity estimate** from position differences — the odometry velocity is in an
   unreliable/inverted frame, so we differentiate `odometry` position ourselves (with a
   jump guard so teleports/resets can't spike it).
4. **Body-frame velocity controller:** desired velocity points at the carrot, with the
   target speed tapering down near the gate. Velocity error -> small bounded tilt:
   pitch accelerates/BRAKES (forward), roll strafes (lateral). Braking the error in body
   frame kills leftover momentum in ANY direction — robust to messy starts.
5. **Heading held FIXED** (`yaw_rate = 0`). The drone starts facing down-track (-X, toward
   every gate) and the roll+pitch controller translates in any direction, so no rotation is
   needed. (The sim's yaw-rate response is inverted/positive-feedback — see bug #7.)
6. **Altitude** = P+I on thrust toward the gate's height; trim at true hover (0.5), no D
   term (the vertical-velocity estimate is too spiky to damp on safely).
7. **Sequencing** straight from `active_gate_index` — no homemade pass-detection.

## 3. The debugging journey (each step was a discovery)

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Drone won't move on velocity setpoints | sim has no velocity controller | command attitude-rate + thrust instead |
| 2 | Tumbles / flies off | attitude-angle inner loop with high gain -> huge commands | keep all commands small & bounded |
| 3 | Coasts at 10 m/s, overshoots gates | raw pitch -> runaway speed, no drag | speed controller with active BRAKING |
| 4 | Drifts up for no reason | altitude trim (0.55) sat above hover (0.5) | trim = 0.5 |
| 5 | Flies under/over gates | odometry velocity is wrong-frame | estimate velocity from position |
| 6 | Shoots straight up at start | altitude D term spiked on a teleport | drop D; jump-guard the velocity estimate |
| 7 | Turns ~180° ("scanning backward") | yaw loop sign inverted -> positive feedback | hold heading fixed (yaw_rate = 0) — **unlocked gate 2** |
| 8 | Mid-race start drifts up, no gates | sim broadcasts the gate table only once, at race start | fall back to the known Round-1 track |

## 4. Path to 6 gates
- **Gate-2 orbit:** yaw slowly drifts (roll/pitch coupling) -> body frame rotates ->
  velocity controller spirals. Fix: actively hold yaw at the fixed down-track heading
  (with the correct, inverted sign) instead of leaving it free.
- Then tune speed cap / approach slowdown for the steeper descent and lateral offsets of
  gates 3-6.
