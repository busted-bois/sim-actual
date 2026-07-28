# The Blue-Line Method

Reference for the cyan-corridor navigation stack as it exists across `blueline`,
`blueline-2`, `ks_vision+blueline` and this branch. Ordered high-level first;
the later parts are line-by-line.

| Field | Value |
|-------|-------|
| **Written against** | `ks_vision+blueline` @ `67d866f` + uncommitted working tree |
| **Also covers** | `blueline` @ `4844d3a`, `blueline-2` @ `77ceaec` |
| **Core files** | `simulator/track_line.py`, `simulator/blue_line_vision.py`, `simulator/blue_line_pilot.py` |

> **On `classical+blueline` (this branch), only the detector half applies.**
> `simulator/blue_line_vision.py`, its `track_line.track_mask` backend and the
> `VisionRX` wiring are present and live. `simulator/blue_line_pilot.py`,
> `auto_blueline.py`, `tests/test_blue_line_pilot.py` and `rl/blueline_env.py`
> are **not** in the tree — so §4 (the standalone control law), the
> `BlueLinePilot` rows of §5, and §6's `test_blue_line_pilot.py` row describe
> other branches only. Here the corridor's four scalars feed
> `GPPilot.TrackVirtualGate` (no-gate fallback) and `_course_direction_cue`
> (post-pass SEARCH direction); flight is `make classical blue` → `GPPilot`
> with the unmodified `rl_failed101` control law. See §1.1.

---

## Contents

1. [What the method is](#1-what-the-method-is)
2. [Architecture and data flow](#2-architecture-and-data-flow)
3. [The detector, in detail](#3-the-detector-in-detail)
4. [The control law, in detail](#4-the-control-law-in-detail)
5. [Branch variants](#5-branch-variants)
6. [Tests, knobs, and gaps](#6-tests-knobs-and-gaps)

---

## 1. What the method is

The AI-GP course is marked by a glowing **cyan floor ribbon** — two roughly
parallel rails that bound a corridor the drone is supposed to fly inside, with a
white-hot core and an occasional gold start-ramp beam flanked by the same rails.

The blue-line method treats that corridor as the **primary navigation cue**. A
classical HSV colour detector extracts the rails from each camera frame, reduces
them to four scalars, and a control law flies to drive those scalars to their
targets. No neural network is required in the loop; YOLO gate detection appears
only as a bounded secondary assist that can be switched off entirely
(`BL_GATE_ASSIST=0`).

**Why the ribbon rather than the gates.** Gates are visible only intermittently,
and PnP range past ~25 m is measurably noise (`GATE_ASSIST_MAX_RANGE_M = 35.0`
is already a generous ceiling; the reacquire logic on `blueline-2` tightens it
to 25 m). The ribbon is visible continuously, and — more importantly — it
encodes *the path between* gates, which a gate-only pilot has to guess at. The
GP pilot's own history shows this: the branch log goes "passes 3 gates", "made
it through gate 4", "5 gates cleared", "barely made it to gate 6" as the ribbon
fallback was added and tightened.

**Four scalars.** Everything downstream consumes only these:

| Signal | Meaning | Sign convention |
|---|---|---|
| `cx_norm` | corridor mid-line, lateral | −1 image-left … +1 image-right |
| `cy_norm` | corridor height in the image | −1 top … +1 bottom |
| `heading_err` | rad; slope of the corridor centreline across depth | + = corridor vanishes **right** of centre |
| `width_norm` | rail separation ÷ image width | 0 when no band resolved both rails |

Plus quality metadata: `found`, `conf`, `single_rail`, `left_found`,
`right_found`, `frame_id`, `points`.

### 1.1 Two ways the signal is consumed

**(a) Standalone corridor pilot** — `blueline`, `blueline-2`.
`BlueLinePilot` steers directly on the corridor error and owns the whole flight:
race gating, speed control, search behaviour, telemetry. The blue line is
primary; the YOLO gate is a bounded assist.

**(b) Virtual gate into the GP guidance law** — `ks_vision+blueline`,
`classical+blueline`. **This is the only mode on this branch.**
`GPPilot.TrackVirtualGate` converts the corridor into a *synthetic gate* 4 m
ahead (`body_x_m = TRACK_LOOKAHEAD_M`), displaced in body-y by where the
corridor sits, held level (`body_z_m = 0`). Feeding that as `vision` makes the
already-proven GP guidance bank to centre the ribbon — **no new control law**.
The blue line is a fallback used when no real gate is locked, and a
direction cue for post-pass search.

### 1.2 Branch map

| Branch | Head | Blue-line role | Distinctive content |
|---|---|---|---|
| `blueline` | `4844d3a` | Standalone `BlueLinePilot`, primary | v1 **centroid** detector; YOLO gate assist; race gating; flight logs; `rl/blueline_env.py` + PPO training |
| `blueline-2` | `77ceaec` | Same, plus experiments | v2 **inner-edge scanline** detector behind `BL_INNER_EDGE`; passive A/B probe `simulator/blue_line_probe.py`; post-collision reverse phase |
| `ks_vision+blueline` | `67d866f` | Fallback + virtual gate inside `GPPilot` | v3 **per-band rail** detector on the shared `track_line` mask; `TrackVirtualGate`; gate-commit punch-through |
| `classical+blueline` | this branch | Fallback + virtual gate inside `GPPilot`, **detector only** | v3 detector + `TrackVirtualGate` + `_course_direction_cue` on the untouched `rl_failed101` control law; no `BlueLinePilot` |
| `main`, everything else | `0a60f92` | absent | — |

`simulator/track_line.py` — the measured HSV mask and the single-ribbon
`detect_track()` — exists on `ks_vision+blueline` and this branch. It was
recovered in `7140770` from the discarded `useless_files` work.

### 1.3 Entry points

```bash
make sim        # blueline / blueline-2: runs auto_blueline.py (BlueLinePilot)
                # ks_vision+blueline:    runs main.py (GPPilot + virtual gate)
make blueline   # blueline / blueline-2: alias of `make sim`
make bl-probe   # blueline-2 only: passive estimator A/B, never commands the drone

# classical+blueline (this branch) — GPPilot, corridor as fallback + search cue:
make classical blue   # the branch's flight; `classical` and `blue` are two
                      # goals sharing one prerequisite, so auto_gp.py runs once
make classical-blue   # same, hyphenated
make control-flight   # same target again (also `make auto-gp`)
```

`auto_blueline.py` sets `AUTO_PILOT=blueline`, arms, opens the vision window,
and runs `controller.update()` until Ctrl+C. It deliberately does **not** use
the overnight `AUTO_FLIGHT` retry loop.

---

## 2. Architecture and data flow

```
FlightSim.exe
  │  camera frames, UDP :5600                    MAVLink, UDP :14550
  ▼                                                        ▲
VisionRX thread ── BlueLineTracker ──► shared_data["blue_line"]
  │                                       │
  │  data["pose"] (YOLO thread)           │
  ▼                                       ▼
GateEstimateSmoother ──► GateAssist ──► BlueLinePilot.tick()  ──► Controller
                                          or                        (attitude_quat)
                                        GPPilot ← TrackVirtualGate
```

Everything is coupled through one plain dict, `shared_data`. The detector never
imports the pilot and the pilot never touches OpenCV.

### 2.1 The `blue_line` contract

`VisionRX._vision_loop` runs the tracker on every decoded frame and publishes
`estimate_to_dict(est)`:

```python
{"found": bool, "cx_norm": float, "cy_norm": float, "heading_err": float,
 "width_norm": float, "left_found": bool, "right_found": bool,
 "conf": float, "single_rail": bool, "frame_id": int,
 "points": [(u, v), ...]}          # corridor centres, full-res px, near→far
```

`estimate_from_dict` reverses it — used by `display.pick` to composite the
corridor HUD on top of the YOLO overlay.

### 2.2 Rate mismatch — the fact that shaped most of the design

The control loop runs at `GP_CONTROL_HZ = 60` nominal but was **log-measured at
~32.6 Hz**, against a camera delivering **~12.5 Hz** of usable frames (YOLO
inference on CPU costs ~80 ms). That is ~2.6 control ticks per new camera frame,
worst case 64. Three separate mechanisms exist purely because of it:

- **Per-frame differencing.** The `d(cx)/dt` damping term differences on
  `frame_id`, not on control ticks. Dividing a frame-to-frame step by the
  control period overstated the rate ~2.6×, and the repeat ticks in between fed
  a fabricated `0` into the EMA.
- **Staleness guard.** `data["blue_line"]` keeps `found=True` when `VisionRX`
  stalls, so nothing marked the estimate lost — logs showed 64 control ticks
  (~2 s) of steering on one frozen frame, gates included.
  `BL_VISION_STALE_S = 0.4` (~5 missed frames) now invalidates it.
- **Frame-counted re-acquisition.** On `blueline-2` the post-collision exit
  counts *distinct camera frames*, never ticks — tick counting scores one frame
  two or three times, so "stable for N" would mean nothing.

### 2.3 Sign conventions

The sim ignores `SET_POSITION_TARGET` velocity and does no auto-levelling, so
the pilot commands **attitude in error space** and self-levels. Both blue-line
consumers use the AndurilGP wire encoding:

```
roll_cmd  = (desired_roll - roll_meas)  * KR
pitch_cmd = (pitch_des    - pitch_meas) * KP
yaw_cmd   =  yaw_err                    * KY
```

Absolute RPY setpoints do not turn the live plant; error-space commands do.
`KR` and `KY` are **negative** — the sim's roll and yaw command signs are
inverted relative to odometry, and its gyros are inverted too (commanding a
right yaw yields a negative `gz`). That is why the yaw damping term is
`+K_YAW_D_GZ * gz_raw`: with inverted gyros, adding raw `gz` opposes the actual
rotation.

---

## 3. The detector, in detail

Current implementation: `simulator/blue_line_vision.py` on
`ks_vision+blueline`, built on the mask in `simulator/track_line.py`. Detection
runs at `_DOWNSCALE = 2`; every output is full-res or normalised.

### 3.1 The mask — `track_line.track_mask()`

Ranges were **measured on a 15-frame survey of real annotated flight video**
(10 with track, 5 without, 640×360), not guessed.

| Component | Range (OpenCV HSV, H 0–179) | Notes |
|---|---|---|
| Ribbon body | `H 80–106, S ≥ 90, V ≥ 110` | 0 false-positive px on all 5 no-track frames; 461–21094 px on all 10 with-track frames |
| White-hot core | `S ≤ 95, V ≥ 220`, any hue | **Must** be adjacency-gated to within ~5 px of the cyan mask — standalone it fires on ceiling lights and gate checker patterns (2300+ px on trackless frames) |

Then, in order:

1. **`H` is capped at 106**, not 120+, because YOLO overlay boxes and labels
   (pure blue, H≈120) smear down into H 105–116 under video compression. The
   v1 detector's `H 85–140` band let **blue sky (H≈113) into the mask
   wholesale**.
2. **Timer text is masked out** explicitly — green sim timer lives in rows 0–34,
   cols 0–130 at full res.
3. **`MORPH_CLOSE` only, never `MORPH_OPEN`.** Close bridges the gaps that
   magenta wash near red gate lights punches in the ribbon. An open would kill
   the YOLO keypoint dots too — but it also erases the 1–2 px thin line the
   ribbon becomes when far ahead. The v1 detector's 5×5 open **erased the far
   ribbon entirely** (830 px → 0), which is why heading read 0 in 75–88 % of
   logged frames.
4. **Connected-component filter instead**: drop components that are both small
   (< 60 px full-res) *and* non-elongated (bbox aspect < 3:1). Small and round =
   YOLO keypoint dot; small and elongated = a genuine far sliver of ribbon.

`blue_line_vision.cyan_mask()` is a thin wrapper so the two detectors share one
definition of "cyan".

### 3.2 Band slicing

Rows below the top 8 % (`_BAND_TOP_FRAC`, skipping the timer strip) are split
into `_N_BANDS = 12` horizontal bands, walked **near-first** (bottom → top).
Twelve is fine enough that a near-horizontal ribbon at a sharp turn still spans
several bands.

### 3.3 Per-band run extraction — `_band_runs()`

This is the core of the v3 rewrite. Within one band:

1. `counts = count_nonzero(band, axis=0)` — a column-occupancy profile.
2. Reject the band if total px < `_MIN_BAND_PX_FULLRES` (24).
3. **Glow-flood reject**: span > 60 % of width **AND** fill > 20 % of band area.
   Wide-*and*-dense is a bloom; wide-but-**thin** is the real ribbon seen
   side-on when banked, and is kept. Span-only rejection lost 4 of the 10
   survey frames.
4. Split the occupied columns into contiguous **runs**, bridging gaps ≤
   `_RUN_MERGE_GAP_FULLRES` (8 px). Drop runs thinner than
   `_MIN_RUN_PX_FULLRES` (12 px) as speckle.
5. Merge adjacent runs closer than `_MIN_RAIL_SEP_FULLRES` (30 px) — two runs
   that close together are one broken rail, not two rails.
6. Return the band's row centroid and the merged runs, **ordered left to right**.

The runs are the rails. **Left and right come from the corridor's own geometry**,
not from which half of the image a blob happens to sit in.

### 3.4 Rail pairing and fits

Bands that resolved **both** rails (≥ 2 runs, separated by ≥ `sep`) are the
corridor's skeleton. Three least-squares fits are taken over those bands only:

- `fit_l(v)` — left rail x as a function of image row
- `fit_r(v)` — right rail x
- `fit_hw(v)` — corridor half-width

Then every band contributes a corridor centre:

- **Both rails present** → centre = midpoint of the outer runs.
- **One rail present, a fit exists** → the side comes from *whichever fitted
  rail this run continues* (`|u − fit_r|` vs `|u − fit_l|`), and the centre is
  the run displaced by `fit_hw(v)`.
- **One rail present, no fit** → fall back to the tracker's remembered
  `hint_side`, else "left of image centre means it is the left rail".

`single_rail = True` when **no** band resolved both rails; the centre is then
inferred throughout, and the pilot is told so.

### 3.5 Producing the outputs

| Output | How |
|---|---|
| `cx_norm` | Pixel-weighted average of the **nearest 3** band centres, normalised about the image centre and clipped to ±1.5. Not the single nearest band: a lone near band can flap ±0.5 between near-identical frames when it catches a fragment of one rail. |
| `cy_norm` | Pixel-weighted mean row of the bands in `[0.25, 0.65]` of frame height — the mid-frame ribbon, i.e. the elevated path, excluding near-floor paint. Falls back to all bands. |
| `heading_err` | `u = a·v + b` least-squares through the corridor centres; `heading_err = atan2(−a, 1)`. Near = large `v`, so the far end lies right of the near end when `a < 0`. Same convention as `detect_track`, so the two detectors are interchangeable. |
| `width_norm` | Mean paired half-width × 2 ÷ frame width. |
| `conf` | `clip(n_centres / 5, 0, 1) × (0.5 + 0.5 × paired_fraction)` — bands resolved, weighted by how many of them saw both rails. |

Minimum `_MIN_VALID_BANDS = 2` corridor centres, or `found = False`.

### 3.6 `BlueLineTracker` — frame-to-frame memory

The detector proper is a **pure function**. All memory lives in the tracker:

- **Corridor half-width EMA** (`α = 0.3`), fed only by frames that actually
  paired rails, passed back in as `hint_half_width_px`. A single-rail frame
  therefore keeps a continuous centre estimate instead of snapping to the
  `_DEFAULT_HALF_WIDTH_FRAC = 0.22` guess.
- **Rail side memory**, set when exactly one of `left_found` / `right_found` is
  true, cleared on a clean two-rail lock.
- **Heading rate limit**, `_MAX_HDG_STEP_DEG = 25°` per frame. 5 % of logged
  frames exceeded a 20° step, peaking at 85°. A clamped frame is re-emitted
  with `conf × 0.5` — the frame is *suspect*, not unusable.
- Loss of lock clears the heading memory so a re-acquire cannot fire a step.

### 3.7 The four defects the rewrite fixed

The v1/v2 detectors split each ROI at a fixed `w // 2` and took per-side blob
centroids. Every defect below was **reproduced on synthetic frames** before the
rewrite, and each has a regression test.

| Defect | Symptom | Fix |
|---|---|---|
| Corridor wholly inside one image half | Both rails scored as one side; the "only one side" fallback (`mid = rail ± 0.25·w`) pulled the answer **past centre** — true `cx = +0.44` came out as `−0.06`. Sign-inverted exactly when the drone is off the corridor and the correction matters most. | Per-band runs; sides from corridor geometry |
| Hard bend puts both far rails in one half | The far/near mid-x comparison inverted too: a true +36° bend read as −1°. | Slope fitted through corridor centres, not a two-ROI difference |
| `MORPH_OPEN 5×5` erased the far ribbon (830 px → 0) | Heading was exactly 0 in 75–88 % of logged frames — the far band was simply empty. | Close-only morphology + CC filter |
| Cyan band reached `H = 140` | Blue sky (H≈113) entered the mask wholesale. | `H` capped at 106 |

### 3.8 Detector constants

| Constant | Value | Rationale |
|---|---|---|
| `_DOWNSCALE` | 2 | detection resolution; outputs full-res |
| `_N_BANDS` | 12 | near-horizontal ribbon still spans several |
| `_BAND_TOP_FRAC` | 0.08 | skip the timer strip |
| `_MIN_TOTAL_PX_FULLRES` | 120 | whole-frame floor: below this there is no line |
| `_MIN_BAND_PX_FULLRES` | 24 | per-band floor |
| `_MIN_RUN_PX_FULLRES` | 12 | thinner column run is speckle |
| `_RUN_MERGE_GAP_FULLRES` | 8 | bridge gaps *inside* one rail |
| `_MIN_RAIL_SEP_FULLRES` | 30 | closer than this = one broken rail |
| `_MAX_BAND_SPAN_FRAC` / `_FLOOD_FILL_FRAC` | 0.60 / 0.20 | glow-flood reject (both must hold) |
| `_MIN_VALID_BANDS` | 2 | minimum for a lock |
| `_CONF_FULL_BANDS` | 5.0 | this many bands already means a solid lock |
| `_ALT_BAND_LO/HI` | 0.25 / 0.65 | altitude cue reads mid-frame ribbon |
| `_DEFAULT_HALF_WIDTH_FRAC` | 0.22 | last-resort corridor half-width |
| `_MAX_HDG_STEP_DEG` | 25.0 | per-frame heading jump beyond this is not physical |
| `_HALF_WIDTH_EMA` | 0.3 | width memory smoothing |

---

## 4. The control law, in detail

`simulator/blue_line_pilot.py`. `compute_blueline_guidance()` is a pure
function — vision dict + IMU scalars in, four wire commands + a debug dict out.
`BlueLinePilot` wraps it with state, race gating, logging and the gate assist.

### 4.1 Modes

`compute_blueline_guidance` resolves exactly one mode per tick, in priority
order:

| Mode | Entered when | Behaviour |
|---|---|---|
| `commit` | a gate is right in front of us (see §4.6) | **Freeze** steering on the last good track command, cruise pitch, kill the brake |
| `track` | fresh corridor lock | the full law below |
| `gate` | line lost **but** a fresh YOLO gate in view | steer at the gate, corner speed |
| `hold` | line lost < `LOSS_HOLD_S` (0.2 s) | yaw `±12°` toward the last-known turn side |
| `creep` | lost < `SEARCH_AFTER_S` (0.5 s) | same yaw, blind speed |
| `search` | lost longer | yaw `±35°` sweep, blind speed |

The blind turn side comes from `gate_hint_sign` (a gate seen within
`GATE_HINT_MEMORY_S = 3 s` — its bearing sign beats a heuristic), else
`_turn_hint_sign(last_cx, last_hdg)`, which prefers heading when
`|hdg°| ≥ 30·|cx|` and falls back to `cx`.

### 4.2 Lateral: roll and yaw

```
desired_roll = clip( gain_scale·(K_ROLL_CX·cx + K_ROLL_HDG·hdg°)
                     + clip(K_ROLL_DCX·dcx, ±8°),          ±26° )
yaw_err      = clip( gain_scale·(K_YAW_CX·cx  + K_YAW_HDG·hdg°),  ±40° )
yaw_err     += K_YAW_D_GZ · gz_raw                       # damping, every mode
```

- `K_ROLL_CX = 36`, `K_YAW_CX = 40`. Raised twice after log analysis: `cx`
  clips at ~±0.5 and the measured sim response is ≈2.2 °/s of yaw per degree of
  wire command, so the 40° clamp targets ~85 °/s peak turn rate.
- `K_ROLL_HDG = 0.35`, `K_YAW_HDG = 0.65`. **Re-tuned for the v3 detector.**
  The old values (`K_YAW_HDG = 1.2`) were set against a heading signal that was
  0 in 75–88 % of frames and sign-inverted at hard corners — they only ever
  fired on noise. Heading now resolves every frame (synthetic bends: gentle
  +18.7°, hard +36°), and at 1.2 a routine hard corner alone saturated the 40°
  yaw clamp. Authority is split instead: **heading anticipates the bend, `cx`
  corrects the drift that remains.**
- `gain_scale = 0.5 + 0.5·conf` (`CONF_GAIN_FLOOR = 0.5`). Detector confidence
  *fades* the vision gains rather than cutting them — a single-rail lock is
  still real guidance.
- **Heading is gated on band count, not `conf`.** Heading is a line fit, so what
  makes it trustworthy is how many bands it was fitted through; `conf` also
  drops when one rail leaves the frame, and a partly-visible corridor still
  yields a good slope. Fewer than `HDG_MIN_BANDS = 3` points → heading is
  forced to 0 (and logged as 0, so the log shows what was actually steered on).
- `K_ROLL_DCX = 6.0`, EMA α 0.35, hard-capped at ±8°. The baseline had yaw-rate
  damping but **none on the roll-induced translation** the `gz` gyro cannot see
  — log-verified `cx` ±0.4 weave at ~0.68 sign-flips/s. The cap means it can
  only damp the weave, never dominate the bank.
- `K_YAW_D_GZ = 0.12` — see §2.3 for why the sign is `+`.

### 4.3 Longitudinal: `turn_mag` and the speed loop

```
turn_mag_raw = clip(|hdg°|/35 + |cx|/1.1, 0, 1)
turn_mag     = symmetric_limiter(turn_mag_raw)      # rise ≤2.5/s, decay ≤0.8/s
v_target     = CRUISE + turn_mag·(CORNER − CRUISE)
```

Both scale factors moved with the v3 detector:

- `_TURN_HDG_SCALE_DEG` 20 → **35**. A merely *gentle* bend now reads ~19°, and
  at the old scale that alone latched `turn_mag ≈ 0.93` — a full brake to corner
  speed on every curve.
- `_TURN_CX_SCALE` 0.7 → **1.1**. With `hdg` dead most of the flight,
  `turn_mag` was driven almost entirely by `cx` **weave noise** (> 0.7 for 77 %
  of track time), permanently braking *and* — via the thrust attenuation —
  starving the gate-2 climb.

The **symmetric limiter** is two separate fixes:

- `TURN_MAG_DECAY_PER_S = 0.8` — latch the peak and decay it, so the brake holds
  through the whole corner and across YOLO gaps. Raw `turn_mag` whipsawed with
  per-frame noise (log-verified `v_target` flickering 2.22 ↔ 3.33 mid-turn).
- `TURN_MAG_RISE_PER_S = 2.5` — added later, because the decay latch made a
  *single* bad frame expensive: `cx = +0.001, hdg = +55.9°` slammed a full brake
  that then held for 1.25 s, straight through a gate. At 2.5/s a real corner
  still brakes fully in 0.4 s; a one-frame artefact cannot.

Speed itself:

| Path | Condition | Law |
|---|---|---|
| Closed loop | `vX` is real | `pitch_des = −2° − K_SPEED_P·(v_target−vX) − K_SPEED_D·â − K_PITCH_CY·cy_err·0.5`, with `K_SPEED_P = 2.5`, `K_SPEED_D = 0.6` and `â` an EMA'd (0.3/0.7) one-tick acceleration |
| Open loop | `vX` is `nan` (VQ2 telemetry block — log-verified 748/748 rows in one session) | cruise/creep pitch, `cy` correction, plus `turn_mag · OPEN_BRAKE_DEG (3.5°)`. At full corner that is `−2 + 3.5 = +1.5°`, i.e. an actual nose-up brake; the earlier `+1.5°·turn_mag` left the nose still **down**. |

Hard cap: above `MAX_SPEED_MPS·1.15` the pitch is pinned to
`PITCH_DES_MAX_DEG`; above `MAX_SPEED_MPS` it is floored at half of it. Below
`UNTRUSTED_VX_MPS = 0.4` the estimate is not trusted to command a dive.

### 4.4 Vertical

```
cy_err  = cy − CY_TARGET                                    # CY_TARGET = 0.05
pitch  −= K_PITCH_CY · cy_err · (1 − 0.5·turn_mag)          # open-loop path
thrust  = clip(hover − K_THRUST_CY·cy_err·(1 − 0.3·turn_mag), 0.15, 0.55)
```

The thrust attenuation was cut 0.7 → **0.3**: with `turn_mag` pinned by weave
noise, the climb correction was running at ~35 % authority, and gate 2 is the
steepest-climb leg (log: thrust never left 0.30 while the drone sat 0.5 m low).
On straights `turn_mag → 0`, so level legs are unaffected and `K_THRUST_CY`
never moved.

### 4.5 Command slew

`BL_CMD_SLEW_DEG_S = 360` versus GP's 90. GP's rate is 1.5°/tick at a nominal
60 Hz, which halves again at the real ~32 Hz — log-verified, the corner appears
as a one-frame `yaw_err` step and the slewed command reached only **56 %** of it
before the line left the FOV. 360 °/s builds the full 40° command in ~0.2 s; the
`gz` damping keeps that agility from becoming oscillation.

### 4.6 YOLO gate assist — bounded, secondary

`GateAssist` wraps `GateEstimateSmoother` on a `{pose, frame}` view of the
shared dict, so the smoother's Anduril/HSV fallbacks **cannot** fire — YOLO-only
by construction. Staleness, nearest-ahead pick, near-gate suppression, EMA and
target-identity lock all come from the smoother. On top of that:

| Guard | Value | Why |
|---|---|---|
| `GATE_ASSIST_MIN_RANGE_M` | 2.5 | the smoother blanks below 2.0; margin against bearing blow-up |
| `GATE_ASSIST_MAX_RANGE_M` | 35.0 | gates are 24–29 m apart; beyond that is PnP noise |
| `GATE_ASSIST_MAX_BEARING_DEG` | 45.0 | half-FOV; beyond is a clip artifact |
| `GATE_ASSIST_FRAME_TIMEOUT_S` | 0.5 | wall-clock: a frozen camera gives no assist, which frame-id staleness alone cannot detect |

The assist does three distinct things:

1. **Slow down (direction-neutral).** Any gate at `|bearing| ≥ 8°` raises
   `turn_mag` by `|bearing|/20`. Braking early is safe even for a phantom
   detection.
2. **Steering bias (agreement-gated, strict).** Adds at most ±10° of `yaw_err`
   and ±9° of bank — a quarter of the global clamps, so it can never fight a
   good line lock. It engages only when the line **confirms** the same-side
   bend (`bearing·hdg > 0` and `|hdg| ≥ 2°`), or when the detection is
   plausibly the on-path next gate (`|bearing| ≤ 20°` **and** range ≥ 14 m).
   The strictness is load-bearing: log-verified phantom gates at +16…+42° and
   9–31 m were biasing the drone **right at a left corner**.
3. **Gate commit.** See below.

### 4.7 Gate commit — the punch-through

At the gate mouth *everything blinds at once*: `GateAssist` floors at 2.5 m, the
YOLO tracker goes blind under `YOLO_NEAR_BX_M = 2.0` and holds a 10-frame
cooldown, and the corridor rails sweep out of the FOV. The pilot answered that
with hold → creep → search, i.e. a ±35° yaw sweep at the exact moment it should
be flying straight — log-verified hesitation.

Commit inverts the interpretation: **the blanking *is* the gate arriving**, not
a detection failure.

- **Arm** while the gate is still resolvable: range ≤ 4.0 m and `|bearing|` ≤
  12°. A wild bearing is not a pass.
- **Latch** the moment the estimate blanks, provided the arming sighting is
  younger than `GATE_COMMIT_READY_MEMORY_S = 0.6 s`.
- **Hold** for `GATE_COMMIT_WINDOW_S = 0.6 s` (≈ the 10-frame cooldown at
  12.5 Hz): freeze `desired_roll`/`yaw_err` on the last good *track* command,
  hold cruise speed, zero the latched brake, and forbid the search ladder.
- **Disarm** on `gates_passed` increasing or on window expiry; the gate just
  threaded can never arm another commit.

### 4.8 Race gating and abort

The pilot holds **zero thrust** until a race that started *after pilot startup*
exists — mirroring `GPPilot`'s anchor so a stale already-running race never
flies. It does **not** wait out the sim's 3-2-1: the user wants wheels-up
immediately after each manual *Restart Race*, and while the sim keeps the drone
frozen the physics-live gate holds naturally.

- **Anchor**: `race_start_boot_time_ms ≥ wait_anchor_ms`, with
  `CLOCK_RESET_SLACK_MS = 500` to chase a rewound sim clock.
- **Physics-live gate**: a stale race can idle the sim physics — the IMU keeps
  streaming but its sensor clock freezes and commands do nothing (log-verified:
  a 6.7 s attempt with `gz` pinned at 0). The IMU `time_us` must have advanced
  since the previous tick.
- **Abort back to hold** on: sim clock rewind, a new race start, race finish, or
  `armed == False` persisting for `DISARM_PERSIST_S = 1.0` (the 1 Hz heartbeat
  blips otherwise).
- On each attempt start the ESKF is reset — the sim teleports the drone on
  restart, which invalidates its gyro-integrated attitude and velocity history.

### 4.9 Telemetry

One CSV per attempt, `rl/data/bl_log_<YYYYmmdd>_<HHMMSS>.csv`, flushed at 1 Hz,
27 columns:

```
t mode cx cy hdg_deg conf width single_rail dcx turn_mag gate_brg gate_rng
vX v_target pitch_des des_roll yaw_err cmd_roll cmd_pitch cmd_yaw thrust
att_roll att_pitch gz_dps lost_s n_passed
```

Logging failure never grounds the pilot — every writer path catches `OSError`
and closes the log. Practically every tuning decision quoted in this document
came from these files.

### 4.10 Pilot constants

| Constant | Value | | Constant | Value |
|---|---|---|---|---|
| `K_ROLL_CX` | 36.0 | | `MAX_BANK_DEG` | 26.0 |
| `K_YAW_CX` | 40.0 | | `YAW_ERR_MAX_DEG` | 40.0 |
| `K_ROLL_HDG` | 0.35 | | `PITCH_DES_MIN/MAX_DEG` | −4.0 / 8.0 |
| `K_YAW_HDG` | 0.65 | | `PITCH_WIRE_MAX_DEG` | 18.0 |
| `K_ROLL_DCX` | 6.0 (cap ±8°) | | `THRUST_MIN/MAX` | 0.15 / 0.55 |
| `K_YAW_D_GZ` | 0.12 | | `CY_TARGET` | 0.05 |
| `K_PITCH_CY` | 8.0 | | `CRUISE_PITCH_DEG` | −2.0 |
| `K_THRUST_CY` | 0.10 | | `CREEP_PITCH_DEG` | −0.5 |
| `K_SPEED_P` / `K_SPEED_D` | 2.5 / 0.6 | | `SEARCH_YAW_ERR_DEG` | 35.0 |
| `MAX_SPEED_MPS` | 25 km/h | | `HOLD_YAW_ERR_DEG` | 12.0 |
| `CRUISE_SPEED_MPS` | 12 km/h (`BL_CRUISE_KMH`) | | `LOSS_HOLD_S` | 0.2 |
| `CORNER_SPEED_MPS` | 8 km/h | | `SEARCH_AFTER_S` | 0.5 |
| `BLIND_SPEED_MPS` | 6 km/h | | `BL_VISION_STALE_S` | 0.4 |
| `CONF_GAIN_FLOOR` | 0.5 | | `HDG_MIN_BANDS` | 3 |
| `BL_CMD_SLEW_DEG_S` | 360.0 | | `HOVER_THRUST` | 0.264 |

Cruise defaults to 12 km/h, not 18: 18 carried too much speed into the gate-2
left-hander to brake in time. 25 km/h is a hard user-imposed cap.

---

## 5. Branch variants

### 5.1 `blueline` — v1, centroid

`_CYAN_LOWER = [85, 70, 50]`, `_CYAN_UPPER = [140, 255, 255]`, `MORPH_OPEN` +
`MORPH_CLOSE` at 5×5. The frame is split at `w // 2`; each half's blob centroid
is one rail. A near ROI (bottom 40 %) gives `cx`/`cy`; a far ROI (top 55 %)
gives heading from the difference of the two mid-x values. One rail only →
`mid = rail ± 0.25·w`.

Every §3.7 defect lives here. It is nonetheless the branch that first flew the
corridor end-to-end, and it carries the parts that outlived the detector: the
race gating, the wire encoding, the gate assist, the flight logs, and the whole
`compute_blueline_guidance` structure.

It also carries the **RL layer**:

- `rl/blueline_env.py` — Gymnasium env, 13-D observation
  (`found, cx, cy, heading_err, width, left_found, right_found, v[3], ω[3]`),
  4-D action in the same attitude-quat wireform, 50 Hz decisions over a
  lightweight internal physics + synthetic corridor projection, with a 3 → 6 →
  10 → 15 gate curriculum. `make est-selftest`.
- `rl/train_blueline_ppo.py` — PPO, 1 M steps, `[64, 64, 64]`, writes
  `rl/data/blueline_ppo.zip` and `rl/data/blueline_best/`. `make train-blueline-ppo`.

The env imports its clamps directly from `blue_line_pilot`, so a learned policy
lands in exactly the same action space as the hand-written one.

### 5.2 `blueline-2` — inner-edge scanline, A/B probe, collision reverse

**Inner-edge estimator** (`BL_INNER_EDGE`, default on). Per-row scanlines track
each ribbon's **inner edge** from the bottom row upward; corridor mid is the
midpoint of the inner edges, least-squares fitted over the scanned rows.

The argument for inner edges is geometric: a ribbon's *centroid* moves with its
apparent **thickness**, so unequal bloom (one ribbon nearer or brighter) shifts
the corridor mid even when the drone is centred. An inner edge is a cyan→road
**hue** transition, unlike the outer edge's exposure-sensitive cyan→black glow
fade.

Mechanics: 28 sampled rows up to `_SCAN_TOP_FRAC = 0.35`; per row, keep only
runs whose width is in `[0.005, 0.15]·w` — a ribbon is measurably thin (9–13 px
at 640 wide even at the bottom of frame) while the lit track surface between the
ribbons also passes the cyan mask on ~24 % of live frames and shows up as one
276–395 px blob. **Width is what separates them.** The scan seeds on the *gap*
containing the image centre (not a `w//2` split, so it survives both ribbons
being in one half), refuses to seed on gaps narrower than `0.10·w`, predicts
each row's edges from the row below, and aborts after 4 consecutive misses.

Crucially it ships with a **three-state switch**: `off` / `shadow` / `fly`.
In `shadow` the new estimator is computed and logged but the **centroid is
still flown**, so a live run scores the candidate without handing control to an
estimator that has not earned it.

**The probe** — `simulator/blue_line_probe.py`, `make bl-probe`. Reads only the
camera stream, never MAVLink; safe to run during a live race. With the drone
parked the true corridor offset is constant, so every frame-to-frame change in
`cx` **is** measurement noise — the probe reports that jitter for both
estimators plus the **fallback rate** (how often the scan silently handed back
the centroid). It saves raw frames to `runs/bl_probe/raw` so `--replay` re-runs
tuning on the exact same pixels.

**Post-collision reverse.** `BlueLinePilot` previously ignored `data["collision"]`
entirely and kept driving into whatever it hit. `77ceaec` adds a `reversing`
phase mirroring `gp_pilot`'s `BACKOFF`, and the write-up of *why it is not a
straight port* is the most instructive part:

- Ground contact spams `COLLISION` "hundreds of times per second while the drone
  is on the pad". Nothing ever popped the key, so a pad hit would sit in the
  dict looking fresh forever and reverse the drone off the launch pad on tick 1.
  Fixed by draining the key every tick in **both** phases — including `wait`,
  which is what guarantees it is empty at GO — plus a 3 s post-GO grace, a
  floor-clearance veto, a cooldown and a per-attempt cap of 3.
- GP's backoff integrates reverse distance from `−vX`. `vX` is `nan` in every
  VQ2-blocked flight, so that accumulates 0.0 forever and a distance-primary
  exit would never fire. The exit is therefore **time-primary** (2.5 s, shorter
  than GP's 4.0 because we reverse blind with no rear camera); distance is a
  bonus that engages only when `vX` is real. The reverse itself is open-loop:
  arrest hard at +4° nose-up, then drift back at +1.5°.
- Re-acquisition is deliberately strict: **4 distinct camera frames** spanning
  ≥ 0.25 s with no gap, consistent bearing and range, `|bearing| ≤ 25°`,
  range 3–25 m, conf ≥ 0.60, reproj ≤ 6 px, and `n_visible ≥ 3` — which rejects
  `gate_pnp`'s edge-pair fallback, whose "perfect" hardcoded `reproj_px = 0.0`
  means nothing.
- The severity gate `REVERSE_MIN_DELTA` **ships inert (0.0)**: MAVLink defines
  `horizontal_minimum_delta` as a *distance* (small = severe) while vendor
  `train_controller` treats the same field as an *impulse* (large = severe). A
  threshold with the wrong sign either never fires or always fires, so the
  console line and the `rev_n` CSV column exist to learn the real distribution
  first.

This only helps glancing hits — a hard crash disarms the sim and the existing
abort wins, which is why the abort check stays first in `tick()`.

### 5.3 `ks_vision+blueline` — per-band rails, fused into GP

The v3 detector of §3, plus the fusion described in §1.1(b).
`TrackVirtualGate.synth()` picks per camera frame, in order:

1. `data["blue_line"]` — the dual-cyan corridor, if `found`, matching
   `frame_id`, and at least one rail seen. `strength = 1.0` when both rails
   resolved, else `0.5`.
2. `detect_track(frame["img"])` — the single-ribbon band centroid detector, as
   fallback.

Either becomes the same virtual gate:

```python
body_x_m = TRACK_LOOKAHEAD_M                    # 4.0
body_y_m = offset·TRACK_LAT_GAIN                # 3.5 m per unit image offset
         + tan(clip(angle, ±0.6)) · lookahead · TRACK_ANGLE_GAIN   # 0.8
body_z_m = 0.0
```

**The heading term is gated on strength.** Offset is the reliable signal; the
up-tilted camera sees little of the floor ribbon, so most `detect_track`
detections are 2–4 bands where `angle` saturates near ±π/2 and points the
**wrong way** — live-observed `a = +1.09` at `strength = 0.33` banked the drone
+14° right into a *left* curve. `TRACK_ANGLE_MIN_STRENGTH = 0.5` (≥ 6 bands) is
the floor for trusting it; the blue-line path uses `both rails resolved` as the
equivalent test.

The corridor also feeds `_course_direction_cue()`, which primes the post-pass
SEARCH direction from `sign(cx_norm)` when `|cx| > 0.15`. It is *never* allowed
to override gate steering — only to choose which way to look when blind.

`BlueLinePilot` itself remains in the tree on this branch and is fully tested,
but see §6.3.

---

## 6. Tests, knobs, and gaps

### 6.1 Tests

| File | Locks down |
|---|---|
| `tests/test_blue_line_vision.py` (19) | presence/absence, offset and heading **signs**, each of the four §3.7 defects as its own case (corridor wholly in one half, hard bends, thin far ribbon survival, blue-sky exclusion), single-rail centre offset, tracker width memory and reset, heading rate limit, dict round-trip |
| `tests/test_blue_line_pilot.py` (60+) — *not on `classical+blueline`* | wire-sign direction for every axis, corner braking and the rise cap, 25 km/h cap, the search ladder, gate-bias engage/skip cases, confidence fade, band-gated heading, `dcx` rate damping and its cap, `gz` damping sign, gate commit arm/hold/expire/disarm, vision staleness, and the whole race-gating state machine |
| `tests/test_gp_pilot.py::TrackVirtualGateBluelineTests` (4) | corridor preferred over `detect_track`, fallback to `detect_track` when absent, and `_course_direction_cue` priority (blueline → gate → track) |

All are in `make test` on the branch that carries them. Note that **CI runs only
ruff and doc checks** — unit tests are not gated, so a committed baseline can
carry failing tests.

`uv run python -m simulator.track_line --selftest` runs the detector's own
synthetic checks (straight / right-offset / curve-R / curve-L / ceiling-lights /
red-glow) and, if the survey frames are on disk, reports detection and
false-positive rates with per-frame timing.

### 6.2 Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `BL_CRUISE_KMH` | `12` | cruise speed target |
| `BL_GATE_ASSIST` | `1` | `0` keeps YOLO running but ignores it entirely |
| `BL_DEBUG` | off | one console line every 30 ticks |
| `BL_DISPLAY` | `1` | live vision window |
| `SKIP_YOLO` | — | disables the detector outright |
| `BL_INNER_EDGE` | `1` | `blueline-2`: `0` / `shadow` / on |
| `BL_PROBE_SECONDS`, `BL_PROBE_WAIT`, `BL_PROBE_SAVE_EVERY` | 20 / 600 / 30 | `blueline-2` probe |
| `GP_TRACK_LOOKAHEAD`, `GP_TRACK_LATGAIN`, `GP_TRACK_ANGGAIN` | 4.0 / 3.5 / 0.8 | `ks_vision+blueline` virtual-gate geometry |

### 6.3 Known gaps

- **`AUTO_PILOT=blueline` is not dispatched on `ks_vision+blueline`.**
  `Controller._make_pilot()` on that branch handles only `gp` and `rl`, so
  `auto_blueline.py` — which is still present and still sets the variable —
  falls through to the default `Pilot`. Standalone blue-line flight there needs
  the two-line dispatch branch carried over from `blueline`.
- **`track_line._SURVEY_DIR` is an absolute path on another machine**
  (`C:/Users/kunal/...`). The self-test degrades gracefully — it just skips the
  real-frame section — but nobody else can run that half.
- **`REVERSE_MIN_DELTA` ships inert** on `blueline-2` by design; the field's
  semantics are contradictory between MAVLink and the vendor controller.
- **The inner-edge A/B never concluded.** `blueline-2` still carries the
  scaffolding fields (`cx_centroid`, `cx_edge`, `heading_*`, `edge_rows`) with a
  `TODO(inner-edge-ab)` marking them for deletion once it does — and the v3
  per-band detector on `ks_vision+blueline` was developed independently of it,
  so the two candidate replacements for the centroid estimator were never scored
  against each other.
- **The detector's HSV ranges are tuned to one course's lighting.** `H ≤ 106`
  in particular is a compression artifact defence, not a physical property of
  the ribbon.
