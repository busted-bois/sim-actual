# Gate Perception: the Classical + YOLO Vision Stack

Low-level reference for how a single monocular FPV frame becomes a gate pose in
the drone's body frame. Written to be readable without the repo open.

| Field | Value |
|-------|-------|
| **Written against** | `ks_vision+blueline` @ `67d866f` + uncommitted working tree |
| **Also covers** | `main` @ `0a60f92`, `ks/YOLO-implementation` @ `2a745b0`, `test-classical` @ `47f7222`, `yahya-gate-detection` @ `894cd0b`, `test-gatenet` @ `3f9abb0` |
| **On `classical+blueline`** | Applies as written, **minus** the image-plane center-lock: `gp_vision.py` here is the `rl_failed101` version, so `best_pose_gate` keeps its `BEARING_COST_M_PER_RAD` scoring and publishes `u_px`/`v_px` as `None`. |
| **Core files** | `simulator/gate_detector.py`, `anduril_gate_detect.py`, `gate_pose.py`, `gate_corners_cv.py`, `gate_pnp.py`, `gp_vision.py` |

---

## Contents

1. [Problem and sensor model](#1-problem-and-sensor-model)
2. [Four detectors on one frame](#2-four-detectors-on-one-frame)
3. [Geometry: PnP and the hybrid pinhole reconstruction](#3-geometry-pnp-and-the-hybrid-pinhole-reconstruction)
4. [Fusion, tracking and gating](#4-fusion-tracking-and-gating)
5. [The Try-1 / Try-2 / Try-3 ladder across branches](#5-the-try-1--try-2--try-3-ladder-across-branches)
6. [Training and data pipeline](#6-training-and-data-pipeline)
7. [Failure modes and the fixes they forced](#7-failure-modes-and-the-fixes-they-forced)
8. [Verification, knobs, and open gaps](#8-verification-knobs-and-open-gaps)

---

## 1. Problem and sensor model

**Task.** Fly a drone through a sequence of square gates using **only** the
onboard forward camera. Under the competition's VQ2 configuration the simulator
withholds the broadcast gate map (`GATE_INFO`), and in most sessions also
withholds `ODOMETRY` / `ATTITUDE` / `LOCAL_POSITION_NED`. There is no GPS and no
absolute coordinate frame. Perception must therefore produce a **relative** gate
pose from pixels alone.

**Camera.** Per the competition spec (§3.8):

| Parameter | Value |
|---|---|
| Resolution | 640 × 360 |
| Focal length | `fx = fy = 320 px` |
| Principal point | `cx = 320`, `cy = 180` |
| Distortion | none (pinhole) |
| Mounting | **20° up-tilt** about the body y-axis |
| Frame transport | UDP :5600, ~30 Hz nominal |

The 20° up-tilt is not cosmetic — it is responsible for a whole class of failure
modes in §7, because it pushes the *bottom* of a gate out of frame on a centred
approach well before the drone reaches it.

**Gate geometry.** A square frame, **2.7 m outer**, **1.5 m inner opening**. The
opening is what the drone must fly through, so the opening — not the frame — is
the thing perception must localise.

**Required output.** For each frame, at most one gate estimate:

```python
{"body_x_m", "body_y_m", "body_z_m",   # gate opening centre, body FRD
 "normal_body",                        # gate-plane unit normal, pointing back at us
 "range_m", "yaw_bearing", "pitch_bearing",
 "reproj_px", "n_visible", "method",   # quality metadata
 "source", "reliable", "frame_id"}
```

Body frame is **FRD**: x forward, y right, z down. The fly-through axis the
guidance wants is `−normal_body`.

---

## 2. Four detectors on one frame

The stack runs several independent detectors and picks between them per frame.
They are ordered here by increasing sophistication, not by preference — the
preference order is a per-branch decision (§5).

### 2.1 Hex-colour HSV blob — `gate_detector.py`

The oldest and cheapest path. Bounds are derived at import time from a single
configured hex colour rather than hand-tuned:

```
GATE_HEX_COLOR = "#F3390F"      # gate orange
HSV_TOLERANCE  = 60             # widened for VQ2 R2's scanned environments
lower/upper    = hsv(hex) ∓ tolerance, clamped to OpenCV ranges
```

Pipeline: BGR→HSV → `inRange` **with explicit hue-wraparound handling** (orange
sits near H=0, so `lower[0] > upper[0]` means two ranges OR'd together) →
`MORPH_OPEN` then `MORPH_CLOSE`, 5×5 ellipse, 2 iterations → `findContours`
(`RETR_EXTERNAL`).

Contours are filtered on area (≥ 250 px) and aspect ratio (0.2–5.0), then scored:

```
score = area / (1 + |cx − w/2| + |cy − h/2|)
```

— i.e. **big and central beats big and peripheral**, a cheap proxy for "the gate
we are flying at" rather than "the gate over there".

A second `_desaturated_orange_mask` (hue ±12, S 25–140) retries when the primary
mask finds nothing, because VQ2 R2's *scanned* gate textures are markedly less
saturated than VQ1's rendered ones.

> **A real bug worth knowing about.** Those fallback bounds are shaped `(1, 3)`.
> On OpenCV 4.13 a `(1, 1, 3)`-shaped bound silently matches **every pixel**
> instead of erroring, so the mask comes back fully white and the detector
> reports a gate filling the frame. The comment in `gate_detector.py` pins the
> shape deliberately.

Output is a `GateDetection` (centroid, area, bbox) which `vision_rx` converts to
`data["gate_target"]` with normalised offsets and, via `gate_estimator.py`, a
bearing/elevation/range using a **self-calibrating focal length** (calibrated
from track ground truth when available, otherwise a pixel-ratio estimate against
the 1.5 m reference width).

### 2.2 Anduril HSV-red + quad + PnP — `anduril_gate_detect.py`

Ported byte-faithfully from the vendor reference implementation so that offline
replay matches live flight. Two red ranges (H 0–10 and H 170–180, S ≥ 120,
V ≥ 50) OR'd, 3×3 rect close-then-open, largest contour above 300 px.

What makes it more than a blob detector:

- **Edge/centre rejection.** A contour touching the left/right border *and*
  sitting more than 25 % of the frame off-centre is discarded outright — that is
  a partial gate at the frame edge, and PnP on a clipped quad solves a mirrored
  pose. A separate `reliable` flag downgrades (rather than drops) contours
  touching top/bottom while small.
- **Quad extraction.** Convex hull → `approxPolyDP` at escalating epsilons
  (0.03, 0.05, 0.08, 0.12 × perimeter) until exactly 4 vertices survive →
  `cornerSubPix` on greyscale (7×7 window, 30 iterations) for sub-pixel corners.
- **Inner opening.** A second finer mask with `RETR_CCOMP` hierarchy finds the
  *hole* contour — a child contour inside the gate's bounding box whose area is
  10–70 % of the frame contour's. Sub-pixel refined at 5×5.
- **PnP, twice.** First `SOLVEPNP_IPPE_SQUARE` on the 4 outer corners against
  the 2.7 m object square, picking the ambiguity branch with positive depth and
  lowest error. Then, if the inner quad was found, an 8-point
  `SOLVEPNP_ITERATIVE` refinement seeded from that solution
  (outer 2.7 m + inner 1.5 m corners). Both gated at 12 px reprojection.
- **Bbox EMA with a reset rule.** α = 0.5, but the filter **resets to raw**
  whenever the centre jumps > 120 px or the width ratio moves beyond 1.6× — i.e.
  it smooths measurement noise but never smooths a target change.
- **Pass-through suppression.** Above 80 000 px of red area the tracker declares
  itself *inside* the gate, publishes nothing, and holds a 10-frame cooldown
  after. Without this the pilot yanks toward the just-passed gate's edge.

### 2.3 YOLO11-pose, 8 corner keypoints — `gate_pose.py`

The learned detector. `simulator/models/gate_pose.pt` predicts, per gate
instance, a bounding box plus **8 keypoints**: inner TL, TR, BL, BR, then the
matching outer four.

Inference is far too slow to run inline in the UDP receiver — the docstring
records **100–350 ms/frame on CPU**, and end-to-end flight logs measured ~12.5 Hz
of usable frames. So `GatePoseRunner` runs it on its own thread with a
**newest-frame-wins** policy: it always grabs the latest decoded frame and drops
anything that piled up during the previous inference. Results land in
`data["pose"] = {gates, annotated, frame_id, infer_ms}`.

PnP is solved **in the detector thread**, so the control loop only ever reads a
finished pose.

### 2.4 Classical CV inner-corner refine — `gate_corners_cv.py`

Adapted from an external reference implementation and wired in as a *refinement*
of the best YOLO hit. It re-uses the shared HSV pipeline from §2.1 (so the colour
definition lives in exactly one place, including the desaturated fallback),
then:

1. `MORPH_CLOSE ×2` then `MORPH_OPEN ×1` with a **3×3** kernel — deliberately
   smaller than `gate_detector`'s 5×5, because a distant gate's opening is only
   a few pixels wide and a larger closing kernel fills it, destroying the child
   contour the whole method depends on.
2. `RETR_CCOMP` contours; take the largest **top-level** contour above a
   resolution-scaled area floor.
3. The opening is that contour's **hole** (its CCOMP child), accepted only if
   the child's area exceeds 15 % of the parent's.
4. `approxPolyDP` at 5 % of perimeter for a 4-gon, else `minAreaRect` box points.
5. Reject any quad within 2 px of the image border — a clipped opening makes PnP
   solve a mirrored pose.
6. If no hole is resolvable, shrink the outer quad about its centroid by
   `1.5/2.7` — but only when the outer quad itself is not border-touching,
   because otherwise the fallback fabricates interior corners from partial
   geometry.
7. Order the result `[TL, TR, BR, BL]` by sorting on v then u.

**Why bother** when YOLO already gives corners: a few pixels of keypoint error
blows up monocular depth at range. The contour-accurate corners fix long-range
PnP depth, where raw keypoint error dominates. The refine is gated by
`_CV_BOX_MARGIN`: the CV quad's centre must sit inside the YOLO box +20 %, so
the two detectors are never mixed across different gates.

`GATE_CV_REFINE=0` disables it.

---

## 3. Geometry: PnP and the hybrid pinhole reconstruction

`gate_pnp.py`. This module is self-contained — it owns its intrinsics, gate
geometry and camera→body rotation so `simulator/` never depends on `rl/`.

### 3.1 Object points, and a baked-in label swap

```python
_hi, _ho = 1.5/2, 2.7/2
_GATE_PTS_3D = [[ _hi,  _hi, 0], [-_hi,  _hi, 0],
                [ _hi, -_hi, 0], [-_hi, -_hi, 0],   # inner
                [ _ho,  _ho, 0], [-_ho,  _ho, 0],
                [ _ho, -_ho, 0], [-_ho, -_ho, 0]]   # outer
```

Note the sign pattern: slot 0 ("TL") maps to **+x**, i.e. the gate's top-**right**.
That is a **left/right swap deliberately baked into the object points**, because
the training-data generator wrote left labels onto right corners. Without the
swap, PnP solves a mirrored correspondence and returns the gate's back face
toward the camera. This is the single most load-bearing line in the file, and it
is the kind of thing that is invisible until the drone banks the wrong way.

A separate `_INNER_CORNERS_3D` in plain image order `[TL, TR, BR, BL]` serves the
CV-refined 4-corner path, which has no slot/label ambiguity at all.

### 3.2 Visibility gating

A keypoint counts only if `conf > 0.7` **and** it is not pinned within 1 px of
the image border — border-pinned corners are clip artifacts, not observations.
Four confident, non-edge corners are required for a PnP attempt.

### 3.3 The solve

`SOLVEPNP_IPPE` on the planar correspondence, which returns a **two-fold
ambiguous** solution set. Pick the branch with lower RMS reprojection, polish it
with `solvePnPRefineLM`, then **reject** the result outright if RMS reprojection
exceeds 10 px. A rejected solve is not silently downgraded — see §3.6.

### 3.4 The hybrid reconstruction — the central design decision

The naive use of PnP is to take `tvec` as the gate position. This stack does
**not** do that. Instead:

- **Depth (metric scale) comes from PnP** against the known gate size.
- **Lateral and vertical position are reconstructed from the gate-centre
  *pixel*** at that depth, through the pinhole model:

```python
x_cam = (u − cx)/fx · depth
y_cam = (v − cy)/fy · depth
gate_pos_cam = [x_cam, y_cam, depth]
```

**Why.** The centre pixel is *exactly where we see the gate*, so its left/right
and up/down sign is correct **by construction** — independent of any
keypoint-label ambiguity in the PnP correspondence, which otherwise mirrors the
gate to the wrong side. PnP is trusted for the one thing pixels cannot give
(scale) and distrusted for the thing pixels give perfectly (direction).

### 3.5 Which pixel is "the centre"

This is where most of the file's complexity lives, and every branch was forced
by a specific observed crash.

| Visible inner corners | Centre pixel | Reason |
|---|---|---|
| All 4 | **Intersection of the diagonals** TL–BR and TR–BL | Projectively exact under any perspective. The *centroid* is pulled 6–10 % toward the nearer (larger-projected) edge, which damps vertical convergence exactly when the drone is off-centre — a persistent low aim while riding low into a gate. The diagonal pairing is also swap-proof: the L/R label swap maps the two diagonals onto each other. |
| A balanced diagonal pair | Midpoint of the two | Geometrically the centre. |
| Anything else (unbalanced) | `_edge_pair_centre`, else the PnP-projected centre | A naive centroid of *whatever is visible* is the failure below. |

> **The unbalanced case is not an edge case.** The 20° up-tilt pushes the two
> **bottom** corners out of frame inside ~4.5 m on a centred approach. The
> centroid of what remains is the **top edge** of the opening — 0.75 m above
> centre. The drone flew to that point and clipped the top bar.

### 3.6 `_edge_pair_centre` — one visible edge is enough

On final approach the crop eats the gate until the last thing in frame is a
single edge of the opening, usually the top bar. The opening centre sits exactly
`_hi = 0.75 m` from that edge's midpoint, perpendicular to it, toward the inside
of the gate — projectable with the pinhole model. Depth comes from the PnP solve
when one exists, else from the edge's pixel length against the known 1.5 m
opening.

The hard part is the **inward sign**, decided most-reliable-first:

1. **Same-edge outer corners** — 0.6 m outward, exactly opposite the centre.
2. **Any other visible keypoint** — opposite and side corners all lie across the
   edge on the centre side. Raw-pixel geometric truth.
3. **The PnP-projected centre** — *direction only*; its magnitude is precisely
   what a one-edge trapezoid solve gets wrong.
4. **The YOLO box centre**, and only when it extends past the projected bar
   thickness. A cropped gate's box hugs the visible bar, whose centre sits
   ~0.2 × length off the edge **on the wrong side** — trusting it aimed the drone
   1.45 m *below* a gate at 3 m (log- and geometry-verified).
5. **Slot identity** — YOLO labels slots 0,1 as the image-top pair, so the centre
   is image-down. Side edges are skipped here.

If the solve was *rejected* (≥ 4 corners visible but reprojection too high), the
edge-pair path is **not** used — inconsistent keypoints must not be resurrected
from a single edge. The fallback exists only for genuine under-determination
(< 4 confident corners).

Without this path the gate vanished 3–5 m out and the pilot retargeted a **far**
gate mid-approach — log-verified as the dominant gate-2+ clip: it banked toward
the new target straight into the frame it was about to thread.

The edge-pair result carries `method = "edge-pair"`, `n_visible = 2`, and a
**hardcoded `reproj_px = 0.0`** — a "perfect" reprojection that means nothing.
Downstream consumers that gate on quality must exclude it explicitly (the
blue-line pilot's re-acquisition check does, via `n_visible ≥ 3`).

### 3.7 Plane normal and camera→body

The gate corners live on the gate-local `z = 0` plane, so the rotated local `+z`
is the plane normal. It is sign-fixed to point **back at the camera**
(`dot(n, gate_pos) > 0 ⇒ flip`), making the fly-through axis `−normal`.

Camera optical → body FRD is a fixed permutation composed with the 20° tilt:

```python
R_BODY_CAM = R_y(+20°) @ [[0,0,1],[1,0,0],[0,1,0]]
```

Bearings follow directly: `yaw = atan2(by, bx)`, `pitch = atan2(−bz, hypot(bx, by))`.

The edge-pair fallback cannot solve a plane orientation from two points, so it
returns the straight-facing assumption `[0, 0, −1]` — near-gate the guidance
blends toward pure bearing anyway, and zero tilt is the do-no-harm yaw input.

### 3.8 Self-test

`uv run python -m simulator.gate_pnp` projects the gate from 200 random known
poses (3–9 m depth, ±0.2 rad rotation), recovers each, and asserts:

- mean position error **< 0.1 m**
- mean plane-normal error **< 5°**

counting only corners actually in-frame as confident, so the visibility logic is
exercised too.

---

## 4. Fusion, tracking and gating

`gp_vision.py` turns "several detectors disagreeing" into one stable estimate.
This layer is where most of the *flight* bugs were fixed, as opposed to the
*geometry* bugs of §3.

### 4.1 Source ladder

```
1. YOLO pose packet    — 8-keypoint PnP, aimed at the opening      (fresh only)
2. data["anduril_gate"] — HSV-red detect + PnP / pinhole
3. HSV gate_target      — bearing/range rays, else pinhole from area
```

A YOLO packet is **stale** when it lags the camera by more than
`YOLO_STALE_GAP_FR = 5` frames. Five, not three: inference sits right at the
3-frame boundary, which made the source alternate YOLO↔Anduril nearly every tick
— a log-verified sawtooth in the commanded bearing.

### 4.2 Which gate

`best_pose_gate` filters on box confidence ≥ 0.5 and reprojection ≤ 10 px,
requires `body_x > 0.1` (in front), then takes the **nearest** gate, with bearing
only as a tie-break inside 1 m. Only the `MAX_GATES_CONSIDERED = 2` nearest are
even scored: a gate 3–4 course-lengths out can briefly outscore the one dead
ahead at handoff (big box, low bearing) and yank the aim toward it while the
drone is threading the near gate — the log-verified gate-2 clip.

> The current working tree simplified this from an explicit
> `cost = range + 5.0·|bearing|` to strict nearest-with-tie-break. Both forms
> exist in branch history; the cost form is on `ks/YOLO-implementation`.

### 4.3 Pass-through suppression

`YoloGateTracker` gives the YOLO path the same blindness the Anduril tracker
already had. A gate is "being threaded" when its pose puts it closer than
`YOLO_NEAR_BX_M = 2.0 m`, **or** its box fills ≥ 85 % of the frame width or
height. While threading, **all sources** are suppressed — not just YOLO — so the
pilot flies through blind rather than chasing the near edge with a lower-grade
detector. A 10-camera-frame cooldown follows.

`threading_gate` exposes this as a standing signal, which is what lets the
blue-line pilot distinguish "gate underneath us" from "nothing detected".

### 4.4 `GateEstimateSmoother` — EMA that never blends across identities

Four mechanisms, each added for a specific observed failure:

- **Dedup on native id.** One output per underlying measurement, keyed on
  `(source, source_frame_id)` before re-stamping.
- **Re-stamp onto the camera clock.** YOLO carries the older pose-packet frame
  id while Anduril carries the camera id, so raw ids **regress** on an
  Anduril→YOLO flip — silently bypassing the EMA gap guard and the guidance
  D-term guards.
- **PnP-sticky.** After a PnP-backed estimate, a non-PnP estimate within
  `PNP_STICKY_FRAMES = 5` is ignored in favour of the last output — YOLO only,
  since Anduril has its own EMA.
- **Target-identity lock.** Two gate positions are the *same gate* only if their
  range ratio is under 1.6× **and** their 3D directions are within 15°.
  Direction is the sharper discriminator: depth noise moves range but not
  direction. A challenger must persist `BREAK_CONFIRM_N = 3` frames before the
  lock moves, and alternating incumbent/challenger frames (two detectors on two
  gates) reset the count, so flapping can never steal the lock.

The EMA (α = 0.35) applies only within an identity and only across a frame gap
of ≤ 6 (200 ms).

**Source handoff is a snap, not a blend.** YOLO and Anduril's systematic offsets
differ by up to ~0.9 m vertically on clipped views, and a same-gate step passes
the 15° identity test, so an EMA would sweep the aim point through the offset.
But a flip that *also* lands on a different gate must still run the debounce —
snapping blindly there re-opened the "bank toward the far gate while threading
the near one" crash.

**Near→far handoff is a gate pass.** If the incumbent was inside 3 m and the
challenger is more than 3 m further out, we just crossed the old gate's plane:
force the blind cooldown rather than instantly chasing the next gate at the old
gate's edge. This is the source-agnostic version of §4.3.

### 4.5 Two derived signals

- `gate_tilt_deg_from_normal` — yaw offset between body-forward and the gate's
  fly-through axis, clipped ±30°. It flips the normal onto the +x half-space
  first, because the live detectors point the normal **back** at the drone while
  the synthetic expert points it forward; without the flip a detector normal read
  ±180° and clipped to a sign-flapping ±30° yaw dither. That bug was dormant
  while HSV rarely solved a normal, and armed the moment YOLO — which always has
  one — became primary.
- `VisionVelocityTracker` — body velocity by differencing consecutive gate body
  positions over the frame gap. It **resets on `track_break`**, because
  differencing across identities makes phantom velocity: a 1 m target switch over
  one frame reads as ~30 m/s.

---

## 5. The Try-1 / Try-2 / Try-3 ladder across branches

The interesting question this stack actually answers is *what should range and
aim come from*. Three answers were built, each on its own branch, each keeping
the others as fallbacks.

| | Branch(es) | Primary aim + range source | Rationale |
|---|---|---|---|
| **Try 1** | `main`, `ks/YOLO-implementation`, `ks_vision+blueline` | YOLO 8-keypoint PnP body pose | Learned detector is the most robust to lighting and partial views; PnP gives a full 6-DoF pose including plane normal. |
| **Try 2** | `test-classical`, `yahya-gate-detection` | **Classical HSV/CV inner-opening centroid + width → pinhole** | The opening is what you fly through; its contour is measured, not inferred. Pinhole range from a measured width is immune to the depth jitter a few px of keypoint error induces. |
| **Try 3** | `test-gatenet` | GateNet U-Net mask moments (centre / spread / fill) | A segmentation mask degrades gracefully where corner detection fails outright. |

All three keep the same fallback chain underneath: Anduril HSV → HSV
`gate_target`. The detectors in §2 are **byte-identical across every branch** —
`gate_detector.py`, `anduril_gate_detect.py` and `gate_corners_cv.py` have the
same blob hash on all of them. What differs is `gp_vision.py`'s preference order
and, on two branches, `gate_pose.py` and `gate_pnp.py`.

### 5.1 Try 1 details — where it went beyond plain PnP

`ks/YOLO-implementation` (commit `a9de27b`) added **width-pinhole range fusion**:
the inner-opening width gives an independent depth estimate, and it may override
the PnP depth only under three guards —

- the pinhole answer lands within a **0.6–1.6× ratio band** of the PnP range,
- the gate is near-frontal (tilt < 25°), so yaw-foreshortening is not shrinking
  the apparent width,
- the opening spans at least **55 px**. At long range the opening is only ~40 px,
  where a 2–3 px corner error is a large percentage depth error — the resulting
  jittery `bz` made the drone porpoise on the tall gate-2 climb. 55 px
  corresponds to roughly ≤ 8–9 m; beyond that, keep the PnP depth.

That branch reached gate 7 (`01bebf0`). A later commit (`2a745b0`) removed the
`ctr_px` / square-centre-fill experiment.

### 5.2 Try 2 details — the genuinely YOLO-free path

`test-classical` is the branch to point an advisor at for "how far does classical
CV alone get you". Commit `a809eac` made two changes:

1. **`cv_corners` publish without PnP.** The CV hole quad is attached to the gate
   dict whether or not a PnP solve succeeds — the pixel aim only needs the quad.
2. **A YOLO-free gate.** `_cv_only_gate` synthesises a complete gate entry from
   the HSV hole alone, with `conf = 1.0` and no keypoints, when YOLO returns
   nothing at all. The classical detector becomes a first-class source rather
   than a refinement.

`gp_vision` then prefers `cv_opening_pixels` → inner keypoints → box, and takes
range from the pinhole against the **1.5 m inner** width rather than the 2.7 m
outer. It also defines an explicit IBVS aim line,
`AIM_V = cy + fx·tan(20°)`, which is where the horizon projects under the camera
tilt.

`test-classical` also drops the whole `_edge_pair_centre` / `_diag_intersection`
machinery from `gate_pnp.py` — it forked before those landed, and the classical
hole centroid solves the same problem differently.

### 5.3 Try 3 details

`gatenet_vision.py` lazy-loads `rl/data/gatenet.pt` and runs it on the same
newest-frame-wins thread pattern, publishing `data["gatenet"]`. The mask is
reduced to centre `(u, v)`, horizontal `spread`, and `fill_frac`; range comes
from the same pinhole. `GATE_MASK=1/0` forces it on or off; unset means "on iff
the weights file exists", so the branch runs unchanged without weights.

### 5.4 Adjacent: pillar avoidance

`pillar_detect.py` (only on `ks_vision+blueline`) is not gate detection but
belongs to the same perception layer. The VQ2 course is an indoor,
parking-garage-like structure whose gates sit between large **dark** pillars, and
the gate-chasing pilot flies straight gate-to-gate and clips them. The detector
exploits a measured signature — lit lanes are bright, pillars and perimeter walls
occlude to near-black — by splitting a forward danger band (rows 0.35–0.68) into
left/centre/right thirds at 1/4 resolution:

- centre much darker than the open sides (< 0.6×) → pillar dead ahead → steer to
  the brighter side;
- one side much darker than the other (imbalance > 0.30) → wall → steer away.

A balanced scene yields ~0 steer, so a clean gate approach is undisturbed.

---

## 6. Training and data pipeline

**The model.** `simulator/models/gate_pose.pt`, 16.6 MB, one class, 8 keypoints
per instance, `kpt_shape: [8, 3]`. Trained on Unity gate renders. Keypoint order
is inner TL, TR, BL, BR then the matching outer four; the dataset's
`flip_idx: [1,0,3,2,5,4,7,6]` swaps left/right on horizontal flip augmentation.

**Retraining** — `scripts/train_gate_pose.ps1` (`make train-gate-pose`, present
on `test-classical` and `test-gatenet`):

- stays in the **pose** family, not detect-only bbox;
- defaults to `yolo11s-pose.pt`, 100 epochs, imgsz 640;
- trains on **real FPV JPEGs captured from this sim** (`make capture-fpv`),
  weighting the inner-4 keypoints over the outer-4 — the inner corners are the
  flight target, the outer ones are scale helpers;
- writes the Ultralytics dataset skeleton (`images/{train,val}`,
  `labels/{train,val}`, `data.yaml`) if missing and refuses to train under 10
  labelled images.

> Discrepancy worth flagging: `gate_pose.py`'s docstring says the bundled model
> is **YOLO11n**-pose, while the retrain script defaults to **YOLO11s**-pose.
> Whichever the shipped weights actually are, the two disagree in writing.

**GateNet (Try 3)** — `rl/gatenet.py`. A compact 4-level U-Net, `DoubleConv`
blocks with BatchNorm, single-logit output, trained at 320×192 (divisible by 16
for four pooling levels) with ImageNet normalisation and cv2/numpy augmentation
(flip, brightness, gamma, noise). Pure PyTorch so it trains on CPU.

**Gate map capture** — `rl/capture_gates.py`. Not part of the vision path, but
the reference data behind it: the sim transmits the track/gate map *only* as a
short burst at race start (`DATA_TRANSMISSION_HANDSHAKE` +
`ENCAPSULATED_DATA` with `data_type = 2`). The tool uses a blocking MAVLink
receive, reassembles the chunks, and writes `rl/data/gate_map.json` — used for
offline evaluation and for the focal-length self-calibration in
`gate_estimator.py`, never as a flight input under VQ2.

---

## 7. Failure modes and the fixes they forced

Every row below was observed in flight logs or reproduced geometrically, and
each one is now load-bearing code.

| Failure | Root cause | Fix |
|---|---|---|
| Gate solved back-face-toward-camera; drone banks the wrong way | Training-data generator wrote left labels onto right corners | Left/right swap baked into `_GATE_PTS_3D` |
| Drone clips the **top bar** on a centred approach | 20° up-tilt crops the bottom corners inside ~4.5 m; the centroid of the remaining corners is the opening's top edge, 0.75 m high | Balanced-visibility test; diagonal intersection for 4 corners; `_edge_pair_centre` otherwise |
| Persistent **low** aim while riding low into a gate | Centroid of 4 projected corners is pulled 6–10 % toward the nearer edge | Diagonal intersection — projectively exact |
| Aimed **1.45 m below** a gate at 3 m | A cropped gate's YOLO box hugs the visible bar, whose centre sits on the wrong side of the edge | Box centre demoted to 4th in the sign ladder, and only when it clears the projected bar thickness |
| Gate vanishes at 3–5 m; pilot retargets a far gate mid-approach and banks into the frame | PnP needs 4 corners; the crop leaves fewer | `edge-pair` single-edge pose |
| Aim yanked toward a distant gate at handoff | A far gate can outscore the near one on a confidence or cost metric | Only the 2 nearest gates are scored; nearest wins, bearing tie-breaks inside 1 m |
| Commanded bearing sawtooths every tick | Inference lag sat exactly on a 3-frame staleness boundary | `YOLO_STALE_GAP_FR = 5` |
| Aim sweeps through empty space between two gates | EMA blending across a target identity change | Range-ratio + direction identity test, 3-frame confirm, snap-don't-blend |
| Aim sweeps ~0.9 m vertically on a source flip | YOLO and Anduril have different systematic offsets on clipped views | Snap on source handoff — but still debounce if the gate identity also changed |
| Frame ids **regress**, bypassing gap guards | YOLO reports the pose-packet id, Anduril the camera id | Re-stamp every estimate onto the camera clock |
| Post-pass blind cooldown never fired; drone re-acquired the just-passed gate's edge | `reset()` on every `None` estimate wiped the tracker's cooldown state | `_clear_track()` drops the track without touching the tracker |
| Sign-flapping ±30° yaw dither near gates | Detector normals point back at the drone (±180°) while the synthetic expert's point forward | Flip onto the +x half-space before `atan2` |
| Phantom ~30 m/s velocity | Differencing gate positions across a target switch | `VisionVelocityTracker` resets on `track_break` |
| Porpoising on the gate-2 climb | Width-pinhole depth from a ~40 px opening; 2–3 px corner error is a large % error | `MIN_PINHOLE_SPREAD_PX = 55`, plus a 0.6–1.6× sanity band vs PnP |
| Mask returns every pixel; "gate" fills the frame | `cv2.inRange` bounds shaped `(1,1,3)` match everything on OpenCV 4.13 | Bounds pinned to `(1,3)` |
| Distant gate's opening disappears from the CV path | 5×5 closing kernel fills a few-px hole | 3×3 kernel in `gate_corners_cv` specifically |

---

## 8. Verification, knobs, and open gaps

### 8.1 Tests and self-tests

| Suite | Count | Covers |
|---|---|---|
| `tests/test_gate_pnp.py` | 15 | pose recovery, visibility gating, the balanced/unbalanced centre branches, edge-pair sign ladder, reprojection rejection |
| `tests/test_gate_corners_cv.py` | 8 | hole-contour extraction, border rejection, corner ordering, shrink fallback |
| `tests/test_gate_detector.py` | 4 | HSV mask, hue wraparound, contour scoring |
| `tests/test_gp_vision.py` (`test-classical`) | 9 | the Try-2 classical-hole preference order |
| `python -m simulator.gate_pnp` | — | 200 synthetic poses: mean position error < 0.1 m, mean normal error < 5° |
| `python -m simulator.pillar_detect --selftest` | — | pillar / wall / open-scene discrimination |

CI runs only ruff and documentation checks — the unit tests are **not** gated, so
a committed baseline can carry failing tests.

### 8.2 Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `GATE_CV_REFINE` | `1` | classical inner-corner refinement of the best YOLO hit |
| `GATE_MASK` | auto | `test-gatenet`: GateNet on/off; unset = on iff weights exist |
| `SKIP_YOLO` | — | disable the learned detector entirely |
| `GATE_HEX_COLOR`, `HSV_TOLERANCE` | `#F3390F`, 60 | `config.py`; drives the §2.1 mask and, transitively, `gate_corners_cv` |

### 8.3 Open gaps

- **The three Trys were never scored head-to-head.** Each lives on its own
  branch with its own `gp_vision.py` preference order, and there is no A/B
  harness comparing aim error or range error on identical frames. This is the
  most valuable missing experiment in the stack, and the blue-line work already
  demonstrated the pattern that would fix it — a passive shadow mode that
  computes a candidate estimator and logs it while still flying the incumbent.
- **The bundled weights are trained on Unity renders**, while the retrain script
  targets real FPV captures from the sim. The domain gap between them is
  unquantified.
- **CPU inference dominates the loop.** 100–350 ms/frame means the perception
  rate is ~4–10× below the camera rate, and essentially every timing guard in §4
  exists to survive that. A GPU path exists (`_DEVICE = "cuda"` when available)
  but is not what the flight logs were taken on.
- **`edge-pair` reports `reproj_px = 0.0`.** Any consumer that ranks estimates by
  reprojection quality will rank the *least* certain method first unless it also
  checks `n_visible`.
- **`gate_estimator.py`'s focal self-calibration needs track ground truth**,
  which VQ2 does not provide — so in the configuration that matters it always
  falls back to the coarse pixel-ratio estimate.
- **Model-family discrepancy** between the docstring (YOLO11n-pose) and the
  retrain script (YOLO11s-pose) is unresolved in writing.
