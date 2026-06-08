# Autonomous Color-Navigation Pilot

A toggleable autonomy stack that detects the **orange gate** (`#f3390f`), flies
through gates one after another, follows the **blue path** as a fallback heading,
and **stops safely** once it can no longer see any gate or path (end of course).

## How it fits together

```
camera frames ──► VisionRX.process_frame ──► analyze_frame()      (simulator/vision_processing.py)
                                              │  detects gate + path (HSV color)
                                              ▼
                              shared_data["vision_analysis"]       (the bus)
                                              │
control loop ──► Controller._update_autonomous ──► GateNavigator.compute()  (simulator/navigation.py)
                                              │  state machine -> velocity + yaw
                                              ▼
                       update_velocity_flight_control()            (simulator/controller.py)
                          body-frame velocity + yaw rate to the sim
```

- **`simulator/vision_processing.py`** — pure color detection. Converts each frame
  to HSV, thresholds the gate/path colors, and returns normalized blob position
  (`cx_norm`, `cy_norm` in `-1..+1`) and `area_frac` (how big it is = how close).
- **`simulator/navigation.py`** — the `GateNavigator` state machine:
  `SEARCHING → APPROACHING → PASSING → (repeat)`, with `FOLLOWING_PATH` fallback
  and a `COMPLETE` safe-stop.
- **`simulator/controller.py`** — sends body-frame velocity + yaw-rate setpoints
  and runs the configured end-of-course action.
- **`settings.json`** — all tuning lives here (see below).

## Toggling autonomy (settings.json)

Everything is driven by `settings.json` at the repo root. The relevant switch:

```json
"autonomy": {
  "enabled": true,
  "algorithm": "orange_gate"
}
```

- `"enabled": false` **or** `"algorithm": "off"` → the pilot reverts to the
  original template motor control (no autonomy). This is the "toggle" you asked for.
- `"algorithm": "orange_gate"` → runs the color-navigation pilot.

The field is intentionally a *name*, not just a boolean, so additional strategies
can be dropped in later (e.g. `"algorithm": "depth_gate"`) and selected the same way.

If `settings.json` is missing or partial, built-in defaults in
`simulator/config.py` fill the gaps, so the sim always starts.

### Key tunables

| Section | Key | Meaning |
|---|---|---|
| `vision` | `gate_hsv_ranges` | OpenCV HSV bands for the orange gate. Two bands cover the red/orange hue wrap. Loosen `S`/`V` floors if shadows hide the gate. |
| `vision` | `path_hsv_ranges` | HSV band(s) for the blue path. |
| `vision` | `min_blob_area_frac` | Ignore blobs smaller than this fraction of the frame (noise filter). |
| `vision` | `debug_save_every` | If `>0`, save an annotated frame every N frames to `debug_dir` for inspection. |
| `navigation` | `gate_pass_area_frac` | When the gate fills this fraction of the frame (and is centered), commit to flying through. |
| `navigation` | `pass_through_seconds` | How long to drive straight "blind" through the gate after committing. |
| `navigation` | `yaw_gain` / `max_yaw_rate` | How hard / how fast to turn toward a gate. |
| `safety` | `end_of_course_seconds` | No gate **and** no path for this long → declare course complete and stop. |
| `safety` | `require_detection_before_end` | Don't "finish" before the first frame/detection ever arrives. |
| `safety` | `max_run_seconds` | Hard time cap (0 = off). A blunt watchdog. |
| `safety` | `end_action` | `hover` (default), `land`, or `disarm` once complete. |

## Safety / "how it knows it reached the end"

The navigator only declares the course **COMPLETE** after it has seen *at least one*
gate or path, and then sees *nothing* for `end_of_course_seconds`. On completion it
zeroes velocity and yaw (hover), then runs `safety.end_action`. There's also a hard
`max_run_seconds` watchdog. This is what stops the drone from flying off forever once
the last gate is behind it.

> Note: while `SEARCHING`, the drone scans with a slow yaw in one direction. On a
> forward-only course this finds the next gate; it intentionally does **not** spin a
> full 360° (which could re-detect a gate already passed). Tune `search_yaw_rate` /
> `search_creep_speed` per scenario.

---

## Testing

### 1. Unit tests (no simulator needed)

The detection and the navigator state machine are pure functions / classes, so they
test on synthetic frames and a fake clock:

```bash
uv run python -m pytest tests/ -q
# or, without pytest:
uv run python tests/test_vision_processing.py
uv run python tests/test_navigation.py
```

These cover: gate/path detection and left/right sign, blank-frame = no detection,
area-grows-with-size, approach yaw direction, gate pass commit, path-follow fallback,
end-of-course completion, no-premature-finish, and the max-runtime cap.

### 2. Tune the colors offline (no simulator needed)

Run the detection pipeline on a still image (e.g. a screenshot/saved frame from the
sim) or a generated synthetic scene, and inspect the masks:

```bash
# synthetic gate+path scene
uv run python tools/vision_preview.py

# a real saved frame
uv run python tools/vision_preview.py path/to/frame.jpg
```

It writes `*_annotated.jpg` (detections drawn) plus `*_gatemask.jpg` / `*_pathmask.jpg`.
If the gate isn't lighting up the gate mask under sim lighting, widen the `S`/`V`
floors of `vision.gate_hsv_ranges` in `settings.json` and re-run.

To capture real frames automatically while flying, set
`"vision": { "debug_save_every": 30 }` and check the `debug_frames/` folder.

### 3. End-to-end in the simulator

1. Launch `FlightSim.exe` and log in / start a qualifier scenario.
2. Confirm `settings.json` has `autonomy.enabled = true` and
   `algorithm = "orange_gate"`.
3. Run the pilot:

   ```bash
   make sim        # or: uv run main.py
   ```

4. Watch the console. You should see:
   - `[config] loaded ... autonomy.enabled=True algorithm=orange_gate`
   - `[controller] autonomy ON (orange_gate navigator)`
   - phase transitions: `[nav] SEARCHING -> APPROACHING`, `[nav] passing gate #1`, …
   - on finish: `[nav] COURSE COMPLETE: no gate/path for 3.0s (gates passed: N)`
     and `[controller] course complete -> end_action=hover`.

5. **A/B the toggle:** set `autonomy.enabled = false`, re-run, and confirm you get
   `[controller] autonomy OFF (template motor control)` and the original behavior.

### Tuning checklist if it misbehaves in-sim

- **Gate not detected / detected late** → loosen `gate_hsv_ranges` S/V floors; lower
  `min_blob_area_frac`. Verify with `tools/vision_preview.py` on a saved frame.
- **Clips the gate edge** → raise `gate_pass_area_frac` (commit later/closer) and/or
  tighten `gate_pass_center_tol`.
- **Turns too slowly / overshoots** → adjust `yaw_gain` and `max_yaw_rate`.
- **Stops too early between gates** → increase `end_of_course_seconds`.
- **Never stops at the end** → decrease `end_of_course_seconds`, or set a
  `max_run_seconds` cap.
