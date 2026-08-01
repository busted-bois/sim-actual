"""Live vision window + per-run mp4 recording.

Pops up a cv2 window showing what the drone's camera sees (raw frame, or the
annotated frame with gate/obstacle overlays produced by vision_rx). Lets you
watch the perception pipeline live while a race runs.

imshow/waitKey are GUI calls and MUST run on the same thread that created the
window. Call start()/tick()/close() all from the entry point's main thread
(fly2.main, main.py) -- never from the VisionRX receiver thread.

Recordings collect in runs/videos/, one timestamped mp4 per run, so they
accumulate rather than overwriting one another.

Usage:
    display.start()                # create the window
    display.tick(frame, elapsed)   # every loop iter; frame may be None
    display.close()                # finalize the mp4
"""

import os
import time

import cv2

from simulator.run_id import RUN_ID

_WINDOW_NAME = "drone vision"
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
# Written into the container header, but frames are only written when a new
# camera frame arrives, so the real rate is lower and varies per run. The true
# rate is measured below and published in the sidecar -- do not use this to map
# a timestamp to a frame number.
_FPS = 30.0
# All recordings collect in one folder of their own. runs/ itself is shared
# with the attitude harness, which drops a directory per run — mixing 67 MB
# videos in among those made the videos hard to find.
_RECORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "runs", "videos"
)
# Named from RUN_ID, not the wall clock at the first frame. Both this and
# rl/data/gp_log_<RUN_ID>_a<N>.csv key off the same id, so a recording and its
# telemetry pair exactly instead of by nearest timestamp — the video used to
# stamp before the countdown and the log after it, minutes apart on a slow start.
_RECORD_FMT = "vision_{run_id}.mp4"

# Set False to skip the mp4 (live window only).
RECORD = True

_video_writer = None
_record_path = None
_window_open = False

# Measured while recording, read by run_meta at close() to publish the real
# frame rate and the epoch origin of the burned-in t= overlay.
_frames = 0
_first_unix = None
_first_elapsed = None
_last_unix = None
_size = None


def _record_name():
    """Filename for this process's recording. Reads the globals at call time so
    tests can patch either piece."""
    return _RECORD_FMT.format(run_id=RUN_ID)


def stats():
    """What was recorded this run, for the sidecar. Empty dict if nothing was."""
    if not _frames or _first_unix is None:
        return {}
    span = (_last_unix or _first_unix) - _first_unix
    return {
        "path": _record_path,
        "fps_nominal": _FPS,
        # Frames land at the camera's rate, not _FPS. Mapping a telemetry
        # timestamp to a frame needs this measured value.
        "fps_actual": round(_frames / span, 3) if span > 0 else None,
        "frames": _frames,
        "width": _size[0] if _size else None,
        "height": _size[1] if _size else None,
        "first_frame_unix": round(_first_unix, 3),
        # The overlay clock differs per entry point (time.time in auto_gp,
        # time.monotonic in main/vision_view), so record where it started
        # rather than assuming an epoch.
        "first_frame_overlay_s": _first_elapsed,
    }


def pick(data):
    """Choose what to show from the shared data dict, returning (img, tag).
    Prefers the YOLO-pose annotated frame (data["pose"]), then the classical
    overlay, then the raw frame. When pose wins, blue-line HUD is composited
    on top (vision_rx only writes it to frame["annotated"], which display
    otherwise never shows during make control-flight / make classical blue).
    `tag` changes only when a new frame is available, so callers can skip
    redundant ticks. (None, None) if no frame."""
    pose = data.get("pose")
    frame = data.get("frame")
    bl = data.get("blue_line")
    bl_fid = bl.get("frame_id") if bl is not None else None

    if pose is not None and pose.get("annotated") is not None:
        img = pose["annotated"]
        tag = ("p", pose["frame_id"], bl_fid)
        if img is not None and bl is not None:
            from simulator.blue_line_vision import (
                annotate_blue_lines,
                estimate_from_dict,
            )

            img = annotate_blue_lines(img, estimate_from_dict(bl), None)
        return img, tag
    if frame is not None:
        return frame.get("annotated", frame.get("img")), ("f", frame["frame_id"], bl_fid)
    return None, None


def start():
    """Create the cv2 window. Call once before the first tick()."""
    global _window_open
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    _window_open = True


def tick(frame, elapsed):
    """Show one frame and (lazily) record it. `frame` may be None -- we still
    pump waitKey so the window stays responsive while waiting for the first
    sim frame. `elapsed` (s) is drawn so screen-recordings self-timestamp."""
    global _video_writer, _record_path
    global _frames, _first_unix, _first_elapsed, _last_unix, _size
    if not _window_open:
        return

    if frame is not None:
        frame = frame.copy()
        cv2.putText(
            frame,
            f"t={elapsed:6.2f}s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if RECORD:
            if _video_writer is None:
                os.makedirs(_RECORD_DIR, exist_ok=True)
                h, w = frame.shape[:2]
                _record_path = os.path.join(_RECORD_DIR, _record_name())
                _video_writer = cv2.VideoWriter(_record_path, _FOURCC, _FPS, (w, h))
                _frames, _last_unix = 0, None
                _first_unix, _first_elapsed, _size = time.time(), elapsed, (w, h)
                print(f"[display] recording -> {_record_path}", flush=True)
            _video_writer.write(frame)
            _frames += 1
            _last_unix = time.time()
        cv2.imshow(_WINDOW_NAME, frame)

    # waitKey is what actually paints the window + pumps OS events.
    cv2.waitKey(1)


def close():
    """Finalize the mp4 and destroy the window."""
    global _video_writer, _record_path, _window_open
    if _video_writer is not None:
        _video_writer.release()
        _video_writer = None
        print(f"[display] video saved -> {_record_path}", flush=True)
        # Publish the measured rate + size next to the mp4. Imported here, not
        # at module scope, so the recorder keeps working if run_meta is missing.
        from simulator import run_meta

        run_meta.note_video(stats())
        run_meta.finalize()
        _record_path = None
    if _window_open:
        cv2.destroyAllWindows()
        _window_open = False
