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

_WINDOW_NAME = "drone vision"
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
_FPS = 30.0
# All recordings collect in one folder of their own. runs/ itself is shared
# with the attitude harness, which drops a directory per run — mixing 67 MB
# videos in among those made the videos hard to find.
_RECORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "runs", "videos"
)
# Stamped per recording so runs accumulate instead of overwriting each other —
# a rare good run used to be destroyed by the next launch. Same %Y%m%d_%H%M%S
# convention as rl/data/gp_log_*.csv, so a video pairs with its telemetry by
# filename (stamped at the first frame vs the log's race start, so expect a few
# seconds of skew — pair by nearest, not exact).
_RECORD_FMT = "vision_%Y%m%d_%H%M%S.mp4"

# Set False to skip the mp4 (live window only).
RECORD = True

_video_writer = None
_record_path = None
_window_open = False


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
                _record_path = os.path.join(
                    _RECORD_DIR, time.strftime(_RECORD_FMT)
                )
                _video_writer = cv2.VideoWriter(_record_path, _FOURCC, _FPS, (w, h))
                print(f"[display] recording -> {_record_path}", flush=True)
            _video_writer.write(frame)
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
        _record_path = None
    if _window_open:
        cv2.destroyAllWindows()
        _window_open = False
