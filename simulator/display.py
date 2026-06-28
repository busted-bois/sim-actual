"""Live vision window + per-run mp4 recording.

Pops up a cv2 window showing what the drone's camera sees (raw frame, or the
annotated frame with gate/obstacle overlays produced by vision_rx). Lets you
watch the perception pipeline live while a race runs.

imshow/waitKey are GUI calls and MUST run on the same thread that created the
window. Call start()/tick()/close() all from the entry point's main thread
(main.py) -- never from the VisionRX receiver thread.

Usage:
    display.start()                # create the window
    display.tick(frame, elapsed)   # every loop iter; frame may be None
    display.close()                # finalize the mp4
"""

import os

import cv2

_WINDOW_NAME = "drone vision"
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
_FPS = 30.0
_RECORD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "runs", "vision.mp4"
)

# Set False to skip the mp4 (live window only).
RECORD = True

_video_writer = None
_window_open = False


def start():
    """Create the cv2 window. Call once before the first tick()."""
    global _window_open
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    _window_open = True


def tick(frame, elapsed):
    """Show one frame and (lazily) record it. `frame` may be None -- we still
    pump waitKey so the window stays responsive while waiting for the first
    sim frame. `elapsed` (s) is drawn so screen-recordings self-timestamp."""
    global _video_writer
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
                os.makedirs(os.path.dirname(_RECORD_PATH), exist_ok=True)
                h, w = frame.shape[:2]
                _video_writer = cv2.VideoWriter(_RECORD_PATH, _FOURCC, _FPS, (w, h))
                print(f"[display] recording -> {_RECORD_PATH}", flush=True)
            _video_writer.write(frame)
        cv2.imshow(_WINDOW_NAME, frame)

    cv2.waitKey(1)


def close():
    """Finalize the mp4 and destroy the window."""
    global _video_writer, _window_open
    if _video_writer is not None:
        _video_writer.release()
        _video_writer = None
        print(f"[display] video saved -> {_RECORD_PATH}", flush=True)
    if _window_open:
        cv2.destroyAllWindows()
        _window_open = False
