"""Unit tests for the live-vision window's run recording (simulator/display).

Every run used to record over the same runs/vision.mp4, so each launch
destroyed the previous recording. These lock in the replacement: one
timestamped mp4 per run, all collected in runs/videos/.

cv2's GUI calls are stubbed throughout -- these must pass headless (CI, SSH).
"""

import glob
import os
import unittest
from unittest.mock import patch

import numpy as np

from simulator import display


class _HeadlessRun:
    """Stub the cv2 GUI calls so start/tick/close work with no display."""

    def __enter__(self):
        self._patches = [
            patch.object(display.cv2, "namedWindow"),
            patch.object(display.cv2, "imshow"),
            patch.object(display.cv2, "waitKey", return_value=-1),
            patch.object(display.cv2, "destroyAllWindows"),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


class RecordingPathTests(unittest.TestCase):
    def test_records_into_a_videos_subfolder(self):
        self.assertEqual(os.path.basename(display._RECORD_DIR), "videos")
        self.assertEqual(
            os.path.basename(os.path.dirname(display._RECORD_DIR)), "runs"
        )

    def test_record_dir_is_repo_relative_not_cwd_relative(self):
        # Derived from __file__, so `make sim` works from any directory.
        self.assertTrue(os.path.isabs(display._RECORD_DIR))
        self.assertNotIn("..", display._RECORD_DIR)

    def test_filename_is_timestamped(self):
        import time

        name = time.strftime(display._RECORD_FMT)
        self.assertTrue(name.startswith("vision_"))
        self.assertTrue(name.endswith(".mp4"))
        # vision_YYYYmmdd_HHMMSS.mp4 -- same stamp format as gp_log_*.csv, so a
        # video pairs with its telemetry by filename.
        stamp = name[len("vision_") : -len(".mp4")]
        self.assertEqual(len(stamp), 15, stamp)
        self.assertEqual(stamp[8], "_")
        self.assertTrue(stamp.replace("_", "").isdigit(), stamp)


class RecordingLifecycleTests(unittest.TestCase):
    """Exercise start/tick/close against a temp directory."""

    def setUp(self):
        self._tmp = os.path.join(
            os.path.dirname(__file__), "_display_tmp_%d" % os.getpid()
        )
        self._dir_patch = patch.object(display, "_RECORD_DIR", self._tmp)
        self._dir_patch.start()
        self.addCleanup(self._dir_patch.stop)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for p in glob.glob(os.path.join(self._tmp, "*")):
            try:
                os.remove(p)
            except OSError:
                pass
        if os.path.isdir(self._tmp):
            os.rmdir(self._tmp)

    def _run(self, n_frames=3):
        with _HeadlessRun():
            display.start()
            img = np.zeros((360, 640, 3), np.uint8)
            for i in range(n_frames):
                display.tick(img, float(i))
            display.close()

    def _recordings(self):
        return sorted(glob.glob(os.path.join(self._tmp, "*.mp4")))

    def test_creates_the_folder_on_demand(self):
        self.assertFalse(os.path.isdir(self._tmp))
        self._run()
        self.assertTrue(os.path.isdir(self._tmp))

    def test_two_runs_produce_two_files(self):
        # The regression this whole change exists for.
        self._run()
        with patch.object(display, "_RECORD_FMT", "vision_second.mp4"):
            self._run()
        names = [os.path.basename(p) for p in self._recordings()]
        self.assertEqual(len(names), 2, names)

    def test_close_resets_state_so_the_next_run_opens_a_new_file(self):
        self._run()
        self.assertIsNone(display._record_path)
        self.assertIsNone(display._video_writer)

    def test_recording_is_non_empty(self):
        self._run(n_frames=5)
        written = self._recordings()
        self.assertEqual(len(written), 1)
        self.assertGreater(os.path.getsize(written[0]), 0)

    def test_tick_before_start_records_nothing(self):
        with _HeadlessRun():
            display.tick(np.zeros((360, 640, 3), np.uint8), 0.0)
        self.assertEqual(self._recordings(), [])

    def test_record_flag_off_writes_no_file(self):
        with patch.object(display, "RECORD", False):
            self._run()
        self.assertEqual(self._recordings(), [])

    def test_no_frames_writes_no_file(self):
        # tick(None) keeps the window pumping before the first sim frame.
        with _HeadlessRun():
            display.start()
            display.tick(None, 0.0)
            display.close()
        self.assertEqual(self._recordings(), [])


if __name__ == "__main__":
    unittest.main()
