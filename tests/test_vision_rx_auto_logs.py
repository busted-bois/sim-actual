import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from simulator.vision_rx import VisionRX


class VisionRxAutoLogTests(unittest.TestCase):
    def setUp(self):
        self.data = {}
        with patch.object(VisionRX, "__init__", lambda self, data: None):
            self.rx = VisionRX(self.data)
        self.rx.data = self.data
        self.rx._gate_was_detected = False

    @patch("simulator.auto_flight.auto_flight_enabled", return_value=True)
    def test_auto_mode_edge_logs_only(self, _auto):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            self.rx._log_gate_detected(True)
            self.rx._log_gate_detected(True)
            self.rx._log_gate_detected(False)
            self.rx._log_gate_detected(False)
        out = buf.getvalue()
        self.assertEqual(out.count("GATE acquired"), 1)
        self.assertEqual(out.count("GATE lost"), 1)
        self.assertNotIn("cx=", out)

    @patch("simulator.auto_flight.auto_flight_enabled", return_value=False)
    def test_sim_mode_per_frame_telemetry(self, _auto):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            self.rx._log_gate_detected(True, 100.0, 50.0, 1000.0, 0.1, -0.2, 8.0)
        self.assertIn("[vision] GATE cx=", buf.getvalue())

    @patch("simulator.auto_flight.auto_flight_enabled", return_value=True)
    @patch("simulator.gate_detector.detect_gate")
    def test_process_frame_auto_no_repeat_spam(self, mock_detect, _auto):
        mock_det = MagicMock()
        mock_det.centroid_x_px = 320.0
        mock_det.centroid_y_px = 180.0
        mock_det.area_px = 5000.0
        mock_det.width_px = 80.0
        mock_det.height_px = 80.0
        mock_detect.return_value = mock_det

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch.object(VisionRX, "_estimate_geometry", return_value=None):
                self.rx.process_frame(1, img)
                self.rx.process_frame(2, img)
        self.assertEqual(buf.getvalue().count("GATE acquired"), 1)


if __name__ == "__main__":
    unittest.main()
