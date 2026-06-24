import unittest

from simulator.config import DroneState, GateDetection, TrackGate
from simulator.gate_estimator import GateEstimator


class TestGateEstimator(unittest.TestCase):
    def test_range_from_detection_with_intrinsics(self):
        est = GateEstimator()
        det = GateDetection(
            frame_id=1,
            sim_time_ns=0,
            centroid_x_px=330.0,
            centroid_y_px=175.0,
            area_px=5000.0,
            width_px=40.0,
            height_px=40.0,
            contour_valid=True,
        )
        drone = DroneState(
            pos_ned=(0.0, 0.0, -5.0),
            vel_ned=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            yaw_rate=0.0,
            time_boot_ms=0,
            has_position=False,
        )
        gates = [
            TrackGate(
                gate_id=0,
                pos_ned=(20.0, 0.0, -5.0),
                orient_quat=(1.0, 0.0, 0.0, 0.0),
                width_m=2.72,
                height_m=2.72,
            )
        ]
        out = est.update(det, drone, gates, 0)
        self.assertIsNotNone(out.range_m)
        self.assertGreater(out.range_m, 5.0)
        self.assertEqual(out.source, "intrinsics")
        self.assertGreater(out.confidence, 0.4)


if __name__ == "__main__":
    unittest.main()
