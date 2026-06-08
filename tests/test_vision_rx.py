import numpy as np

from simulator.vision_processing import GateTargetFilter
from simulator.vision_rx import VisionRX


def test_process_frame_populates_shared_data():
    data = {}
    rx = VisionRX.__new__(VisionRX)
    rx.data = data
    rx._gate_filter = GateTargetFilter()

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    import cv2

    cv2.circle(image, (120, 50), 20, (255, 120, 40), thickness=6)

    rx.process_frame(frame_id=7, img=image, sim_time_ns=12345)

    assert data["camera"]["frame_id"] == 7
    assert data["camera"]["width"] == 200
    assert data["camera"]["height"] == 100
    assert data["camera"]["sim_time_ns"] == 12345
    assert "received_at" in data["camera"]
    assert "gate_target" in data
    assert "detected" in data["gate_target"]
