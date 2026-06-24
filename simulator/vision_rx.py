from __future__ import annotations

import socket
import struct
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from simulator.gate_classifier import ClassifiedGate

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600


class VisionRX:
    def __init__(self, data, vio=None, gate_tracker=None):
        self.data = data
        self.vio = vio
        self.gate_tracker = gate_tracker
        from simulator.gate_classifier import GateClassifier
        from simulator.obstacle_tracker import ObstacleTracker

        self._gate_classifier = GateClassifier()
        self._obstacle_tracker = ObstacleTracker()
        self.thread = threading.Thread(target=self._vision_loop, daemon=False)
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("Listening for camera frames...")

        while self.is_running:
            packet, addr = sock.recvfrom(65536)  # max UDP size

            header = packet[:header_sz]
            payload = packet[header_sz:]

            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = (
                struct.unpack(header_format, header)
            )

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns,
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()
                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        print(
                            "Missing packet %s in frame %s" % (i, frame_id),
                        )
                        frame_complete = False
                        continue
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    self.process_frame(frame_id, image, sim_time_ns)
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

    def process_frame(self, frame_id, img, sim_time_ns=0):
        try:
            import time as _time

            from simulator.gate_detector import build_gate_mask, detect_gate
            from simulator.obstacle_detector import detect_obstacles

            h, w = img.shape[:2]
            gate_mask = build_gate_mask(img)
            raw_detection = detect_gate(img, frame_id, sim_time_ns)
            classified = self._gate_classifier.classify(raw_detection, w, h)

            obstacle_dets = detect_obstacles(
                img,
                frame_id,
                sim_time_ns,
                gate_detection=raw_detection,
                gate_mask=gate_mask,
            )
            obstacle_tracks = self._obstacle_tracker.update(
                obstacle_dets, frame_id, w, h
            )

            self.data["camera"] = {"received_at": _time.monotonic()}
            self.data["frame"] = {
                "img": img,
                "frame_id": frame_id,
                "sim_time_ns": sim_time_ns,
                "received_at": _time.monotonic(),
            }

            self._publish_gate_target(classified, w, h)
            self._publish_obstacles(obstacle_tracks)

            detection_for_fusion = (
                classified.detection
                if classified.validated
                or (
                    classified.geometric_valid
                    and classified.temporal_streak >= 2
                )
                else None
            )

            if self.gate_tracker is not None:
                updated = self.gate_tracker.process_detection(
                    detection_for_fusion, frame_id, sim_time_ns
                )
                if updated:
                    gt = self.data.get("gate_track", {})
                    print(
                        f"[gate] pos={gt.get('pos_ned')} "
                        f"q={gt.get('quality', 0):.2f} "
                        f"P={gt.get('P_trace', 0):.2f}",
                        flush=True,
                    )

            if self.vio is not None:
                updated = False
                if detection_for_fusion is not None:
                    updated = self.vio.update_from_gate_detection(
                        detection_for_fusion, frame_id, sim_time_ns
                    )
                if not updated and gate_mask is not None and classified.validated:
                    updated = self.vio.update_from_gate_mask(
                        gate_mask, frame_id, sim_time_ns
                    )
                if updated:
                    vio = self.data.get("vio", {})
                    print(
                        f"[vio] pos={vio.get('pos_ned')} "
                        f"reproj={vio.get('reproj_err_px', 0):.2f}px "
                        f"P={vio.get('P_trace', 0):.2f}",
                        flush=True,
                    )

        except Exception as e:
            from simulator import config

            if config.DEBUG:
                print(f"[vision_rx] process_frame error: {e}")

    def _publish_gate_target(self, classified: ClassifiedGate, w: int, h: int) -> None:
        from simulator.config import GATE_CONFIDENCE_MIN_NAV

        det = classified.detection
        self.data["gate_classification"] = {
            "gate_confidence": classified.gate_confidence,
            "temporal_streak": classified.temporal_streak,
            "geometric_valid": classified.geometric_valid,
            "ambiguous": classified.ambiguous,
            "validated": classified.validated,
        }

        if det is None:
            self.data["gate_target"] = {
                "detected": False,
                "nx": 0.0,
                "ny": 0.0,
                "r_frac": 0.0,
                "quality": 0.0,
                "gate_confidence": 0.0,
                "ambiguous": True,
                "validated": False,
            }
            return

        nx = (det.centroid_x_px - w / 2.0) / (w / 2.0)
        ny = (det.centroid_y_px - h / 2.0) / (h / 2.0)
        r_frac = det.area_px / (w * h)
        nav_goal = (
            classified.validated
            or (
                classified.geometric_valid
                and classified.temporal_streak >= 2
                and classified.gate_confidence >= GATE_CONFIDENCE_MIN_NAV
            )
            or (
                classified.temporal_streak >= 3
                and r_frac > 0.04
                and abs(nx) < 0.55
            )
        )

        self.data["gate_target"] = {
            "detected": nav_goal,
            "raw_detected": True,
            "nx": nx,
            "ny": ny,
            "r_frac": r_frac,
            "quality": det.quality,
            "gate_confidence": classified.gate_confidence,
            "temporal_streak": classified.temporal_streak,
            "ambiguous": classified.ambiguous,
            "validated": classified.validated,
            "reproj_err_px": det.reproj_err_px,
            "width_px": det.width_px,
            "height_px": det.height_px,
            "corners_px": det.corners_px,
        }

        tag = "GATE" if nav_goal else ("AMBIG" if classified.ambiguous else "RAW")
        print(
            f"[vision] {tag} cx={det.centroid_x_px:.0f} cy={det.centroid_y_px:.0f} "
            f"area={det.area_px:.0f} conf={classified.gate_confidence:.2f} "
            f"streak={classified.temporal_streak} nx={nx:+.3f} ny={ny:+.3f}",
            flush=True,
        )

    def _publish_obstacles(self, tracks: list[dict]) -> None:
        self.data["obstacles"] = tracks
        self.data["obstacle_track"] = {
            "count": len(tracks),
            "tracks": tracks,
            "blocking": any(
                t.get("confidence", 0) >= 0.4
                and abs(t.get("nx", 0)) < 0.25
                and t.get("r_frac", 0) > 0.005
                for t in tracks
            ),
        }
