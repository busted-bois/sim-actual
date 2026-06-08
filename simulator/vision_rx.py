import socket
import struct
import threading
import time

import cv2
import numpy as np

from simulator.gate_detector import GateDetector

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600


class VisionRX:
    def __init__(self, data):
        self.data = data
        self.detector = GateDetector()
        self.last_debug_log_s = 0.0
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

            # frame_id - identifier for this vision frame
            # chunk_id - identifier for this chunk packet of data of this frame
            # total_chunks - total number of chunk packets that make up this frame
            # jpeg_size - full size of jpeg data
            # payload_size - size of this packet
            # sim_time_ns - frame's epoch timestamp in ns on the server
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(
                header_format, header
            )

            if frame_id not in frames:
                frames[frame_id] = {"chunks": {}, "total": total_chunks, "size": jpeg_size, "time": sim_time_ns}

            frames[frame_id]["chunks"][chunk_id] = payload

            # Check if frame is complete
            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()

                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        print(
                            "Missing packet %s in frame %s"
                            % (
                                i,
                                frame_id,
                            )
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
                    self.process_frame(frame_id, image)
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

    def process_frame(self, frame_id, img):
        detection = self.detector.detect(frame_id, img)
        now_s = time.monotonic()
        with self.data["lock"]:
            self.data["latest_detection"] = detection
            self.data["latest_frame_id"] = frame_id
            self.data["latest_vision_time"] = now_s
        if now_s - self.last_debug_log_s >= 0.2:
            self.last_debug_log_s = now_s
            if detection is None:
                print(f"vision frame={frame_id} det=none", flush=True)
            else:
                print(
                    "vision frame=%s conf=%.2f range=%.2f target=(%.0f,%.0f) bbox=%s candidates=%s"
                    % (
                        frame_id,
                        detection.confidence,
                        detection.range_m,
                        detection.target_x,
                        detection.target_y,
                        detection.bbox,
                        detection.candidate_count,
                    ),
                    flush=True,
                )
