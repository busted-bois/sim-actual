import socket
import struct
import threading
import time

import cv2
import numpy as np

from simulator.vision_processing import GateTargetFilter, detect_gate_target

SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600
MAX_INCOMPLETE_FRAMES = 32
SOCKET_TIMEOUT_S = 0.5


class VisionRX:
    def __init__(self, data):
        self.data = data
        self._gate_filter = GateTargetFilter()
        self.sock = None
        self.thread = threading.Thread(target=self._vision_loop, daemon=False)
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        if self.sock is not None:
            self.sock.close()
        return self.thread

    def _prune_frames(self, frames):
        if len(frames) <= MAX_INCOMPLETE_FRAMES:
            return
        oldest_id = min(frames)
        del frames[oldest_id]

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(SOCKET_TIMEOUT_S)
        self.sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("Listening for camera frames...", flush=True)

        while self.is_running:
            try:
                packet, _addr = self.sock.recvfrom(65536)
            except TimeoutError:
                continue
            except OSError:
                break

            header = packet[:header_sz]
            payload = packet[header_sz:]

            frame_id, chunk_id, total_chunks, jpeg_size, _payload_size, sim_time_ns = (
                struct.unpack(header_format, header)
            )

            if frame_id not in frames:
                self._prune_frames(frames)
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns,
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            if len(frames[frame_id]["chunks"]) != total_chunks:
                continue

            jpeg_bytes = bytearray()
            frame_complete = True
            for i in range(total_chunks):
                if i not in frames[frame_id]["chunks"]:
                    print(f"Missing packet {i} in frame {frame_id}", flush=True)
                    frame_complete = False
                    break
                jpeg_bytes.extend(frames[frame_id]["chunks"][i])

            del frames[frame_id]
            if not frame_complete:
                continue

            img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is not None:
                self.process_frame(frame_id, image, sim_time_ns)
            else:
                print(f"Failed to decode frame: {frame_id}", flush=True)

    def process_frame(self, frame_id, img, sim_time_ns=0):
        height, width = img.shape[:2]
        received_at = time.time()

        self.data["camera"] = {
            "frame_id": frame_id,
            "sim_time_ns": sim_time_ns,
            "width": width,
            "height": height,
            "received_at": received_at,
        }
        self.data["gate_target"] = self._gate_filter.apply(detect_gate_target(img))
