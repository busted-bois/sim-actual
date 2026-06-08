import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

from simulator.vision_processing import analyze_frame, annotate

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

# Key under which the latest FrameAnalysis is published into shared_data for the
# controller/navigator to read.
VISION_ANALYSIS_KEY = "vision_analysis"


class VisionRX:
    def __init__(self, data, config=None):
        self.data = data
        self.config = config or {}
        self.vision_cfg = self.config.get("vision")
        self.frame_count = 0
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
        #
        # image is the decoded FPV camera frame (BGR). Run color detection and
        # publish the result for the navigator to consume in the control loop.
        #
        if not self.vision_cfg:
            # Detection disabled / no config: behave like the original template.
            return

        analysis = analyze_frame(
            img, self.vision_cfg, frame_id=frame_id, timestamp=time.time()
        )
        # Single-assignment publish: the control loop reads this key; a plain dict
        # write of one reference is atomic enough in CPython for our purposes.
        self.data[VISION_ANALYSIS_KEY] = analysis

        self.frame_count += 1
        self._maybe_save_debug(img, analysis)

    def _maybe_save_debug(self, img, analysis):
        every = self.vision_cfg.get("debug_save_every", 0)
        if not every or self.frame_count % every != 0:
            return
        debug_dir = self.vision_cfg.get("debug_dir", "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"frame_{analysis.frame_id:06d}.jpg")
        cv2.imwrite(out_path, annotate(img, analysis))
