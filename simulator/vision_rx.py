import socket
import struct
import threading

import cv2
import numpy as np

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

# Overlay colors (BGR).
_GATE_COLOR = (0, 200, 0)
_OBSTACLE_COLOR = (0, 0, 255)


def _annotate(img, detection, obstacle_px):
    """Return a copy of img with the gate detection + obstacles drawn, plus a
    one-line HUD. Consumed by simulator.display for the live vision window."""
    out = img.copy()
    if detection is not None:
        cx, cy = int(detection.centroid_x_px), int(detection.centroid_y_px)
        x0 = int(cx - detection.width_px / 2.0)
        y0 = int(cy - detection.height_px / 2.0)
        x1 = int(cx + detection.width_px / 2.0)
        y1 = int(cy + detection.height_px / 2.0)
        cv2.rectangle(out, (x0, y0), (x1, y1), _GATE_COLOR, 2)
        cv2.circle(out, (cx, cy), 4, _GATE_COLOR, -1)
        hud = f"GATE cx={cx} cy={cy} area={detection.area_px:.0f}"
    else:
        hud = "no gate"
    for ocx, ocy in obstacle_px:
        cv2.circle(out, (int(ocx), int(ocy)), 6, _OBSTACLE_COLOR, 2)
    cv2.putText(
        out,
        hud,
        (10, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _GATE_COLOR,
        2,
        cv2.LINE_AA,
    )
    return out


class VisionRX:
    def __init__(self, data):
        self.data = data
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
                    self.process_frame(frame_id, image, sim_time_ns)
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

    def process_frame(self, frame_id, img, sim_time_ns=0):
        try:
            import time as _time

            from simulator.countdown_detector import countdown_visible, update_countdown_gate
            from simulator.gate_detector import detect_gate

            h, w = img.shape[:2]

            update_countdown_gate(self.data, countdown_visible(img))
            detection = detect_gate(img, frame_id, sim_time_ns)

            self.data["camera"] = {"received_at": _time.monotonic()}
            # Raw BGR frame for dataset generation (Module 2) / GateNet inference.
            self.data["frame"] = {
                "img": img,
                "frame_id": frame_id,
                "sim_time_ns": sim_time_ns,
                "received_at": _time.monotonic(),
            }

            if detection is not None:
                nx = (detection.centroid_x_px - w / 2.0) / (w / 2.0)
                ny = (detection.centroid_y_px - h / 2.0) / (h / 2.0)
                r_frac = detection.area_px / (w * h)
                self.data["gate_target"] = {
                    "detected": True,
                    "nx": nx,
                    "ny": ny,
                    "r_frac": r_frac,
                }
                print(
                    f"[vision] GATE cx={detection.centroid_x_px:.0f} cy={detection.centroid_y_px:.0f} "
                    f"area={detection.area_px:.0f} nx={nx:+.3f} ny={ny:+.3f}",
                    flush=True,
                )
            else:
                self.data["gate_target"] = {
                    "detected": False,
                    "nx": 0.0,
                    "ny": 0.0,
                    "r_frac": 0.0,
                }

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, obs_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            obs_mask[gray > 80] = 0  # exclude gate orange (~100+) and bright objects
            if detection is not None:
                cv2.circle(
                    obs_mask,
                    (int(detection.centroid_x_px), int(detection.centroid_y_px)),
                    int(max(detection.width_px, detection.height_px)),
                    0,
                    -1,
                )
            obs_contours, _ = cv2.findContours(
                obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            obstacles = []
            obstacle_px = []  # (cx, cy) in pixels, for the live overlay
            for oc in obs_contours:
                oa = cv2.contourArea(oc)
                if oa < 200:
                    continue
                om = cv2.moments(oc)
                om00 = max(om["m00"], 1e-6)
                ocx = om["m10"] / om00
                ocy = om["m01"] / om00
                onx = (ocx - w / 2.0) / (w / 2.0)
                ony = (ocy - h / 2.0) / (h / 2.0)
                orf = oa / (w * h)
                obstacles.append({"nx": onx, "ny": ony, "r_frac": orf})
                obstacle_px.append((ocx, ocy))
            self.data["obstacles"] = obstacles

            # Annotated copy for the live display window (drawn here, next to
            # detection, so the main/control thread just shows the result).
            self.data["frame"]["annotated"] = _annotate(img, detection, obstacle_px)
        except Exception as e:
            from simulator import config

            if config.DEBUG:
                print(f"[vision_rx] process_frame error: {e}")
