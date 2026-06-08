from dataclasses import dataclass

from simulator.gate_detector import FRAME_HEIGHT, FRAME_WIDTH, LOCK_CONFIDENCE

DEBUG_NAVIGATION = False
DEBUG_LOG_HZ = 5.0
YAW_GAIN = 0.7
LATERAL_GAIN = 0.0
ENABLE_VERTICAL_CONTROL = False
SCAN_FORWARD_SPEED = 0.0
SCAN_YAW_RATE = 0.0

STALE_DETECTION_S = 0.15
LOST_FRAMES = 6
COAST_FRAMES = 10
RESET_HISTORY_FRAMES = 30
PASS_DISAPPEAR_FRAMES = 5
PASS_RANGE_M = 0.8
PASS_FRAME_RATIO = 0.60


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate: float


def clamp(value, low, high):
    return max(low, min(value, high))


class GateNavigator:
    def __init__(self):
        self.last_frame_id = None
        self.no_detect_frames = 0
        self.lost_frames = 0
        self.coast_frames = 0
        self.pass_missing_frames = 0
        self.near_pass_candidate = False
        self.last_command = VelocityCommand(SCAN_FORWARD_SPEED, 0.0, 0.0, 0.0)
        self.last_ex = 0.0
        self.scan_direction = 1.0
        self.last_scan_flip_s = 0.0
        self.last_log_s = 0.0
        self.last_loss_step_s = 0.0
        self.state = "search"

    def update(self, frame_id, detection, detection_age_s, now_s):
        fresh_frame = frame_id is not None and detection_age_s <= STALE_DETECTION_S
        if frame_id == self.last_frame_id and fresh_frame:
            command = self.last_command
        elif fresh_frame:
            self.last_frame_id = frame_id
            if detection is not None:
                self.no_detect_frames = 0
                command = self._track(detection)
            else:
                self.no_detect_frames += 1
                command = self._lost(now_s)
        else:
            if now_s - self.last_loss_step_s < 1.0 / 30.0:
                self._log(detection, self.last_command, now_s)
                return self.last_command
            self.last_loss_step_s = now_s
            self.no_detect_frames += 1
            command = self._lost(now_s)

        self._log(detection, command, now_s)
        self.last_command = command
        return command

    def _track(self, detection):
        self.lost_frames = 0
        self.coast_frames = 0
        self.pass_missing_frames = 0
        self.last_ex = detection.ex

        centered = abs(detection.ex) < 0.25 and abs(detection.ey) < 0.25
        if detection.strong_lock and centered and self._near_pass(detection):
            self.near_pass_candidate = True

        yaw_rate = clamp(YAW_GAIN * detection.ex, -0.6, 0.6)
        vy = clamp(LATERAL_GAIN * detection.ex, -0.05, 0.05)
        vz = clamp(0.8 * detection.ey, -0.8, 0.8) if ENABLE_VERTICAL_CONTROL else 0.0
        vx = self._forward_speed(detection, centered)

        self._set_state("track")
        return VelocityCommand(vx, vy, vz, yaw_rate)

    def _lost(self, now_s):
        self.lost_frames += 1

        if self.near_pass_candidate:
            self.pass_missing_frames += 1
            if self.pass_missing_frames >= PASS_DISAPPEAR_FRAMES:
                self.near_pass_candidate = False
                self._set_state("passed")

        if self.lost_frames <= LOST_FRAMES + COAST_FRAMES:
            self.coast_frames += 1
            self._set_state("coast")
            return VelocityCommand(
                max(self.last_command.vx * 0.5, SCAN_FORWARD_SPEED),
                self.last_command.vy * 0.5,
                0.0,
                self.last_command.yaw_rate * 0.5,
            )

        if self.lost_frames >= RESET_HISTORY_FRAMES:
            self.last_ex = 0.0

        self._set_state("scan")
        return self._scan(now_s)

    def _scan(self, now_s):
        if self.last_ex != 0.0:
            self.scan_direction = 1.0 if self.last_ex > 0.0 else -1.0
        elif now_s - self.last_scan_flip_s >= 2.0:
            self.scan_direction *= -1.0
            self.last_scan_flip_s = now_s
        return VelocityCommand(SCAN_FORWARD_SPEED, 0.0, 0.0, SCAN_YAW_RATE * self.scan_direction)

    def _forward_speed(self, detection, centered):
        if detection.confidence < LOCK_CONFIDENCE:
            return 0.05
        if detection.range_m < 1.0:
            return 0.08
        if detection.range_m < 2.0:
            return 0.12
        if detection.strong_lock and centered and detection.range_m > 5.0:
            return 0.25
        if detection.confidence >= 0.75 and centered:
            return 0.2
        return 0.1

    def _near_pass(self, detection):
        _, _, w, h = detection.bbox
        wide = w / FRAME_WIDTH > PASS_FRAME_RATIO or h / FRAME_HEIGHT > PASS_FRAME_RATIO
        return detection.range_m < PASS_RANGE_M or wide or detection.clipped

    def _set_state(self, state):
        if state != self.state and DEBUG_NAVIGATION:
            print(f"nav state={state}", flush=True)
        self.state = state

    def _log(self, detection, command, now_s):
        if not DEBUG_NAVIGATION or now_s - self.last_log_s < 1.0 / DEBUG_LOG_HZ:
            return
        self.last_log_s = now_s
        if detection is None:
            print(
                "nav state=%s det=none cmd=(%.2f %.2f %.2f %.2f)"
                % (self.state, command.vx, command.vy, command.vz, command.yaw_rate),
                flush=True,
            )
            return
        print(
            "nav state=%s conf=%.2f range=%.2f ex=%.2f ey=%.2f bbox=%s target=(%.0f,%.0f) cmd=(%.2f %.2f %.2f %.2f)"
            % (
                self.state,
                detection.confidence,
                detection.range_m,
                detection.ex,
                detection.ey,
                detection.bbox,
                detection.target_x,
                detection.target_y,
                command.vx,
                command.vy,
                command.vz,
                command.yaw_rate,
            ),
            flush=True,
        )
