"""Gymnasium environment that steps the REAL simulator over MAVLink.

This is the environment. There is no surrogate in this path: every step sends a
real command to FlightSim over MAVLink and reads back real camera frames, real
IMU, and the real race status. A policy trained here is trained on the thing it
will be scored on.

What the env is allowed to see (measured live, 2026-08-01/02, Qualification):

    HIGHRES_IMU            114.0 Hz   gyro clean; mag/baro NaN
    ENCAPSULATED_DATA        4.4 Hz   race_status -> active_gate_index,
                                      race_finish_time_ns, last_gate_race_time
    camera UDP 5600         30.0 Hz   -> YOLO-pose -> PnP
    ODOMETRY/ATTITUDE/LOCAL_POSITION_NED   absent (requested, withheld)
    track_info                    0   no gate poses

So the observation is rl.core.vq2_observation's 29-D vector -- vision + gyro +
AHRS gravity + last action + race_status gate index -- and NOTHING else.

Reward and termination come from race_status, the only oracle VQ2 gives us.
Deliberately NOT used, both of which wrecked the previous live attempt:

  * dead-reckoned position for out-of-bounds. GPEstimation is pure integration
    with no correction and VQ2 accel is garbage under thrust; 2,880 of that
    attempt's 6,493 episodes ended on a bogus out_of_bounds.
  * COLLISION as a crash flag. It is a PROXIMITY stream that fires 2.5-3 m from
    a gate frame -- i.e. it fires on every correct approach to a 1.5 m opening.

Wall clock is the binding constraint here, not compute: the command wire is
capped below 100 Hz and the achievable control loop is ~30 Hz, so this env
yields tens of steps per second, not thousands. Use it with an OFF-POLICY
learner that reuses every transition (see rl/training/train_live.py); on-policy
PPO discards each sample after one update and cannot pay for itself at this
price.

    uv run -m rl.environment.live_env --smoke
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# preflight is dependency-free (os/socket/time), so it is imported at module
# level: that makes it patchable as rl.environment.live_env.preflight. Patching
# sys.modules is NOT enough -- "from simulator import preflight" resolves the
# package ATTRIBUTE, which sys.modules patching leaves untouched, so a test
# suite that imported the real module first would silently call it and block
# forever waiting on a live simulator. setup_components stays lazy because it
# pulls in VisionRX -> torch.
from simulator import preflight

from rl.core import spec
from rl.core import vq2_observation as vo

# Control rate. The spec caps the command wire below 100 Hz and the measured
# achievable loop (pilot work + vision in-band) is 28-32 Hz, so asking for more
# just produces a loop that silently misses its deadline.
CONTROL_HZ = 30.0
COURSE_GATES = 17

# Reward. Gate passage is an oracle event from race_status; completion is the
# competition's actual objective.
GATE_BONUS = 10.0
COMPLETE_BONUS = 100.0
CRASH_PENALTY = 10.0
TIME_PENALTY = 0.02

# Episode limits.
MAX_EPISODE_S = 90.0
STALL_S = 12.0  # no gate progress for this long -> end the episode
FLIP_GRAVITY_Z = 0.0  # gravity's body-z component below this = inverted

# Reset lifecycle timeouts (seconds).
RESET_SETTLE_S = 0.5
READY_TIMEOUT_S = 20.0
RACE_START_TIMEOUT_S = 30.0
RACE_GO_TIMEOUT_S = 15.0
# YOLO cold start. MEASURED on a live observe run (2026-08-03): the first gate
# detection appeared 14.4 s after process start -- torch/CUDA init plus the
# 16.6 MB weights load -- while the camera stream was flowing the whole time.
# preflight.wait_for_session_ready keys on CAMERA FRAMES, so it returns
# immediately and does NOT cover this. Any episode begun inside that window
# flies completely blind on a 29-D observation whose vision half is empty.
VISION_WARMUP_TIMEOUT_S = 45.0

# Command shaping. UNCALIBRATED: flightlab/calibration.json has never been
# produced on this machine, so these are defaults, not measurements. They are
# the same knobs rl/deploy_vq2.py exposes, kept identical so a policy trained
# here flies the same way when deployed.
SIGN_ROLL, SIGN_PITCH, SIGN_YAW = -1.0, +1.0, -1.0
RATE_GAIN = 2.7
GYRO_SIGN = -1.0
RATE_CLIP = 1.2
THRUST_MIN, THRUST_MAX = 0.05, 0.85


# MAVLinkRX publishes the IMU as ax/ay/az/gx/gy/gz (simulator/mavlink_rx.py:215-228),
# NOT xacc/xgyro. Reading the wrong names returns 0.0 silently: the gyro goes
# dead, the AHRS never seeds or integrates, gravity_body freezes at its seed,
# and 6 of the 29 observation dims become constants. Fallbacks kept to match
# simulator/gp_estimation.py.
def _gravity_from_accel(imu):
    if not imu:
        return None
    a = np.array(
        [
            imu.get("ax", imu.get("xacc", 0.0)),
            imu.get("ay", imu.get("yacc", 0.0)),
            imu.get("az", imu.get("zacc", 0.0)),
        ],
        float,
    )
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-3:
        return None
    return -a / n  # at rest the accelerometer reads -g


class LiveVQ2Env(gym.Env):
    """One live simulator, wrapped as a Gymnasium env.

    Not vectorizable: there is exactly one FlightSim instance and one MAVLink
    endpoint. Construct it once and keep it for the whole training session --
    setup_components starts receiver threads and binds UDP 14550.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_hz: float = CONTROL_HZ,
        max_episode_s: float = MAX_EPISODE_S,
        n_gates: int = COURSE_GATES,
        ip: str = "127.0.0.1",
        port: int = 14550,
        verbose: bool = True,
        vision_warmup_s: float = VISION_WARMUP_TIMEOUT_S,
    ):
        super().__init__()
        self.vision_warmup_s = float(vision_warmup_s)
        from simulator.gyro_ahrs import GyroAHRS
        from simulator.setup import setup_components

        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(
            -vo.OBS_ABS_MAX, vo.OBS_ABS_MAX, (vo.OBS_DIM,), np.float32
        )

        self.period = 1.0 / float(control_hz)
        self.max_episode_s = float(max_episode_s)
        self.n_gates = int(n_gates)
        self.verbose = verbose

        self.data: dict = {}
        boot_ms = int(time.time() * 1000)
        self._components = setup_components(self.data, boot_ms, ip, port)
        self.controller = self._components["controller"]
        self.controller.control_hz = control_hz
        # Controller defaults to "motor" mode, which ignores set_attitude_rates
        # entirely -- the commands would be latched and never transmitted.
        self.controller.set_control_mode("attitude")
        # The Controller drives whatever pilot it holds; this env issues the
        # commands itself, so give it an inert pilot and never call update().
        self.controller.pilot = _NullPilot()

        self._GyroAHRS = GyroAHRS
        self.tracker = vo.GateFeatureTracker(n_gates=self.n_gates)
        self.ahrs = GyroAHRS(initial_pitch_deg=-17.8)
        self._seeded = False
        self._last_imu_t = None
        self._last_action = np.zeros(4)
        self._t0 = 0.0
        self._last_gate = 0
        self._last_gate_t = 0.0
        self._episode = 0
        self.gates_passed = 0

    # -- sensing ---------------------------------------------------------
    def _gyro(self):
        imu = self.data.get("imu") or {}
        g = np.array(
            [
                imu.get("gx", imu.get("xgyro", 0.0)),
                imu.get("gy", imu.get("ygyro", 0.0)),
                imu.get("gz", imu.get("zgyro", 0.0)),
            ],
            dtype=float,
        )
        return GYRO_SIGN * np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    def _gravity(self):
        """Gravity in body from the gyro-integrated AHRS.

        Accel is only trustworthy at rest -- thrust corrupts specific force --
        so it seeds attitude once and the gyro carries it. dt comes from
        ARRIVAL time because HIGHRES_IMU.time_usec is frozen on this sim.
        """
        imu = self.data.get("imu") or {}
        now = time.monotonic()
        dt = 0.0 if self._last_imu_t is None else max(now - self._last_imu_t, 0.0)
        self._last_imu_t = now

        if not self._seeded:
            g = _gravity_from_accel(imu)
            if g is not None and abs(float(np.linalg.norm(g)) - 1.0) < 0.2:
                import math

                # gravity in body for ZYX euler is
                #     [-sin(pitch), sin(roll)cos(pitch), cos(roll)cos(pitch)]
                # so pitch takes -g[0] and roll takes +g[1]. Getting these
                # backwards seeds the AHRS nose-UP 17.8 deg when the drone
                # actually sits nose-DOWN 17.8 deg on the pad -- a 35 deg
                # attitude error before the first command is ever sent.
                pitch = math.degrees(math.atan2(-g[0], g[2]))
                roll = math.degrees(math.atan2(g[1], g[2]))
                self.ahrs = self._GyroAHRS(
                    initial_pitch_deg=pitch, initial_roll_deg=roll
                )
                self._seeded = True

        gx, gy, gz = self._gyro()
        if 0.0 < dt < 0.5:
            self.ahrs.update(gx, gy, gz, dt)
        w, x, y, z = self.ahrs.q
        return np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )

    def _vision(self):
        from simulator.gp_vision import best_pose_gate

        g = best_pose_gate(self.data)
        if not g:
            return None
        p = g.get("pose") or {}
        if "gate_pos_body" not in p:
            return None
        est = dict(p)
        est["conf"] = float(g.get("conf", 0.0))
        return est

    def _gate_index(self) -> int:
        return int(self.data.get("active_gate_index", 0) or 0)

    def _obs(self):
        return self.tracker.update(
            time.monotonic(),
            self._vision(),
            self._gyro(),
            self._gravity(),
            self._last_action,
            self._gate_index(),
        )

    def wait_for_vision(self, timeout_s: float | None = None) -> bool:
        """Block until the detector produces its FIRST real gate estimate.

        Called once, before the first episode. Camera frames are not enough --
        YOLO needs ~14 s of cold start after them (measured), and an episode
        started before that has no vision at all.
        """
        timeout_s = self.vision_warmup_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout_s
        said = False
        while time.monotonic() < deadline:
            if self._vision() is not None:
                self._say(
                    f"vision ready after {timeout_s - (deadline - time.monotonic()):.1f}s"
                )
                return True
            if not said:
                self._say("waiting for the detector to warm up (YOLO cold start ~14s)")
                said = True
            time.sleep(0.1)
        return False

    # -- lifecycle -------------------------------------------------------
    def _say(self, msg):
        if self.verbose:
            print(f"[live] {msg}", flush=True)

    def reset(self, *, seed=None, options=None):
        """Teleport the drone, re-arm, and wait for the race countdown to hit GO.

        This is the expensive part of a live episode. Every wait has a bounded
        timeout so a stuck simulator surfaces as an exception instead of hanging
        an overnight training run forever.
        """
        # First reset only: pay the detector's cold start before flying, not
        # during episode 1.
        if self._episode == 0 and not self.wait_for_vision():
            raise RuntimeError(
                "live reset: detector never produced a gate estimate "
                f"within {self.vision_warmup_s:.1f}s -- is a gate in view?"
            )

        self._episode += 1
        self._say(f"episode {self._episode}: resetting sim")

        # 1. Teleport back to the pad. Sent twice 0.5 s apart, matching
        #    simulator/auto_flight.py -- a single command is occasionally lost.
        self.controller.send_sim_reset_command()
        time.sleep(RESET_SETTLE_S)
        self.controller.send_sim_reset_command()
        time.sleep(RESET_SETTLE_S)

        # 2. Telemetry + camera flowing again.
        if not preflight.wait_for_session_ready(self.data, timeout_s=READY_TIMEOUT_S):
            raise RuntimeError("live reset: session not ready (sim running?)")
        if not preflight.wait_for_race_status(self.data, timeout_s=READY_TIMEOUT_S):
            raise RuntimeError("live reset: no race_status after reset")

        # 3. Arm, then wait for a FRESH race start and the countdown to reach GO.
        self.controller.arm()
        armed_boot = (self.data.get("race_status") or {}).get("sim_boot_time_ms")
        if not preflight.wait_for_fresh_race_start_vq2(
            self.data, timeout_s=RACE_START_TIMEOUT_S, is_restart=True
        ):
            raise RuntimeError("live reset: no fresh race_start")
        if not preflight.wait_for_race_go(
            self.data, timeout_s=RACE_GO_TIMEOUT_S, armed_sim_boot_ms=armed_boot
        ):
            raise RuntimeError("live reset: race never reached GO")

        # 4. Episode state.
        self.tracker.reset()
        self.ahrs = self._GyroAHRS(initial_pitch_deg=-17.8)
        self._seeded = False
        self._last_imu_t = None
        self._last_action = np.zeros(4)
        self._t0 = time.monotonic()
        self._last_gate = self._gate_index()
        self._last_gate_t = self._t0
        self.gates_passed = 0
        self._say(f"episode {self._episode}: GO")
        return self._obs(), {}

    # -- stepping --------------------------------------------------------
    def _send(self, action):
        """Latch the setpoint. Controller.update() is what puts it on the wire."""
        roll, pitch, yaw, thrust = spec.scale_action(action)
        roll = float(np.clip(SIGN_ROLL * roll / RATE_GAIN, -RATE_CLIP, RATE_CLIP))
        pitch = float(np.clip(SIGN_PITCH * pitch / RATE_GAIN, -RATE_CLIP, RATE_CLIP))
        yaw = float(np.clip(SIGN_YAW * yaw / RATE_GAIN, -RATE_CLIP, RATE_CLIP))
        thrust = float(np.clip(thrust, THRUST_MIN, THRUST_MAX))
        self.controller.set_attitude_rates(roll, pitch, yaw, thrust)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1, 1)
        self._send(action)
        self._last_action = action

        # Controller.update() ticks the (inert) pilot, transmits the latched
        # SET_ATTITUDE_TARGET, re-arms if the sim disarmed us, and sleeps
        # 1/control_hz. That sleep IS the env's real-time pacing -- there is no
        # separate send call on Controller, and no second sleep here.
        self.controller.update()

        obs = self._obs()
        reward, terminated, truncated, info = self._outcome()
        info["gates_passed"] = self.gates_passed
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _outcome(self):
        """Reward and termination from race_status + AHRS only.

        Every signal here is one VQ2 actually delivers. Nothing reads a
        dead-reckoned position or the COLLISION proximity stream.
        """
        info: dict = {}
        now = time.monotonic()
        reward = -TIME_PENALTY

        idx = self._gate_index()
        if idx > self._last_gate:
            passed = idx - self._last_gate
            reward += GATE_BONUS * passed
            self._last_gate = idx
            self.gates_passed = idx
            self._last_gate_t = now
            info["gate_passed"] = True
            self._say(f"gate {idx} at {now - self._t0:.1f}s")

        if preflight.race_finished(self.data):
            info["course_complete"] = True
            self._say(f"COURSE COMPLETE in {now - self._t0:.1f}s")
            return reward + COMPLETE_BONUS, True, False, info

        # Inverted: gravity's body-z goes negative. Gyro-integrated attitude
        # drifts, but a full flip is far larger than the drift.
        gz = float(self._gravity()[2])
        if gz < FLIP_GRAVITY_Z:
            info["crash"] = "flipped"
            self._say("flipped")
            return reward - CRASH_PENALTY, True, False, info

        if (now - self._last_gate_t) > STALL_S:
            info["crash"] = "stall"
            self._say(f"stalled ({STALL_S:.0f}s without a gate)")
            return reward - CRASH_PENALTY, False, True, info

        if (now - self._t0) > self.max_episode_s:
            info["timeout"] = True
            return reward, False, True, info

        return reward, False, False, info

    def close(self):
        try:
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        for key in ("ts_loop", "mavlink_rx", "vision_rx"):
            c = (self._components or {}).get(key)
            if c is not None and hasattr(c, "get_thread_for_join"):
                try:
                    c.get_thread_for_join().join(timeout=2.0)
                except Exception:
                    pass


class _NullPilot:
    """Inert pilot so Controller holds a valid object; this env never uses it."""

    def tick(self):
        return None


def _smoke():
    """Connect, reset once, fly 100 zero-thrust-ish steps, report. Needs a live sim."""
    env = LiveVQ2Env()
    try:
        obs, _ = env.reset()
        print(f"[smoke] obs {obs.shape} finite={np.all(np.isfinite(obs))}")
        t0 = time.monotonic()
        n = 0
        for _ in range(100):
            obs, r, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0, -0.5]))
            n += 1
            if term or trunc:
                break
        dt = time.monotonic() - t0
        print(f"[smoke] {n} steps in {dt:.1f}s = {n / dt:.1f} steps/s")
        print(f"[smoke] gates={env.gates_passed} info={info}")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    _smoke()
