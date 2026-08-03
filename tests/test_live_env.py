"""Tests for the LIVE-simulator RL environment and trainer.

The env talks to a real simulator, so these tests substitute a fake MAVLink
stack (fake Controller + shared data dict + fake preflight) and assert the
contract that matters: that the env drives the real Controller correctly, that
reward and termination come only from signals VQ2 actually delivers, and that
nothing reads blocked telemetry.

Two of these lock in bugs that were live in the first draft of live_env.py:
Controller defaults to "motor" mode (which silently ignores attitude-rate
setpoints), and there is no send_control_command() -- Controller.update() is
what transmits and paces the loop.
"""

import types
import unittest
from unittest import mock

import numpy as np

from rl.core import vq2_observation as vo


class _FakeController:
    """Records what a real Controller would have been asked to do."""

    def __init__(self, data):
        self.data = data
        self.control_mode = "motor"
        self.control_hz = 90
        self.pilot = None
        self.rates = []
        self.updates = 0
        self.arms = 0
        self.resets = 0
        self._thrust = 0.0

    def set_control_mode(self, mode):
        self.control_mode = mode

    def set_attitude_rates(self, roll, pitch, yaw, thrust):
        self.rates.append((roll, pitch, yaw, thrust))
        self._thrust = thrust

    def update(self):
        self.updates += 1
        if self.pilot is not None:
            self.pilot.tick()

    def arm(self):
        self.arms += 1
        self.data["armed"] = True

    def send_sim_reset_command(self):
        self.resets += 1


def _fake_setup(data, boot_ms, ip, port):
    # REAL key names, as published by simulator/mavlink_rx.py:215-228. An
    # earlier fixture used xacc/xgyro and masked a live bug: the env read those
    # names, got 0.0 for everything, and ran with a dead gyro and a frozen
    # gravity vector -- 6 of 29 observation dims silently constant.
    # Values are the measured pad attitude: |a|=9.8097 at 17.80 deg nose-down.
    data.setdefault(
        "imu",
        {
            "ax": -2.999,
            "ay": -0.002,
            "az": -9.340,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "time_us": 0,
        },
    )
    data.setdefault(
        "race_status",
        {
            "sim_boot_time_ms": 1000,
            "race_start_boot_time_ms": 900,
            "race_finish_time_ns": -1,
        },
    )
    data.setdefault("active_gate_index", 0)
    return {
        "controller": _FakeController(data),
        "ts_loop": None,
        "mavlink_rx": None,
        "vision_rx": None,
    }


class _PatchLive:
    """Patch the live boundary: setup_components + every preflight wait.

    Patches the ATTRIBUTES the code actually resolves, not sys.modules. An
    earlier version patched sys.modules and passed in isolation but blocked the
    full suite forever: "from simulator import preflight" reads the package
    attribute, so once any other test had imported the real module the fake was
    ignored and the env sat waiting on a simulator that was not running.
    """

    def __init__(self, finished=False):
        self.pf = types.SimpleNamespace(
            wait_for_session_ready=lambda data, timeout_s=0: True,
            wait_for_race_status=lambda data, timeout_s=0: True,
            wait_for_fresh_race_start_vq2=(
                lambda data, timeout_s=0, is_restart=False: True
            ),
            wait_for_race_go=lambda data, timeout_s=0, armed_sim_boot_ms=None: True,
            race_finished=lambda data: finished,
        )
        self._patches = []

    def __enter__(self):
        from rl.environment import live_env

        self._patches = [
            mock.patch.object(live_env, "preflight", self.pf),
            mock.patch("simulator.setup.setup_components", _fake_setup),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _patch_live(finished=False):
    return _PatchLive(finished=finished)


def _mk_env(**kw):
    from rl.environment.live_env import LiveVQ2Env

    kw.setdefault("verbose", False)
    kw.setdefault("control_hz", 1000.0)  # keep tests fast; pacing is Controller's
    return LiveVQ2Env(**kw)


class WiringTests(unittest.TestCase):
    def test_controller_is_switched_to_attitude_mode(self):
        """Controller defaults to 'motor', which ignores set_attitude_rates
        entirely -- the env's commands would be latched and never sent."""
        with _patch_live():
            env = _mk_env()
            self.assertEqual(env.controller.control_mode, "attitude")
            env.close()

    def test_step_transmits_via_controller_update(self):
        """There is no Controller.send_control_command(); update() is what
        transmits and paces. One env step must be exactly one update()."""
        with _patch_live():
            env = _mk_env()
            env.reset()
            before = env.controller.updates
            for _ in range(5):
                env.step(np.zeros(4, np.float32))
            self.assertEqual(env.controller.updates - before, 5)
            env.close()

    def test_reset_sends_sim_reset_and_arms(self):
        with _patch_live():
            env = _mk_env()
            env.reset()
            self.assertGreaterEqual(env.controller.resets, 1)
            self.assertGreaterEqual(env.controller.arms, 1)
            env.close()

    def test_action_is_clipped_and_sign_mapped_before_the_wire(self):
        with _patch_live():
            env = _mk_env()
            env.reset()
            env.step(np.array([5.0, -5.0, 5.0, 5.0], np.float32))
            roll, pitch, yaw, thrust = env.controller.rates[-1]
            from rl.environment import live_env as LE

            for v in (roll, pitch, yaw):
                self.assertLessEqual(abs(v), LE.RATE_CLIP + 1e-9)
            self.assertGreaterEqual(thrust, LE.THRUST_MIN)
            self.assertLessEqual(thrust, LE.THRUST_MAX)
            env.close()


class SpaceTests(unittest.TestCase):
    def test_observation_is_the_vq2_contract(self):
        with _patch_live():
            env = _mk_env()
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (vo.OBS_DIM,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(obs)))
            env.close()

    def test_action_space_is_four_normalized_channels(self):
        with _patch_live():
            env = _mk_env()
            self.assertEqual(env.action_space.shape, (4,))
            env.close()


class IMUPlumbingTests(unittest.TestCase):
    """MAVLinkRX publishes ax/ay/az/gx/gy/gz, not xacc/xgyro. Reading the wrong
    names returns 0.0 silently, which kills the gyro channel, freezes the AHRS
    at its seed, and zeroes 6 of the 29 observation dims."""

    def test_gyro_reaches_the_observation(self):
        with _patch_live():
            env = _mk_env()
            env.reset()
            quiet, *_ = env.step(np.zeros(4, np.float32))
            env.data["imu"] = dict(env.data["imu"], gx=1.5, gy=-0.9, gz=0.4)
            spun, *_ = env.step(np.zeros(4, np.float32))
            g_quiet = quiet[vo.OBS_LAYOUT["gyro"]]
            g_spun = spun[vo.OBS_LAYOUT["gyro"]]
            self.assertFalse(
                np.allclose(g_quiet, g_spun),
                "gyro must reach obs -- wrong IMU keys would leave it at zero",
            )
            self.assertGreater(float(np.abs(g_spun).max()), 0.0)
            env.close()

    def test_ahrs_seeds_from_the_measured_pad_attitude(self):
        with _patch_live():
            env = _mk_env()
            obs, _ = env.reset()
            env.step(np.zeros(4, np.float32))
            g = obs[vo.OBS_LAYOUT["gravity_body"]]
            self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0, places=4)
            self.assertGreater(g[0], 0.0, "nose-down puts gravity forward in body x")
            env.close()

    def test_attitude_integrates_when_the_gyro_spins(self):
        with _patch_live():
            env = _mk_env()
            env.reset()
            env.step(np.zeros(4, np.float32))
            before = env.ahrs.euler_deg()
            env.data["imu"] = dict(env.data["imu"], gx=2.0)
            for _ in range(40):
                env.step(np.zeros(4, np.float32))
            after = env.ahrs.euler_deg()
            self.assertNotAlmostEqual(before[0], after[0], places=3)
            env.close()


class RewardTerminationTests(unittest.TestCase):
    def test_gate_advance_from_race_status_pays_the_bonus(self):
        from rl.environment import live_env as LE

        with _patch_live():
            env = _mk_env()
            env.reset()
            env.data["active_gate_index"] = 3
            _obs, reward, _term, _trunc, info = env.step(np.zeros(4, np.float32))
            self.assertTrue(info.get("gate_passed"))
            self.assertGreater(reward, LE.GATE_BONUS * 2)  # three gates at once
            self.assertEqual(env.gates_passed, 3)
            env.close()

    def test_race_finished_terminates_with_the_completion_bonus(self):
        from rl.environment import live_env as LE

        with _patch_live(finished=True):
            env = _mk_env()
            env.reset()
            _obs, reward, term, _trunc, info = env.step(np.zeros(4, np.float32))
            self.assertTrue(term)
            self.assertTrue(info.get("course_complete"))
            self.assertGreater(reward, LE.COMPLETE_BONUS * 0.5)
            env.close()

    def test_stall_truncates_rather_than_terminating(self):
        """No gate progress is a failed attempt, not a terminal state of the
        MDP -- bootstrapping should continue, so it truncates."""
        from rl.environment import live_env as LE

        with _patch_live():
            env = _mk_env()
            env.reset()
            env._last_gate_t -= LE.STALL_S + 1.0
            _obs, _r, term, trunc, info = env.step(np.zeros(4, np.float32))
            self.assertFalse(term)
            self.assertTrue(trunc)
            self.assertEqual(info.get("crash"), "stall")
            env.close()

    def test_episode_time_limit_truncates(self):
        with _patch_live():
            env = _mk_env(max_episode_s=0.0)
            env.reset()
            _obs, _r, term, trunc, info = env.step(np.zeros(4, np.float32))
            self.assertFalse(term)
            self.assertTrue(trunc)
            env.close()


def _executable_source(module):
    """Module source with docstrings and comments removed.

    These guards must inspect CODE, not prose. The module deliberately explains
    in its docstring why it avoids odometry and the COLLISION stream, and a
    naive substring scan flags exactly the comment that documents the fix.
    Round-tripping through the AST drops comments outright and lets us strip
    docstrings explicitly.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


class BlockedTelemetryTests(unittest.TestCase):
    def test_env_never_reads_blocked_telemetry(self):
        """ODOMETRY / ATTITUDE / LOCAL_POSITION_NED are withheld under VQ2, and
        the previous live attempt died terminating on a dead-reckoned position
        (2,880 of 6,493 episodes ended on a bogus out_of_bounds)."""
        from rl.environment import live_env

        code = _executable_source(live_env)
        for key in ("odometry", "local_position_ned", "pos_ned", "snap.pos"):
            self.assertNotIn(key, code, f"live env must not read {key}")

    def test_collision_stream_is_not_used_as_a_crash_flag(self):
        """COLLISION fires 2.5-3 m from a gate frame -- i.e. on every correct
        approach to a 1.5 m opening. Using it as a crash flag punished the
        previous attempt for flying the course properly."""
        from rl.environment import live_env

        self.assertNotIn("collision", _executable_source(live_env))


class TrainerTests(unittest.TestCase):
    def test_seed_replay_buffer_fills_the_buffer(self):
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env

        from rl.training.train_live import seed_replay_buffer

        with _patch_live():
            env = _mk_env()
            try:
                model = SAC(
                    "MlpPolicy",
                    env,
                    buffer_size=100,
                    learning_starts=10,
                    device="cpu",
                    verbose=0,
                )
                obs = np.zeros(vo.OBS_DIM, np.float32)
                trans = [(obs, obs, np.zeros(4, np.float32), 1.0, False)] * 7
                n = seed_replay_buffer(model, trans)
                self.assertEqual(n, 7)
                self.assertEqual(model.replay_buffer.size(), 7)
            finally:
                env.close()
        del make_vec_env

    def test_seeding_an_empty_demo_list_is_a_noop(self):
        from rl.training.train_live import seed_replay_buffer

        self.assertEqual(seed_replay_buffer(object(), []), 0)


if __name__ == "__main__":
    unittest.main()
