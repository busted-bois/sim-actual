"""Tests for the VQ2 racing environment.

The env exists to fix one specific defect: rl/environment/env.py hands the
policy world position, world velocity and an absolute gate map, none of which
exist during a qualifier run. This env keeps the fast internal physics (bulk
RL needs ~1e7 steps and the live sim is capped at 50 steps/s) but RENDERS the
observation through the real camera model, so what the policy learns to read
is what the camera can actually deliver.
"""

import unittest

import numpy as np

from rl.core import vq2_observation as vo
from rl.environment import vq2_env


class SpacesTests(unittest.TestCase):
    def setUp(self):
        self.env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)

    def test_observation_space_matches_the_vq2_contract(self):
        self.assertEqual(self.env.observation_space.shape, (vo.OBS_DIM,))
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (vo.OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_action_is_four_normalized_channels(self):
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertTrue(np.all(self.env.action_space.low == -1.0))
        self.assertTrue(np.all(self.env.action_space.high == 1.0))

    def test_sb3_env_checker_passes(self):
        from stable_baselines3.common.env_checker import check_env

        check_env(vq2_env.VQ2RaceEnv(n_gates=17, seed=0), warn=True)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        def rollout(seed):
            env = vq2_env.VQ2RaceEnv(n_gates=17, seed=seed)
            obs, _ = env.reset(seed=seed)
            out = [obs.copy()]
            for _ in range(60):
                obs, r, term, trunc, _ = env.step(np.array([0.1, -0.1, 0.0, 0.2]))
                out.append(obs.copy())
                if term or trunc:
                    break
            return np.array(out)

        np.testing.assert_allclose(rollout(3), rollout(3))

    def test_different_seeds_give_different_courses(self):
        a = vq2_env.VQ2RaceEnv(n_gates=17, seed=1)
        b = vq2_env.VQ2RaceEnv(n_gates=17, seed=2)
        a.reset(seed=1)
        b.reset(seed=2)
        pa = np.array([g["pos"] for g in a.gates])
        pb = np.array([g["pos"] for g in b.gates])
        self.assertFalse(np.allclose(pa, pb))


class CourseTests(unittest.TestCase):
    def test_course_has_the_requested_gate_count(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)
        env.reset(seed=0)
        self.assertEqual(len(env.gates), 17)

    def test_gates_are_ordered_forward_along_the_course(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)
        env.reset(seed=0)
        d = [
            float(
                np.linalg.norm(
                    np.asarray(env.gates[i + 1]["pos"])
                    - np.asarray(env.gates[i]["pos"])
                )
            )
            for i in range(len(env.gates) - 1)
        ]
        self.assertTrue(all(x > 1.0 for x in d), "gates must not be coincident")


class RenderedObservationTests(unittest.TestCase):
    """The observation must come out of the camera model, not the gate map."""

    def test_gate_ahead_is_seen(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, detector_dropout=0.0)
        obs, _ = env.reset(seed=0)
        # Let the fixed-rate detector tick at least once.
        for _ in range(10):
            obs, *_ = env.step(np.zeros(4, dtype=np.float32))
        self.assertGreater(
            obs[vo.OBS_LAYOUT["conf"]][0], 0.0, "gate ahead should be detected"
        )
        d = obs[vo.OBS_LAYOUT["gate_dir_body"]]
        self.assertGreater(d[0], 0.0, "target gate is in front -> +body x")

    def test_gate_behind_is_not_seen(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, detector_dropout=0.0)
        env.reset(seed=0)
        # Spin 180 deg: the gate leaves the 90-deg FOV entirely.
        env.q = np.array([0.0, 0.0, 0.0, 1.0])  # yaw 180
        for _ in range(30):
            obs, *_ = env.step(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["detected"]][0], 0.0)

    def test_detector_runs_slower_than_the_control_loop(self):
        """Perception is 12-16 Hz against a 50 Hz loop; if every tick carried a
        fresh fix the policy would learn a cadence it never gets live."""
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, detector_dropout=0.0)
        env.reset(seed=0)
        fresh = 0
        for _ in range(50):
            obs, *_ = env.step(np.zeros(4, dtype=np.float32))
            fresh += int(obs[vo.OBS_LAYOUT["detected"]][0] > 0.5)
        self.assertLess(fresh, 40, "detector must not fire on every control tick")
        self.assertGreater(fresh, 0, "detector must fire sometimes")

    def test_dropout_is_actually_applied(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, detector_dropout=1.0)
        env.reset(seed=0)
        for _ in range(40):
            obs, *_ = env.step(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["detected"]][0], 0.0)

    def test_observation_never_leaks_absolute_position(self):
        """Translating the whole world must not change the observation: if it
        did, the policy would be reading a global frame it cannot have."""
        a = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, detector_dropout=0.0)
        obs_a, _ = a.reset(seed=0)
        b = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, detector_dropout=0.0)
        obs_b, _ = b.reset(seed=0)
        shift = np.array([137.0, -49.0, -18.0])
        b.p = b.p + shift
        for g in b.gates:
            g["pos"] = list(np.asarray(g["pos"], dtype=float) + shift)
        for _ in range(12):
            obs_a, *_ = a.step(np.zeros(4, dtype=np.float32))
            obs_b, *_ = b.step(np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(obs_a, obs_b, atol=1e-4)


class GateGeometryTests(unittest.TestCase):
    def test_passing_through_the_opening_advances_the_index(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)
        env.reset(seed=0)
        g = env.gates[0]
        env.p = np.asarray(g["pos"], dtype=float) - env._gate_normal(0) * 0.4
        env.v = env._gate_normal(0) * 8.0
        before = env.gate_idx
        for _ in range(5):
            _, _, term, _, info = env.step(np.zeros(4, dtype=np.float32))
            if info.get("gate_passed"):
                break
        self.assertGreater(env.gate_idx, before)

    def test_hitting_the_frame_terminates_as_a_crash(self):
        """The opening is 1.5 m inside a 2.7 m frame. Clipping the frame must
        end the episode -- otherwise the policy learns that grazing a gate is
        nearly free."""
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)
        env.reset(seed=0)
        n = env._gate_normal(0)
        right = env._gate_axes(0)[0]
        # Aim at the frame: outside the 0.75 m opening, inside the 1.35 m frame.
        env.p = np.asarray(env.gates[0]["pos"], dtype=float) - n * 0.4 + right * 1.05
        env.v = n * 8.0
        term = False
        for _ in range(5):
            _, _, term, _, info = env.step(np.zeros(4, dtype=np.float32))
            if term:
                break
        self.assertTrue(term)
        self.assertEqual(info.get("crash"), "gate_frame")

    def test_missing_the_gate_entirely_is_not_a_frame_strike(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)
        env.reset(seed=0)
        n = env._gate_normal(0)
        right = env._gate_axes(0)[0]
        env.p = np.asarray(env.gates[0]["pos"], dtype=float) - n * 0.4 + right * 6.0
        env.v = n * 8.0
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertNotEqual(info.get("crash"), "gate_frame")


class RewardTests(unittest.TestCase):
    def test_course_completion_pays_the_sparse_terminal_reward(self):
        env = vq2_env.VQ2RaceEnv(n_gates=2, seed=0)
        env.reset(seed=0)
        total = 0.0
        for _ in range(2):
            g = env.gates[env.gate_idx]
            env.p = (
                np.asarray(g["pos"], dtype=float) - env._gate_normal(env.gate_idx) * 0.4
            )
            env.v = env._gate_normal(env.gate_idx) * 8.0
            for _ in range(5):
                _, r, term, _, info = env.step(np.zeros(4, dtype=np.float32))
                total += r
                if info.get("gate_passed") or term:
                    break
            if term:
                break
        self.assertTrue(
            info.get("course_complete"), "should have finished a 2-gate course"
        )
        self.assertGreater(total, vq2_env.COMPLETE_BONUS * 0.5)

    def test_gate_bonus_survives_full_shaping_anneal(self):
        """Annealing dense shaping to zero must NOT delete the gate reward.
        With shaping=0 and no gate bonus, a 17-gate course has completion as
        its only signal and the policy just learns to hover."""
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, shaping=0.0)
        env.reset(seed=0)
        env.p = np.asarray(env.gates[0]["pos"], dtype=float) - env._gate_normal(0) * 0.4
        env.v = env._gate_normal(0) * 8.0
        got = 0.0
        for _ in range(5):
            _, r, _, _, info = env.step(np.zeros(4, dtype=np.float32))
            if info.get("gate_passed"):
                got = r
                break
        self.assertTrue(info.get("gate_passed"), "should have passed the gate")
        self.assertGreater(got, vq2_env.GATE_BONUS * 0.5)

    def test_shaping_weight_zero_leaves_only_sparse_terms(self):
        """The official objective is sparse terminal success. Shaping must be
        an annealable scaffold, not baked in."""
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, shaping=0.0)
        env.reset(seed=0)
        r_total = 0.0
        for _ in range(40):
            _, r, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
            r_total += r
            if term or trunc:
                break
        # With no shaping and no gate passed, only the time cost accrues.
        self.assertLessEqual(r_total, 0.0)

    def test_crashing_into_the_ground_is_penalized_and_terminates(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0)
        env.reset(seed=0)
        env.p = np.array([0.0, 0.0, 5.0])  # NED z down -> below ground
        _, r, term, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(term)
        self.assertEqual(info.get("crash"), "ground")
        self.assertLess(r, 0.0)


class SolvabilityTests(unittest.TestCase):
    """If the privileged expert cannot fly the course, the env is broken and no
    amount of RL will fix it. This is the env's own sanity gate."""

    def test_expert_clears_gates(self):
        cleared = []
        for seed in range(4):
            env = vq2_env.VQ2RaceEnv(n_gates=6, seed=seed, domain_rand=False)
            env.reset(seed=seed)
            term = trunc = False
            while not (term or trunc):
                _, _, term, trunc, _ = env.step(env.expert_action())
            cleared.append(env.gate_idx)
        self.assertGreaterEqual(
            float(np.mean(cleared)), 3.0, f"expert only cleared {cleared}"
        )


class DomainRandomizationTests(unittest.TestCase):
    def test_plant_constants_vary_across_resets(self):
        """env.py claimed domain randomization in its docstring and implemented
        none; every plant constant was fixed. That is the classic sim-to-real
        failure and it is what this asserts against."""
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, domain_rand=True)
        seen = set()
        for s in range(12):
            env.reset(seed=s)
            seen.add(
                (
                    round(env.thrust_accel, 6),
                    round(env.rate_tau, 6),
                    round(env.drag, 6),
                    round(env.rate_gain, 6),
                )
            )
        self.assertGreater(len(seen), 6, "plant must be randomized per episode")

    def test_randomization_can_be_disabled_for_evaluation(self):
        env = vq2_env.VQ2RaceEnv(n_gates=17, seed=0, domain_rand=False)
        vals = []
        for s in range(5):
            env.reset(seed=s)
            vals.append((env.thrust_accel, env.rate_tau, env.drag, env.rate_gain))
        self.assertEqual(len(set(vals)), 1)


if __name__ == "__main__":
    unittest.main()


class RandomizationCurriculumTests(unittest.TestCase):
    """dr_scale exists because the BC clone is brittle to exactly two plant
    parameters (rate_gain and thrust_accel each drop it from 1.00 to 0.00 on
    their own; the clone scores 0.37 on the full range). PPO must ramp into
    randomization rather than start inside it."""

    def test_scale_zero_is_exactly_nominal(self):
        seen = set()
        for k in range(8):
            env = vq2_env.VQ2RaceEnv(n_gates=1, seed=k, dr_scale=0.0)
            env.reset(seed=k)
            seen.add((env.rate_gain, env.thrust_accel, env.drag, env.rate_tau))
        self.assertEqual(len(seen), 1, "dr_scale=0 must pin the plant")

    def test_scale_widens_monotonically(self):
        def spread(scale):
            vals = []
            for k in range(24):
                env = vq2_env.VQ2RaceEnv(n_gates=1, seed=k, dr_scale=scale)
                env.reset(seed=k)
                vals.append(env.rate_gain)
            return max(vals) - min(vals)

        self.assertAlmostEqual(spread(0.0), 0.0)
        self.assertLess(spread(0.4), spread(1.0))

    def test_full_scale_still_brackets_the_nominal_plant(self):
        vals = []
        for k in range(40):
            env = vq2_env.VQ2RaceEnv(n_gates=1, seed=k, dr_scale=1.0)
            env.reset(seed=k)
            vals.append(env.rate_gain)
        self.assertLess(min(vals), 2.7)
        self.assertGreater(max(vals), 2.7)


class ObservationConsistencyAcrossStagesTests(unittest.TestCase):
    """The curriculum must vary DIFFICULTY, not the meaning of an input.

    gate_idx was normalized by the stage's gate count, so the same number meant
    different things per stage and did not match deployment (always a 17-gate
    course). Measured cost: BC on stage-0 demos alone scored 0.75, BC on all
    four stages mixed scored 0.00 -- four times the data, strictly worse.
    """

    def test_gate_index_normalizer_is_the_course_length_not_the_stage(self):
        for n in (1, 3, 8, 17):
            env = vq2_env.VQ2RaceEnv(n_gates=n, seed=0)
            env.reset(seed=0)
            self.assertEqual(env.tracker.n_gates, vq2_env.COURSE_GATES, f"n={n}")

    def test_same_gate_index_gives_same_feature_in_every_stage(self):
        vals = []
        for n in (3, 8, 17):
            env = vq2_env.VQ2RaceEnv(n_gates=n, seed=0, detector_dropout=0.0)
            env.reset(seed=0)
            env.gate_idx = 1
            obs = env._obs(None)
            vals.append(float(obs[vq2_env.vo.OBS_LAYOUT["gate_idx"]][0]))
        self.assertAlmostEqual(max(vals), min(vals), places=6)


class BPTTTruncationTests(unittest.TestCase):
    """Demo episodes run 125-2689 steps. Backprop through a whole one is
    untrainable: measured all-stage BC scored 0.00 with full-episode BPTT and
    0.40 at BPTT=128, with loss dropping 0.0029 -> 0.0008."""

    def test_long_episodes_are_split(self):
        from rl.training.train_vq2 import BPTT_LEN, truncate

        eps = [(np.zeros((1000, 4), np.float32), np.zeros((1000, 2), np.float32))]
        segs = truncate(eps)
        self.assertGreater(len(segs), 1)
        self.assertTrue(all(len(o) <= BPTT_LEN for o, _ in segs))

    def test_short_episodes_survive_intact(self):
        from rl.training.train_vq2 import truncate

        eps = [(np.zeros((40, 4), np.float32), np.zeros((40, 2), np.float32))]
        segs = truncate(eps)
        self.assertEqual(len(segs), 1)
        self.assertEqual(len(segs[0][0]), 40)

    def test_runt_tail_is_dropped(self):
        from rl.training.train_vq2 import truncate

        eps = [(np.zeros((130, 4), np.float32), np.zeros((130, 2), np.float32))]
        segs = truncate(eps, maxlen=128, minlen=8)
        self.assertEqual(len(segs), 1, "a 2-step tail is not a usable sequence")
