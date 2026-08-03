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


class GateSizeIsolationTests(unittest.TestCase):
    """spec.GATE_SIZE_M = 2.72 is applied elsewhere as the OPENING, which scales
    range by 1.81x -- the real opening is 1.5 m inside a 2.7 m frame (measured
    inner/outer width ratio 0.560 over 551 gates). VQ2 geometry is isolated
    from those legacy constants. The check reads the source as an AST so the
    comments explaining WHY the constants are absent are not mistaken for use.
    """

    VQ2_SOURCE_FILES = (
        "rl/core/vq2_observation.py",
        "rl/environment/vq2_env.py",
    )

    def test_vq2_source_never_references_legacy_gate_size(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        offenders = []
        for rel in self.VQ2_SOURCE_FILES:
            tree = ast.parse((root / rel).read_text(), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in (
                    "GATE_SIZE_M",
                    "GATE_HALF",
                ):
                    offenders.append(f"{rel}:{node.lineno} spec.{node.attr}")
                if isinstance(node, ast.Name) and node.id in (
                    "GATE_SIZE_M",
                    "GATE_HALF",
                ):
                    offenders.append(f"{rel}:{node.lineno} {node.id}")
        self.assertEqual(
            offenders,
            [],
            f"VQ2 geometry must not use legacy spec gate-size constants: {offenders}",
        )


class CurriculumSmokeTests(unittest.TestCase):
    """Every curriculum stage must build, reset and step a finite 29-D
    observation, and the gate ladder must be the competition course (1 -> 3 ->
    8 -> 17). A regression that breaks stage wiring or shortens the ladder is
    otherwise invisible until hours into a training run.
    """

    EXPECTED_GATES = [1, 3, 8, 17]

    def test_every_stage_smokes_with_the_competition_gate_counts(self):
        self.assertEqual(len(vq2_env.CURRICULUM), len(self.EXPECTED_GATES))
        for stage, want in enumerate(self.EXPECTED_GATES):
            env = vq2_env.make_env(stage=stage, seed=0)()
            try:
                obs, _ = env.reset(seed=0)
                self.assertEqual(obs.shape, (vo.OBS_DIM,))
                self.assertEqual(obs.dtype, np.float32)
                self.assertTrue(
                    np.all(np.isfinite(obs)), f"non-finite obs at stage {stage}"
                )
                self.assertEqual(
                    len(env.gates), want, f"stage {stage} gate count mismatch"
                )
                for _ in range(5):
                    obs, *_ = env.step(np.zeros(4, dtype=np.float32))
                    self.assertEqual(obs.shape, (vo.OBS_DIM,))
                    self.assertTrue(
                        np.all(np.isfinite(obs)),
                        f"non-finite obs stepping stage {stage}",
                    )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
