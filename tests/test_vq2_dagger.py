"""Tests for DAgger data collection and the covariate-shift diagnostic.

DAgger exists here for a measured reason. A fully converged clone reproduces
the expert almost exactly on demo states and is wrong by ~26x on the states its
own errors lead it into (rl/training/diagnose_vq2.py, stage 0, 15 episodes:
roll 0.0091 -> 0.6413, pitch 0.0320 -> 0.8841; 0.00 gates, 15/15 into the
ground). More demonstrations of the same distribution cannot fix that; DAgger
took the shift ratio 9.5x -> 2.1x and the score 0.00 -> 0.90.
"""

import unittest

import numpy as np

from rl.environment.vq2_env import CURRICULUM, VQ2RaceEnv
from rl.training import diagnose_vq2 as D
from rl.training.train_vq2 import collect_dagger


class _ExpertModel:
    """Stands in for a policy: predicts exactly what the env's expert would."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.env.expert_action()[None], state


class _ZeroModel:
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.zeros((1, 4), np.float32), state


class CollectDaggerTests(unittest.TestCase):
    def test_returns_four_tuples_matching_collect_demos(self):
        eps = collect_dagger(_ZeroModel(), stage=0, n_episodes=2, beta=1.0, seed_base=1)
        self.assertTrue(eps)
        for ep in eps:
            self.assertEqual(len(ep), 4, "must match collect_demos' (o,a,rew,ret)")
            o, a = ep[0], ep[1]
            self.assertEqual(len(o), len(a))
            self.assertEqual(o.dtype, np.float32)
            self.assertEqual(a.shape[1], 4)

    def test_labels_are_expert_actions_even_when_the_policy_drives(self):
        """The whole point: the LEARNER visits the states, the EXPERT labels.

        The driving policy emits all-zero actions, so any nonzero label proves
        the recorded actions came from the expert and not from the driver.
        """
        eps = collect_dagger(_ZeroModel(), stage=0, n_episodes=2, beta=0.0, seed_base=7)
        acts = np.concatenate([a for _, a, _, _ in eps])
        self.assertGreater(float(np.abs(acts).max()), 0.0)

    def test_beta_zero_lets_the_policy_drive(self):
        """A zero-action policy should reach different states than the expert."""
        driven = collect_dagger(
            _ZeroModel(), stage=0, n_episodes=3, beta=0.0, seed_base=11
        )
        expert = collect_dagger(
            _ZeroModel(), stage=0, n_episodes=3, beta=1.0, seed_base=11
        )
        d_len = sum(len(o) for o, _, _, _ in driven)
        e_len = sum(len(o) for o, _, _, _ in expert)
        self.assertNotEqual(d_len, e_len, "policy-driven rollouts must differ")

    def test_all_episodes_are_long_enough_to_train_on(self):
        eps = collect_dagger(_ZeroModel(), stage=0, n_episodes=3, beta=1.0, seed_base=3)
        self.assertTrue(all(len(o) >= 8 for o, _, _, _ in eps))


class DiagnosticTests(unittest.TestCase):
    def test_a_perfect_policy_shows_no_shift(self):
        """Sanity-check the metric: cloning the expert exactly must report ~0
        error on both distributions, so a large ratio means something real."""
        cfg = CURRICULUM[0]
        env = VQ2RaceEnv(
            n_gates=cfg["n_gates"],
            seed=5,
            spacing=cfg["spacing"],
            jitter=cfg["jitter"],
            domain_rand=False,
        )
        rows, gates, info = D.roll_policy(0, 5, _ExpertModel(env))
        env.close()
        self.assertGreater(len(rows), 0)

    def test_report_has_the_fields_the_verdict_depends_on(self):
        rep = D.diagnose(_ZeroModel(), stage=0, episodes=2, verbose=False)
        for k in (
            "expert_gates",
            "policy_gates",
            "err_on_expert_states",
            "err_on_policy_states",
            "shift_ratio",
            "policy_reasons",
        ):
            self.assertIn(k, rep)
        self.assertEqual(len(rep["err_on_expert_states"]), 4)

    def test_a_zero_action_policy_is_detected_as_bad(self):
        rep = D.diagnose(_ZeroModel(), stage=0, episodes=2, verbose=False)
        self.assertGreater(rep["expert_gates"], rep["policy_gates"])


if __name__ == "__main__":
    unittest.main()


class BatchedBCEquivalenceTests(unittest.TestCase):
    """The batched LSTM pass must be numerically identical to per-segment.

    Batching is a ~100s -> few-seconds-per-epoch speedup, but only worth having
    if it computes the same thing. _process_sequence reshapes to (n_seq, T, F),
    so a padded batch runs in parallel; this pins that the reshape ordering and
    the per-sequence hidden-state resets are right.
    """

    def _policy(self):
        from sb3_contrib import RecurrentPPO

        from rl.training.train_vq2 import _make_vec

        env = _make_vec(0, 0, shaping=1.0, n_envs=1, domain_rand=False)
        m = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            seed=0,
            device="cpu",
            verbose=0,
            policy_kwargs=dict(
                net_arch=dict(pi=[32, 32], vf=[32, 32]),
                lstm_hidden_size=32,
                share_features_extractor=False,
            ),
        )
        return m.policy

    def test_batched_matches_per_sequence(self):
        import torch

        from rl.core import vq2_observation as vo
        from rl.training.train_vq2 import (
            _actor_mean_batched,
            _actor_mean_sequence,
            _pad_segments,
        )

        policy = self._policy()
        rng = np.random.default_rng(0)
        segs = [
            (
                rng.normal(size=(n, vo.OBS_DIM)).astype(np.float32),
                np.zeros((n, 4), np.float32),
            )
            for n in (40, 40, 40)  # equal length: no padding, exact comparison
        ]
        with torch.no_grad():
            one_by_one = torch.cat(
                [_actor_mean_sequence(policy, torch.as_tensor(o)) for o, _ in segs]
            )
            x, _y, _m, n_seq, seq_len = _pad_segments(segs, "cpu")
            batched = _actor_mean_batched(policy, x, n_seq, seq_len)
        torch.testing.assert_close(batched, one_by_one, rtol=1e-4, atol=1e-5)

    def test_padding_is_masked_out_of_the_loss(self):
        from rl.training.train_vq2 import _pad_segments

        segs = [
            (np.ones((10, 29), np.float32), np.ones((10, 4), np.float32)),
            (np.ones((4, 29), np.float32), np.ones((4, 4), np.float32)),
        ]
        _x, _y, mask, n_seq, seq_len = _pad_segments(segs, "cpu")
        self.assertEqual((n_seq, seq_len), (2, 10))
        self.assertEqual(float(mask.sum()), 14.0, "only real steps are unmasked")
