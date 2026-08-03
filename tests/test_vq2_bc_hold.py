"""Tests for VQ2 BC→PPO handoff: critic fitting, actor preservation, pre-solved gate."""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from rl.environment.vq2_env import VQ2RaceEnv
from rl.training.train_vq2 import (
    GAMMA,
    _actor_mean_sequence,
    _critic_params,
    _make_ppo,
    _make_vec,
    _stage_solved,
    _train_stage,
    collect_demos,
    discounted_returns,
    evaluate,
    fit_critic,
)


class DiscountedReturnsTests(unittest.TestCase):
    def test_single_step_returns_equal_reward(self):
        r = np.array([5.0])
        ret = discounted_returns(r, gamma=0.99)
        np.testing.assert_allclose(ret, [5.0], atol=1e-6)

    def test_two_steps(self):
        r = np.array([1.0, 2.0])
        ret = discounted_returns(r, gamma=0.99)
        expected = np.array([1.0 + 0.99 * 2.0, 2.0])
        np.testing.assert_allclose(ret, expected, atol=1e-5)

    def test_gamma_0_995_matches_ppo(self):
        r = np.array([0.0, 0.0, 10.0])
        ret = discounted_returns(r, gamma=0.995)
        np.testing.assert_allclose(ret[2], 10.0, atol=1e-6)
        np.testing.assert_allclose(ret[1], 9.95, atol=1e-5)
        self.assertAlmostEqual(float(ret[0]), 0.995 * 9.95, places=4)

    def test_output_is_float32(self):
        r = np.array([1.0, 2.0, 3.0])
        ret = discounted_returns(r)
        self.assertEqual(ret.dtype, np.float32)

    def test_output_is_finite(self):
        r = np.array([-10.0, 100.0, -50.0, 200.0])
        ret = discounted_returns(r)
        self.assertTrue(np.all(np.isfinite(ret)))

    def test_long_sequence_stays_finite(self):
        rng = np.random.default_rng(0)
        r = rng.uniform(-10, 10, 500).astype(np.float32)
        ret = discounted_returns(r)
        self.assertTrue(np.all(np.isfinite(ret)))
        self.assertEqual(ret.dtype, np.float32)

    def test_nan_rewards_raises(self):
        with self.assertRaises(ValueError):
            discounted_returns(np.array([1.0, np.nan, 3.0]))

    def test_inf_rewards_raises(self):
        with self.assertRaises(ValueError):
            discounted_returns(np.array([1.0, np.inf, 3.0]))

    def test_gamma_negative_raises(self):
        with self.assertRaises(ValueError):
            discounted_returns(np.array([1.0, 2.0]), gamma=-0.1)

    def test_gamma_above_one_raises(self):
        with self.assertRaises(ValueError):
            discounted_returns(np.array([1.0, 2.0]), gamma=1.5)


class CollectDemosTests(unittest.TestCase):
    def test_demo_shape_and_returns(self):
        episodes = collect_demos(episodes_per_stage=3, seed=0)
        self.assertGreater(len(episodes), 0)
        obs, act, rew, rets = episodes[0]
        T = obs.shape[0]
        self.assertEqual(obs.ndim, 2)
        self.assertEqual(act.ndim, 2)
        self.assertEqual(rew.ndim, 1)
        self.assertEqual(rets.ndim, 1)
        self.assertEqual(obs.shape[0], T)
        self.assertEqual(act.shape[0], T)
        self.assertEqual(len(rew), T)
        self.assertEqual(len(rets), T)
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(act.dtype, np.float32)

    def test_returns_match_gamma_discounted_rewards(self):
        episodes = collect_demos(episodes_per_stage=3, seed=0)
        for obs, act, rew, rets in episodes[:5]:
            expected = discounted_returns(rew, GAMMA)
            np.testing.assert_allclose(rets, expected, atol=1e-5)

    def test_demos_have_positive_rewards(self):
        episodes = collect_demos(episodes_per_stage=5, seed=0)
        for obs, act, rew, rets in episodes:
            total = float(rew.sum())
            self.assertGreater(
                total, -20.0, "expert demos should not have huge negative reward"
            )

    def test_collect_demos_with_runnerlog(self):
        buf = io.StringIO()
        from rl.core.diagnostics import RunnerLog
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log = RunnerLog(tag="test-demos", log_dir=tmp, stream=buf)
            collect_demos(episodes_per_stage=2, seed=0, log=log)
            log.close()
        text = buf.getvalue()
        self.assertIn("[demos]", text)
        self.assertIn("transitions", text)


class CriticFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)
        cls.obs_dim = cls.env.observation_space.shape[0]
        cls.act_dim = cls.env.action_space.shape[0]
        cls.env.close()

    def _make_model(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)])
        m = _make_ppo(env, seed=0, device="cpu")
        return m, env

    def test_critic_mse_decreases(self):
        m, env = self._make_model()
        policy = m.policy
        device = "cpu"

        T = 20
        rng = np.random.default_rng(42)
        obs_np = rng.uniform(-1, 1, (T, self.obs_dim)).astype(np.float32)
        rewards = np.ones(T, dtype=np.float32)
        returns = discounted_returns(rewards, GAMMA)

        obs_t = torch.as_tensor(obs_np, device=device)
        rets_t = torch.as_tensor(returns, device=device)

        def _critic_mse():
            h = torch.zeros(1, 1, policy.lstm_critic.hidden_size)
            c = h.clone()
            starts = torch.zeros(T)
            starts[0] = 1.0
            with torch.no_grad():
                vals = policy.predict_values(obs_t, (h, c), starts)
            return float(torch.nn.functional.mse_loss(vals.squeeze(-1), rets_t))

        mse_before = _critic_mse()

        episodes = [
            (obs_np, np.zeros((T, self.act_dim), dtype=np.float32), rewards, returns)
        ]
        fit_critic(m, episodes, epochs=15, lr=5e-3, device=device, eps_per_batch=1)

        mse_after = _critic_mse()
        self.assertLess(mse_after, mse_before, "critic MSE must decrease after fitting")
        env.close()

    def test_actor_params_unchanged_after_critic_fit(self):
        m, env = self._make_model()
        policy = m.policy

        T = 10
        rng = np.random.default_rng(42)
        obs_np = rng.uniform(-1, 1, (T, self.obs_dim)).astype(np.float32)
        act_np = rng.uniform(-1, 1, (T, self.act_dim)).astype(np.float32)
        rewards = np.ones(T, dtype=np.float32)
        returns = discounted_returns(rewards, GAMMA)

        actor_param_ids = set()
        for name, param in policy.named_parameters():
            if any(
                name.startswith(p)
                for p in (
                    "lstm_actor.",
                    "action_net.",
                    "mlp_extractor.policy_net.",
                    "log_std",
                )
            ):
                actor_param_ids.add(id(param))
        actor_before = {
            id(p): p.data.clone()
            for p in policy.parameters()
            if id(p) in actor_param_ids
        }

        episodes = [(obs_np, act_np, rewards, returns)]
        fit_critic(m, episodes, epochs=10, lr=5e-3, device="cpu", eps_per_batch=1)

        for pid, saved in actor_before.items():
            for p in policy.parameters():
                if id(p) == pid:
                    torch.testing.assert_close(
                        p.data, saved, msg="actor param changed during critic fit"
                    )
                    break

        env.close()

    def test_actor_prediction_retained_after_critic_fit(self):
        m, env = self._make_model()
        policy = m.policy

        T = 15
        rng = np.random.default_rng(42)
        obs_np = rng.uniform(-1, 1, (T, self.obs_dim)).astype(np.float32)

        obs_t = torch.as_tensor(obs_np)
        with torch.no_grad():
            means_before = _actor_mean_sequence(policy, obs_t).clone()

        act_np = np.zeros((T, self.act_dim), dtype=np.float32)
        rewards = np.ones(T, dtype=np.float32)
        returns = discounted_returns(rewards, GAMMA)
        episodes = [(obs_np, act_np, rewards, returns)]
        fit_critic(m, episodes, epochs=10, lr=5e-3, device="cpu", eps_per_batch=1)

        with torch.no_grad():
            means_after = _actor_mean_sequence(policy, obs_t)

        torch.testing.assert_close(
            means_before,
            means_after,
            atol=1e-6,
            rtol=1e-5,
            msg="actor predictions changed during critic fit",
        )
        env.close()

    def test_critic_params_are_exactly_value_path(self):
        m, env = self._make_model()
        cp = _critic_params(m.policy)
        critic_ids = set(id(p) for p in cp)
        for name, param in m.policy.named_parameters():
            if id(param) in critic_ids:
                is_critic = any(
                    name.startswith(p)
                    for p in ("lstm_critic.", "value_net.", "mlp_extractor.value_net.")
                )
                self.assertTrue(
                    is_critic, f"{name} is in critic set but not a critic param"
                )
            else:
                is_actor = any(
                    name.startswith(p)
                    for p in (
                        "lstm_actor.",
                        "action_net.",
                        "mlp_extractor.policy_net.",
                        "log_std",
                        "pi_features_extractor.",
                        "vf_features_extractor.",
                    )
                )
                self.assertTrue(
                    is_actor, f"{name} is not in critic set but not an actor param"
                )
        env.close()

    def test_critic_params_no_duplicates(self):
        m, env = self._make_model()
        cp = _critic_params(m.policy)
        ids = [id(p) for p in cp]
        self.assertEqual(
            len(ids), len(set(ids)), "critic params must not contain duplicates"
        )
        self.assertGreater(len(cp), 0, "critic params must be nonempty")
        env.close()

    def test_fit_critic_with_runnerlog(self):
        m, env = self._make_model()
        buf = io.StringIO()
        import tempfile
        from rl.core.diagnostics import RunnerLog

        T = 10
        rng = np.random.default_rng(42)
        obs_np = rng.uniform(-1, 1, (T, self.obs_dim)).astype(np.float32)
        rewards = np.ones(T, dtype=np.float32)
        returns = discounted_returns(rewards, GAMMA)
        episodes = [
            (obs_np, np.zeros((T, self.act_dim), dtype=np.float32), rewards, returns)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = RunnerLog(tag="test-critic", log_dir=tmp, stream=buf)
            fit_critic(
                m, episodes, epochs=5, lr=5e-3, device="cpu", eps_per_batch=1, log=log
            )
            log.close()
        text = buf.getvalue()
        self.assertIn("[critic]", text)
        self.assertIn("value_loss", text)
        env.close()


class MakeVecTests(unittest.TestCase):
    def test_stage0_nominal_domain_rand_false(self):
        envs = _make_vec(0, seed=0, shaping=1.0, n_envs=2, domain_rand=False)
        envs.reset()
        envs.close()

    def test_stage0_default_is_nominal(self):
        envs = _make_vec(0, seed=0, shaping=1.0, n_envs=2)
        envs.reset()
        envs.close()

    def test_later_stage_with_domain_rand_true(self):
        envs = _make_vec(2, seed=0, shaping=0.5, n_envs=2, domain_rand=True)
        envs.reset()
        envs.close()


class TrainStageTests(unittest.TestCase):
    """Tests for _train_stage using mocked evaluate/learn/save."""

    def _make_model_and_env(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)])
        m = _make_ppo(env, seed=0, device="cpu")
        m.num_timesteps = 0
        return m, env

    def test_final_stage_progress_without_completion_is_not_solved(self):
        metrics = {
            "gates": 14.0,
            "gate_frac": 14.0 / 17.0,
            "success": 0.0,
            "time": float("nan"),
        }
        self.assertFalse(_stage_solved(3, metrics))

    def test_final_stage_requires_repeatable_course_completion(self):
        metrics = {
            "gates": 17.0,
            "gate_frac": 1.0,
            "success": 0.6,
            "time": 48.0,
        }
        self.assertTrue(_stage_solved(3, metrics))

    def test_presolved_stage_zero_learn_calls(self):
        m, env = self._make_model_and_env()

        solved_eval = {"gates": 1.0, "gate_frac": 0.8, "success": 1.0, "time": 5.0}
        mock_evaluate = MagicMock(return_value=solved_eval)
        mock_save = MagicMock()
        learn_calls = [0]

        def fake_learn(*a, **kw):
            learn_calls[0] += 1
            m.num_timesteps += 256

        m.learn = fake_learn
        m.save = mock_save

        import rl.training.train_vq2 as tvq2

        orig_eval = tvq2.evaluate
        tvq2.evaluate = mock_evaluate
        try:
            result = _train_stage(
                m,
                stage=0,
                budget=1000,
                eval_episodes=5,
                checkpoint_path="/tmp/test.zip",
                domain_rand=False,
            )
        finally:
            tvq2.evaluate = orig_eval

        mock_evaluate.assert_called_once()
        self.assertEqual(
            learn_calls[0], 0, "model.learn must not be called for pre-solved stage"
        )
        mock_save.assert_called_once_with("/tmp/test.zip")
        self.assertFalse(result["learned"])
        self.assertEqual(result["learn_steps"], 0)
        env.close()

    def test_unsolved_stage_calls_learn_and_saves(self):
        m, env = self._make_model_and_env()

        unsolved_eval = {
            "gates": 0.0,
            "gate_frac": 0.0,
            "success": 0.0,
            "time": float("nan"),
        }
        solved_eval = {"gates": 1.0, "gate_frac": 0.8, "success": 1.0, "time": 5.0}
        eval_side_effects = [unsolved_eval, solved_eval]
        eval_idx = [0]

        def next_eval(*a, **kw):
            result = eval_side_effects[min(eval_idx[0], len(eval_side_effects) - 1)]
            eval_idx[0] += 1
            return result

        mock_evaluate = MagicMock(side_effect=next_eval)
        mock_save = MagicMock()

        def fake_learn(*a, **kw):
            m.num_timesteps += 256

        m.learn = fake_learn
        m.save = mock_save

        import rl.training.train_vq2 as tvq2

        orig_eval = tvq2.evaluate
        tvq2.evaluate = mock_evaluate
        try:
            result = _train_stage(
                m,
                stage=0,
                budget=1000,
                eval_episodes=5,
                checkpoint_path="/tmp/test2.zip",
                domain_rand=False,
            )
        finally:
            tvq2.evaluate = orig_eval

        self.assertTrue(result["learned"])
        self.assertGreater(result["learn_steps"], 0)
        mock_save.assert_called_once_with("/tmp/test2.zip")
        self.assertEqual(len(result["post_evals"]), 1)
        env.close()

    def test_learn_steps_matches_num_timesteps_delta(self):
        m, env = self._make_model_and_env()

        unsolved_eval = {
            "gates": 0.0,
            "gate_frac": 0.0,
            "success": 0.0,
            "time": float("nan"),
        }
        solved_eval = {"gates": 1.0, "gate_frac": 0.8, "success": 1.0, "time": 5.0}
        eval_side_effects = [unsolved_eval, solved_eval]
        eval_idx = [0]

        def next_eval(*a, **kw):
            result = eval_side_effects[min(eval_idx[0], len(eval_side_effects) - 1)]
            eval_idx[0] += 1
            return result

        mock_evaluate = MagicMock(side_effect=next_eval)
        m.save = MagicMock()

        def fake_learn(*a, **kw):
            m.num_timesteps += 512

        m.learn = fake_learn

        import rl.training.train_vq2 as tvq2

        orig_eval = tvq2.evaluate
        tvq2.evaluate = mock_evaluate
        try:
            result = _train_stage(
                m,
                stage=0,
                budget=2000,
                eval_episodes=5,
                checkpoint_path="/tmp/test3.zip",
                domain_rand=False,
            )
        finally:
            tvq2.evaluate = orig_eval

        self.assertEqual(result["learn_steps"], 512)
        env.close()

    def test_stage_result_contains_pre_eval(self):
        m, env = self._make_model_and_env()

        solved_eval = {"gates": 1.0, "gate_frac": 0.8, "success": 1.0, "time": 5.0}
        mock_evaluate = MagicMock(return_value=solved_eval)
        m.save = MagicMock()

        import rl.training.train_vq2 as tvq2

        orig_eval = tvq2.evaluate
        tvq2.evaluate = mock_evaluate
        try:
            result = _train_stage(
                m,
                stage=2,
                budget=1000,
                eval_episodes=5,
                checkpoint_path="/tmp/test4.zip",
                domain_rand=True,
            )
        finally:
            tvq2.evaluate = orig_eval

        self.assertEqual(result["stage"], 2)
        self.assertEqual(result["pre_eval"]["gate_frac"], 0.8)
        mock_evaluate.assert_called_once_with(m, 2, episodes=5, domain_rand=True)
        env.close()


class PPOConstructorTests(unittest.TestCase):
    def test_lr_and_target_kl(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)])
        m = _make_ppo(env, seed=0)
        # Lowered 1e-4 -> 3e-5 on measurement, not taste: at 1e-4 a stage-1
        # update logged approx_kl=0.26273 against this same target_kl of 0.03
        # (8.7x over), and the policy went from 0.55 gates to 0.00 in one
        # 75k-step chunk. The target only gets checked between epochs, so a
        # single epoch can already blow through it.
        self.assertEqual(m.learning_rate, 3e-5)
        self.assertEqual(m.target_kl, 0.03)
        self.assertAlmostEqual(m.gamma, 0.995, places=4)
        env.close()

    def test_policy_kwargs_include_lstm(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)])
        m = _make_ppo(env, seed=0)
        self.assertEqual(m.policy.lstm_actor.hidden_size, 128)
        self.assertIsNotNone(m.policy.lstm_critic)
        env.close()

    def test_share_features_extractor_false(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)])
        m = _make_ppo(env, seed=0)
        self.assertFalse(m.policy.share_features_extractor)
        self.assertIsNot(m.policy.pi_features_extractor, None)
        self.assertIsNot(m.policy.vf_features_extractor, None)
        self.assertIsNot(
            m.policy.pi_features_extractor,
            m.policy.vf_features_extractor,
            "pi and vf features extractors must be distinct objects",
        )
        env.close()


class EvaluateClosesEnvsTests(unittest.TestCase):
    def test_evaluate_closes_envs(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: VQ2RaceEnv(n_gates=1, seed=0, detector_dropout=0.0)])
        m = _make_ppo(env, seed=0, device="cpu")
        result = evaluate(m, 0, episodes=3, seed=0)
        self.assertIn("gates", result)
        self.assertIn("gate_frac", result)
        env.close()


if __name__ == "__main__":
    unittest.main()
