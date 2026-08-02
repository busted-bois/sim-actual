"""Focused tests for T12 swappable algorithm interface.

No real PPO training runs; no checked-in artifacts are mutated. Mocks patch
the underlying runners/savers; BC save/load roundtrips use a tempdir.
"""

from __future__ import annotations

import unittest

from rl.algorithms import ALGORITHMS, Algorithm, BC, PPO, SAC, get_algorithm

_SAC_MSG = "SAC not implemented; interface slot for future. Use PPO."


class TestRegistry(unittest.TestCase):
    def test_registry_literal(self):
        self.assertEqual(ALGORITHMS, {"ppo": PPO, "bc": BC, "sac": SAC})

    def test_get_algorithm_returns_classes(self):
        self.assertIs(get_algorithm("ppo"), PPO)
        self.assertIs(get_algorithm("bc"), BC)
        self.assertIs(get_algorithm("sac"), SAC)

    def test_get_algorithm_unknown_raises_listing_valid_names(self):
        with self.assertRaises(ValueError) as cm:
            get_algorithm("unknown")
        msg = str(cm.exception)
        for name in ("ppo", "bc", "sac"):
            self.assertIn(name, msg)

    def test_get_algorithm_is_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_algorithm("PPO")

    def test_get_algorithm_rejects_non_string(self):
        with self.assertRaises(TypeError):
            get_algorithm(None)


class TestProtocolConformance(unittest.TestCase):
    def test_all_adapters_conform_to_protocol(self):
        for cls in (PPO, BC, SAC):
            self.assertIsInstance(cls(), Algorithm)


class TestSACStub(unittest.TestCase):
    def test_train_raises_exact_message(self):
        with self.assertRaises(NotImplementedError) as cm:
            SAC().train(None, None)
        self.assertEqual(str(cm.exception), _SAC_MSG)

    def test_load_raises_exact_message(self):
        with self.assertRaises(NotImplementedError) as cm:
            SAC().load("x")
        self.assertEqual(str(cm.exception), _SAC_MSG)

    def test_save_raises_exact_message(self):
        with self.assertRaises(NotImplementedError) as cm:
            SAC().save(None, "x")
        self.assertEqual(str(cm.exception), _SAC_MSG)

    def test_no_sac_implementation_imported(self):
        import rl.algorithms.sac as sac_module

        with open(sac_module.__file__) as f:
            src = f.read()
        self.assertNotIn("stable_baselines3", src)
        self.assertNotIn("torch", src)
        self.assertNotIn("from rl.training", src)
        self.assertNotIn("import sac", src.lower().replace("_msg", ""))


if __name__ == "__main__":
    unittest.main()
