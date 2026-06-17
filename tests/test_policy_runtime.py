"""Deploy-path tests (torch required). Validates export->script->load->infer and the
pilot's deploy loop WITHOUT the simulator. Skipped if the RL deps aren't installed."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator import pilot as pilot_mod  # noqa: E402
from simulator.config import ROUND1_GATES  # noqa: E402
from simulator.policy_runtime import DeployPolicy, PolicyRuntime  # noqa: E402
from simulator.rl_core import ACTION_DIM, OBS_DIM  # noqa: E402


def _script_a_policy(path):
    deploy = DeployPolicy()
    deploy.eval()
    deploy.save(str(path))


def test_deploy_policy_script_load_infer(tmp_path):
    p = tmp_path / "policy.pt"
    _script_a_policy(p)
    rt = PolicyRuntime.load(str(p))
    action = rt.act(np.zeros(OBS_DIM, dtype=np.float32))
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(action))


class _FakeController:
    def set_control_mode(self, mode):
        pass

    def set_attitude_rates(self, *a):
        self.last = a


def _race_data(yaw=math.pi):
    return {
        "armed": True,
        "odometry": {"x": -5.0, "y": 0.0, "z": 0.0},
        "track_gates": ROUND1_GATES,
        "active_gate_index": 0,
        "yaw_rad": yaw,
        "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
    }


def test_pilot_deploy_loop_applies_policy(tmp_path, monkeypatch):
    p = tmp_path / "policy.pt"
    _script_a_policy(p)
    monkeypatch.setattr(pilot_mod, "POLICY_PATH", str(p))

    pilot = pilot_mod.Pilot(_FakeController(), {})
    assert pilot._policy is not None  # loaded the scripted policy

    pilot.data = _race_data()
    pilot._policy_last_t = -1e9  # force the throttled inference to run this tick
    pilot.tick()
    # deploy loop produced an action and a valid residual; bare-pilot fallback not triggered
    assert pilot._policy is not None
    assert pilot.last_action.shape == (ACTION_DIM,)
    assert 0.4 <= pilot.residual.speed_mult <= 2.5


def test_export_from_sb3_matches_installed_version(tmp_path):
    """Build a real PPO model (no training) and export it — validates that
    export_from_sb3's layer names match the installed stable-baselines3 version."""
    sb3 = pytest.importorskip("stable_baselines3")
    gym = pytest.importorskip("gymnasium")
    import torch.nn as nn

    from simulator.policy_runtime import export_from_sb3

    class _Stub(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1, 1, (OBS_DIM,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (ACTION_DIM,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, True, {}

    model = sb3.PPO(
        "MlpPolicy",
        _Stub(),
        n_steps=8,
        batch_size=8,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh
        ),
    )
    path = tmp_path / "policy.pt"
    export_from_sb3(model, None, str(path))  # no VecNormalize
    rt = PolicyRuntime.load(str(path))
    out = rt.act(np.zeros(OBS_DIM, dtype=np.float32))
    assert out.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(out))


def test_pilot_watchdog_falls_back_on_bad_policy(monkeypatch):
    class BadPolicy:
        def act(self, obs):
            return np.array([float("nan"), 0.0, 0.0], dtype=np.float32)

    pilot = pilot_mod.Pilot(_FakeController(), {})
    pilot._policy = BadPolicy()
    pilot.data = _race_data()
    pilot._policy_last_t = -1e9
    pilot.tick()
    # NaN action -> watchdog disables policy and reverts to identity residual
    assert pilot._policy is None
    assert pilot.residual.speed_mult == 1.0
