"""Deploy-time runtime for the trained residual policy.

Kept separate from training so the DEPLOY path depends only on `torch` (no
stable-baselines3 / gymnasium). The trainer exports a self-contained TorchScript
module via `export_from_sb3`; at race time the pilot loads it with `PolicyRuntime.load`
and calls `.act(obs)` for the deterministic (mean) action. Observation normalization
(from VecNormalize) is baked into the module so no extra state is needed.

torch is imported at module top — but this module is only imported when a policy file
actually exists (see pilot._maybe_load_policy), so `make sim` needs no RL deps by default.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from simulator.rl_core import ACTION_DIM, OBS_DIM


class DeployPolicy(nn.Module):
    """Standalone MLP that maps a raw observation to the deterministic mean action.

    Mirrors an SB3 MlpPolicy policy branch (net_arch=[64,64], Tanh) + action head, with
    VecNormalize obs stats folded into a normalize step. Scriptable for version-robust
    deployment.
    """

    def __init__(
        self, hidden=(64, 64), obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM
    ):
        super().__init__()
        self._hidden = tuple(hidden)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("clip_obs", torch.tensor(10.0))
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(last, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_n = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        obs_n = torch.clamp(obs_n, -self.clip_obs, self.clip_obs)
        return self.action_net(self.policy_net(obs_n))

    def save(self, path: str) -> None:
        """Save a plain checkpoint (state_dict + arch). Avoids TorchScript, which is
        deprecated on Python 3.14+; this format survives torch upgrades."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden": list(self._hidden),
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
            },
            path,
        )


class PolicyRuntime:
    """Thin wrapper around a scripted DeployPolicy for inference on the control thread."""

    def __init__(self, module):
        self._module = module
        self._module.eval()

    @classmethod
    def load(cls, path: str) -> "PolicyRuntime":
        # Trusted local checkpoint produced by our own trainer -> weights_only=False.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        module = DeployPolicy(
            hidden=tuple(ckpt["hidden"]),
            obs_dim=ckpt["obs_dim"],
            act_dim=ckpt["act_dim"],
        )
        module.load_state_dict(ckpt["state_dict"])
        return cls(module)

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        a = self._module(x).reshape(-1)
        return a.cpu().numpy().astype(np.float32)


def export_from_sb3(model, vec_normalize, path: str, hidden=(64, 64)) -> None:
    """Build a DeployPolicy from a trained SB3 PPO model (+ optional VecNormalize),
    script it, and save to `path`. Called at the end of training (P3)."""
    deploy = DeployPolicy(hidden=hidden)

    # Copy obs normalization stats.
    if (
        vec_normalize is not None
        and getattr(vec_normalize, "obs_rms", None) is not None
    ):
        deploy.obs_mean.copy_(
            torch.as_tensor(vec_normalize.obs_rms.mean, dtype=torch.float32)
        )
        deploy.obs_var.copy_(
            torch.as_tensor(vec_normalize.obs_rms.var, dtype=torch.float32)
        )
        deploy.clip_obs.copy_(torch.tensor(float(vec_normalize.clip_obs)))

    # Copy the policy branch + action head weights from the SB3 policy.
    sb3 = model.policy
    sb3_policy_net = sb3.mlp_extractor.policy_net
    src_linears = [m for m in sb3_policy_net if isinstance(m, nn.Linear)]
    dst_linears = [m for m in deploy.policy_net if isinstance(m, nn.Linear)]
    if len(src_linears) != len(dst_linears):
        raise ValueError(
            f"net_arch mismatch: SB3 has {len(src_linears)} hidden linears, "
            f"DeployPolicy has {len(dst_linears)}"
        )
    with torch.no_grad():
        for d, s in zip(dst_linears, src_linears):
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
        deploy.action_net.weight.copy_(sb3.action_net.weight)
        deploy.action_net.bias.copy_(sb3.action_net.bias)

    deploy.eval()
    deploy.save(path)
