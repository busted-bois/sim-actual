"""Versioned YAML configuration for the RL pipeline."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


class ConfigError(ValueError):
    """Raised when an RL configuration is malformed or unsupported."""


@dataclass
class EnvConfig:
    curriculum_stage: int = 2
    max_steps: int = 1000
    gate_size: float = 2.72


@dataclass
class PPOConfig:
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    n_epochs: int = 10
    net_arch: list[int] = field(default_factory=lambda: [64, 64, 64])


@dataclass
class BCConfig:
    epochs: int = 40
    lr: float = 1e-3


@dataclass
class DomainRandConfig:
    mass_pct: float = 0.10
    inertia_pct: float = 0.10
    hover_offset: float = 0.05
    wind: float = 0.0


@dataclass
class CheckpointConfig:
    dir: str = "rl/data/best"
    atomic: bool = True
    resume: bool = True
    schema_min_version: int = 1


@dataclass
class RLConfig:
    config_version: int = 1
    seed: int = 0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    eval_episodes: int = 50
    policy_path: str = "rl/data/policy.pt"
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    device: str = "auto"


def default_config() -> RLConfig:
    return RLConfig()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise ConfigError(f"invalid device {device!r}") from exc


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{location} keys must be strings")
    return value


def _construct(cls, value: object, location: str):
    values = _mapping(value, location)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {location}: {exc}") from exc


def _validate(cfg: RLConfig) -> None:
    minimum = CheckpointConfig().schema_min_version
    if cfg.config_version < minimum:
        raise ConfigError(
            f"config_version {cfg.config_version} is older than supported version {minimum}"
        )
    if cfg.checkpoint.schema_min_version != minimum:
        raise ConfigError(f"checkpoint.schema_min_version must be {minimum}")
    if not cfg.seeds:
        raise ConfigError("seeds must not be empty")
    if cfg.eval_episodes < 1 or cfg.env.max_steps < 1:
        raise ConfigError("eval_episodes and env.max_steps must be positive")
    if cfg.ppo.n_steps < 1 or cfg.ppo.batch_size < 1 or cfg.ppo.n_epochs < 1:
        raise ConfigError("PPO step, batch, and epoch counts must be positive")
    if not cfg.ppo.net_arch or any(width < 1 for width in cfg.ppo.net_arch):
        raise ConfigError("ppo.net_arch widths must be positive")
    if cfg.bc.epochs < 1:
        raise ConfigError("bc.epochs must be positive")
    resolve_device(cfg.device)


def load_config(path: str | Path) -> RLConfig:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not load {source}: {exc}") from exc
    data = _mapping(raw, "config")
    sections = {
        "env": EnvConfig,
        "ppo": PPOConfig,
        "bc": BCConfig,
        "domain_rand": DomainRandConfig,
        "checkpoint": CheckpointConfig,
    }
    values = dict(data)
    for name, cls in sections.items():
        if name in values:
            values[name] = _construct(cls, values[name], name)
    try:
        cfg = RLConfig(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid config: {exc}") from exc
    _validate(cfg)
    return cfg


def save_config(cfg: RLConfig, path: str | Path) -> None:
    _validate(cfg)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True)
    )


def _selftest() -> None:
    default_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    cfg = load_config(default_path)
    assert cfg == default_config()
    assert resolve_device("auto").type in {"cpu", "cuda"}
    with tempfile.TemporaryDirectory() as directory:
        roundtrip_path = Path(directory) / "roundtrip.yaml"
        save_config(cfg, roundtrip_path)
        assert load_config(roundtrip_path) == cfg
        invalid_path = Path(directory) / "invalid.yaml"
        invalid_path.write_text("config_version: 0\n")
        try:
            load_config(invalid_path)
        except ConfigError:
            pass
        else:
            raise AssertionError("config_version=0 was accepted")
    print("[selftest] OK — config load/save round-trip and schema enforcement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
