"""Atomic, versioned PPO checkpoint save / load / resume.

Each checkpoint is a single zip bundle (``*.ckpt``) holding the complete
SB3 PPO archive plus a ``metadata.json`` member.  Writes go to a temp file
in the destination directory and are installed with :func:`os.replace` so a
crash never leaves a partially-written checkpoint.

Bundle layout (zip)::

    metadata.json   — JSON metadata (see ``_REQUIRED_KEYS``)
    model.zip       — raw SB3 ``PPO.save`` archive bytes

    uv run -m rl.training.checkpoint --selftest
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from rl.core.config import RLConfig

BUNDLE_SUFFIX = ".ckpt"
_META_MEMBER = "metadata.json"
_MODEL_MEMBER = "model.zip"

_REQUIRED_KEYS: tuple[str, ...] = (
    "config_version",
    "training_step",
    "seed",
    "stage",
    "stage_step",
)
_INT_KEYS: tuple[str, ...] = (
    "config_version",
    "training_step",
    "seed",
    "stage",
    "stage_step",
)
_RESERVED_KEYS: tuple[str, ...] = (*_REQUIRED_KEYS, "schema_min_version")


class CheckpointError(Exception):
    """Raised for corrupt, unreadable, or otherwise invalid checkpoints."""


class CheckpointNotFoundError(CheckpointError):
    """Raised by ``resume_latest`` when the directory has zero candidates."""


class CheckpointVersionError(CheckpointError):
    """Raised when a checkpoint's config/schema version is too old."""


def _validate_metadata(raw: dict[str, Any], cfg: RLConfig) -> dict[str, Any]:
    """Validate metadata structure / types / ranges.

    Returns the checked metadata dict.  Raises :class:`CheckpointVersionError`
    for version mismatches, :class:`CheckpointError` for other problems.
    """
    if not isinstance(raw, dict):
        raise CheckpointError("metadata.json is not a JSON object")

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise CheckpointError(f"metadata missing required keys: {missing}")

    for key in _INT_KEYS:
        val = raw[key]
        # JSON bools are ints in Python — reject explicitly.
        if isinstance(val, bool) or not isinstance(val, int):
            raise CheckpointError(
                f"metadata.{key} must be an integer, got {type(val).__name__}"
            )
        if val < 0:
            raise CheckpointError(f"metadata.{key} must be non-negative, got {val}")

    if raw["config_version"] == 0:
        raise CheckpointVersionError(
            "metadata.config_version is 0 (unsupported); supported >= "
            f"{cfg.checkpoint.schema_min_version}"
        )

    if raw["config_version"] < cfg.checkpoint.schema_min_version:
        raise CheckpointVersionError(
            f"checkpoint config_version {raw['config_version']} is older than "
            f"supported schema_min_version {cfg.checkpoint.schema_min_version}"
        )

    return raw


def _read_bundle(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """Open ``path`` as a checkpoint bundle and return ``(model_bytes, metadata)``.

    Validates zip integrity.  Raises :class:`CheckpointError` on any problem.
    """
    path = str(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            bad = archive.testzip()
            if bad is not None:
                raise CheckpointError(f"corrupt zip member '{bad}' in {path}")
            names = archive.namelist()
            if _META_MEMBER not in names:
                raise CheckpointError(f"missing {_META_MEMBER} in checkpoint {path}")
            if _MODEL_MEMBER not in names:
                raise CheckpointError(f"missing {_MODEL_MEMBER} in checkpoint {path}")
            meta_bytes = archive.read(_META_MEMBER)
            model_bytes = archive.read(_MODEL_MEMBER)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"not a valid zip / checkpoint: {path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise CheckpointError(f"checkpoint not found: {path}") from exc
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc

    try:
        raw_meta = json.loads(meta_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointError(f"corrupt metadata.json in {path}: {exc}") from exc

    return model_bytes, raw_meta


def _read_metadata_strict(path: str | Path) -> dict[str, Any]:
    """Read metadata.json from a checkpoint, raising on any corruption.

    Unlike the old ``_read_metadata``, this never returns ``None`` — it
    raises :class:`CheckpointError` so the caller can identify the bad file.
    Model bytes are intentionally not read.
    """
    path = str(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            bad = archive.testzip()
            if bad is not None:
                raise CheckpointError(f"corrupt zip member '{bad}' in {path}")
            if _META_MEMBER not in archive.namelist():
                raise CheckpointError(f"missing {_META_MEMBER} in checkpoint {path}")
            meta_bytes = archive.read(_META_MEMBER)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"not a valid zip / checkpoint: {path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise CheckpointError(f"checkpoint not found: {path}") from exc
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc

    try:
        return json.loads(meta_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointError(f"corrupt metadata.json in {path}: {exc}") from exc


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically via same-dir temp + os.replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_atomic(
    model: PPO,
    path: str | Path,
    cfg: RLConfig,
    *,
    training_step: int | None = None,
    seed: int | None = None,
    stage: int = 0,
    stage_step: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``model`` + metadata as an atomic ``.ckpt`` bundle at ``path``.

    SB3 serialises into a BytesIO so its open_path suffix logic (which
    appends ``.zip`` when the path has no suffix) never runs.  The bundle is
    written to a same-directory temp file and installed via
    :func:`os.replace`, guaranteeing atomicity on POSIX.

    Raises :class:`CheckpointError` if an explicit *training_step* differs
    from ``model.num_timesteps``, or if *extra* collides with reserved keys.
    """
    if extra:
        collisions = set(extra) & set(_RESERVED_KEYS)
        if collisions:
            raise CheckpointError(
                f"extra keys collide with reserved metadata: {sorted(collisions)}"
            )

    actual_ts = int(model.num_timesteps)
    if training_step is None:
        training_step = actual_ts
    else:
        if isinstance(training_step, bool) or not isinstance(training_step, int):
            raise CheckpointError(
                f"training_step must be an int, got {type(training_step).__name__}"
            )
        if training_step != actual_ts:
            raise CheckpointError(
                f"training_step ({training_step}) does not match "
                f"model.num_timesteps ({actual_ts})"
            )
    if seed is None:
        seed = cfg.seed

    metadata: dict[str, Any] = {
        "config_version": cfg.config_version,
        "schema_min_version": cfg.checkpoint.schema_min_version,
        "training_step": int(training_step),
        "seed": int(seed),
        "stage": int(stage),
        "stage_step": int(stage_step),
    }
    if extra:
        metadata.update(extra)

    _validate_metadata(metadata, cfg)

    sb3_buf = io.BytesIO()
    model.save(sb3_buf)
    sb3_bytes = sb3_buf.getvalue()

    bundle_buf = io.BytesIO()
    with zipfile.ZipFile(bundle_buf, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(_META_MEMBER, json.dumps(metadata, sort_keys=True).encode())
        bundle.writestr(_MODEL_MEMBER, sb3_bytes)

    _atomic_write_bytes(Path(path), bundle_buf.getvalue())
    return metadata


def save_sb3_atomic(model: PPO, path: str | Path) -> None:
    """Write a raw SB3 ``.zip`` archive atomically (no metadata bundle).

    Used for the convenience ``policy_ppo.zip`` output so it is never
    half-written.  The canonical resume source remains the ``.ckpt`` bundle.
    """
    buf = io.BytesIO()
    model.save(buf)
    _atomic_write_bytes(Path(path), buf.getvalue())


def load(
    path: str | Path,
    cfg: RLConfig,
    *,
    env=None,
    device: str = "auto",
) -> tuple[PPO, dict[str, Any]]:
    """Validate and load a checkpoint bundle.

    Returns ``(model, metadata)``.  Raises :class:`CheckpointVersionError`
    for version mismatches and :class:`CheckpointError` for corruption,
    missing members, metadata/model inconsistency, or any other load failure.
    """
    model_bytes, raw_meta = _read_bundle(path)
    metadata = _validate_metadata(raw_meta, cfg)

    try:
        buf = io.BytesIO(model_bytes)
        model = PPO.load(buf, env=env, force_reset=True, device=device)
    except Exception as exc:
        raise CheckpointError(f"failed to deserialise PPO from {path}: {exc}") from exc

    if int(model.num_timesteps) != int(metadata["training_step"]):
        raise CheckpointError(
            f"metadata.training_step ({metadata['training_step']}) does not match "
            f"model.num_timesteps ({model.num_timesteps}) in {path}"
        )

    return model, metadata


def _checkpoint_candidates(directory: str | Path) -> list[Path]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    return sorted(dir_path.glob(f"*{BUNDLE_SUFFIX}"))


def resume_latest(
    directory: str | Path,
    cfg: RLConfig,
    *,
    env=None,
    device: str = "auto",
) -> tuple[PPO, dict[str, Any]]:
    """Find, validate and load the newest checkpoint by ``training_step``.

    Selection is by validated ``metadata.training_step`` — **not** by file
    mtime or lexicographic name.  Every candidate is validated; if **any**
    file is unreadable, malformed, or schema-incompatible, a
    :class:`CheckpointError` (or :class:`CheckpointVersionError`) is raised
    identifying the offending file.  No silent fallback, no mtime ranking.
    No files are quarantined or deleted.
    """
    candidates = _checkpoint_candidates(directory)
    if not candidates:
        raise CheckpointNotFoundError(
            f"no checkpoint files ({BUNDLE_SUFFIX}) in {directory}"
        )

    validated: list[tuple[int, Path, dict[str, Any]]] = []
    for cand in candidates:
        raw_meta = _read_metadata_strict(cand)
        try:
            meta = _validate_metadata(raw_meta, cfg)
        except CheckpointError as exc:
            raise type(exc)(f"{cand}: {exc}") from exc
        validated.append((meta["training_step"], cand, meta))

    validated.sort(key=lambda item: item[0], reverse=True)
    best_step, best_path, _best_meta = validated[0]

    model, meta = load(best_path, cfg, env=env, device=device)
    return model, meta


def _selftest() -> None:
    """Exercise save/load parity, corruption rejection, and version rejection."""
    import shutil
    import tempfile as _tf
    import time

    import torch
    import torch.nn as nn

    from rl.core.config import default_config
    from rl.training.train_ppo import NET_ARCH, _vec_env

    cfg = default_config()
    cfg.checkpoint.dir = _tf.mkdtemp(prefix="ckpt-selftest-")
    try:
        env = _vec_env(0, n_envs=1, seed=0)
        policy_kwargs: dict[str, Any] = dict(
            net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            seed=0,
            device="cpu",
        )
        model.learn(total_timesteps=128)
        env.close()
        model.num_timesteps = 7777

        # 1. save → load parity
        ckpt_path = os.path.join(cfg.checkpoint.dir, "parity.ckpt")
        save_atomic(model, ckpt_path, cfg, stage=0, stage_step=7777)

        assert os.path.exists(ckpt_path), "checkpoint file not created"
        leftovers = [
            f for f in os.listdir(cfg.checkpoint.dir) if f.startswith(".parity.ckpt.")
        ]
        assert not leftovers, f"temp files left behind: {leftovers}"

        loaded, meta = load(ckpt_path, cfg, device="cpu")
        assert meta["training_step"] == 7777, f"meta training_step wrong: {meta}"
        assert loaded.num_timesteps == 7777, (
            f"num_timesteps not restored: {loaded.num_timesteps}"
        )

        obs = np.random.RandomState(42).uniform(-1, 1, (16, 24)).astype(np.float32)
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        assert np.array_equal(a1, a2), "prediction mismatch after save/load"

        opt_a = model.policy.optimizer.state_dict()["state"]
        opt_b = loaded.policy.optimizer.state_dict()["state"]
        for k in opt_a:
            for buf_name in opt_a[k]:
                va = opt_a[k][buf_name]
                vb = opt_b[k][buf_name]
                if isinstance(va, torch.Tensor):
                    assert torch.equal(va, vb), f"optim state[{k}][{buf_name}] mismatch"
        print("[selftest] save/load parity OK (predictions, num_timesteps, optim)")

        # 1b. training_step/model mismatch rejected at save time
        try:
            save_atomic(model, ckpt_path + ".bad", cfg, training_step=999)
        except CheckpointError:
            print("[selftest] training_step mismatch rejection at save OK")
        else:
            raise AssertionError("save_atomic accepted mismatched training_step")

        # 1c. extra keys collision rejected at save time
        try:
            save_atomic(model, ckpt_path + ".bad2", cfg, extra={"stage": 99})
        except CheckpointError:
            print("[selftest] extra-key collision rejection OK")
        else:
            raise AssertionError("save_atomic accepted colliding extra keys")

        # 2. Corruption rejection (truncated file)
        corrupt_path = os.path.join(cfg.checkpoint.dir, "corrupt.ckpt")
        shutil.copy(ckpt_path, corrupt_path)
        with open(corrupt_path, "r+b") as f:
            f.truncate(os.path.getsize(corrupt_path) // 2)
        try:
            load(corrupt_path, cfg, device="cpu")
        except CheckpointError:
            print("[selftest] corruption rejection OK")
        else:
            raise AssertionError("truncated checkpoint was accepted silently")

        # 3. Version rejection (config_version=0)
        ver0_path = os.path.join(cfg.checkpoint.dir, "ver0.ckpt")
        save_atomic(model, ver0_path, cfg, stage=0, stage_step=7777)
        with zipfile.ZipFile(ver0_path, "r") as zf:
            model_bytes = zf.read(_MODEL_MEMBER)
        meta0 = {
            "config_version": 0,
            "training_step": 7777,
            "seed": 0,
            "stage": 0,
            "stage_step": 7777,
        }
        fd, tmp = tempfile.mkstemp(dir=cfg.checkpoint.dir)
        try:
            with os.fdopen(fd, "wb") as tf_:
                with zipfile.ZipFile(tf_, "w") as zf:
                    zf.writestr(_META_MEMBER, json.dumps(meta0))
                    zf.writestr(_MODEL_MEMBER, model_bytes)
            os.replace(tmp, ver0_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        try:
            load(ver0_path, cfg, device="cpu")
        except CheckpointVersionError:
            print("[selftest] version-0 rejection OK")
        else:
            raise AssertionError("config_version=0 checkpoint was accepted")

        # Clean up prior test artifacts so they don't interfere with resume_latest.
        for stale in (corrupt_path, ver0_path, ckpt_path):
            Path(stale).unlink(missing_ok=True)

        # 4. resume_latest picks highest training_step, not mtime
        for step in (100, 300, 200):
            model.num_timesteps = step
            p = os.path.join(cfg.checkpoint.dir, f"resume_{step:08d}.ckpt")
            save_atomic(model, p, cfg, stage=0, stage_step=step)
            old_ts = time.time() - 9999 + step
            os.utime(p, (old_ts, old_ts))
        p100 = os.path.join(cfg.checkpoint.dir, "resume_00000100.ckpt")
        os.utime(p100, (time.time(), time.time()))
        _, rmeta = resume_latest(cfg.checkpoint.dir, cfg, device="cpu")
        assert rmeta["training_step"] == 300, (
            f"resume_latest picked wrong checkpoint: {rmeta['training_step']}"
        )
        print("[selftest] resume_latest selection by training_step OK")

        # 5. resume_latest fails loudly on ANY corrupt candidate (no fallback)
        bad = os.path.join(cfg.checkpoint.dir, "bad.ckpt")
        with open(bad, "wb") as f:
            f.write(os.urandom(512))
        try:
            resume_latest(cfg.checkpoint.dir, cfg, device="cpu")
        except CheckpointError:
            print("[selftest] resume_latest corrupt-candidate rejection OK")
        else:
            raise AssertionError("resume_latest ignored a corrupt candidate")

        print("[selftest] OK — all checkpoint assertions passed")
    finally:
        shutil.rmtree(cfg.checkpoint.dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atomic PPO checkpoint utilities.")
    parser.add_argument(
        "--selftest", action="store_true", help="Run built-in self-test."
    )
    args = parser.parse_args()
    if args.selftest:
        _selftest()
