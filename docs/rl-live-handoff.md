# RL live handoff runbook

This is the exact, repeatable path from a cold checkout to flying a trained
PPO policy on the live sim. Every command here is copy-pasteable. Run them in
order. If a step does not produce the artifact it promises, stop and read the
troubleshooting section before continuing.

The short version: collect frames, optionally train a GateNet, train PPO,
evaluate it deterministically, and only then hand it to the live deploy loop. The
checked-in `rl/data/policy.pt` is a regression anchor, not a flight policy.
Do not fly it expecting gates.

## What lives where

The repository layout matters because the deploy loop hard-codes some of
these paths.

- `rl/data/policy.pt` is the checked-in anchor. It is the zero-gate baseline
  artifact, not a competent policy.
- `rl/data/gatenet.pt` is the GateNet U-Net. The baseline records it as
  `absent`, so expect to train it.
- `rl/data/gate_map.json` is the race-start gate burst. The sim only emits it
  once, at race start.
- `rl/data/best/<run>/` holds per-run training output: checkpoints, the
  exported `policy.pt`, `progress.csv`, and the SB3 `policy_ppo.zip`.
- `rl/data/tb/<run>/` holds the TensorBoard event files.
- `rl/data/baseline.json` is the frozen T0 contract. Do not edit it by hand.
  Regenerate it with `make rl-baseline`.
- `flightlab/calibration.json` and `flightlab/signs.json` are the measured
  attitude plant. They are `.gitignore`d, so each machine captures its own.
- `configs/default.yaml` is the single source of truth for PPO hyperparameters
  and the env stage.

## Prerequisites

You need `uv` installed and the sim runnable. CPU is enough for smoke tests
and behavior cloning. Use a GPU for meaningful full PPO training; CPU-only
PPO is supported but is not a practical convergence workflow.

```bash
make            # uv sync, installs everything
make rl-test    # offline self-tests for every RL module, no live sim needed
```

`make rl-test` must pass before you do anything else. It exercises the state
estimator, dataset labeler, GateNet, PnP, EKF, observation builder, the env,
and the deploy self-test. If any module fails here it will fail worse in the
air.

Run the attitude harness before live flight to validate the current sim's
signs and response. It writes machine-local calibration artifacts, but deploy
does not read them to suppress checkpoint warnings. Training and deploy use
the current constants in `rl/core/spec.py`; checkpoint warnings compare saved
training metadata with those constants. The checked-in anchor predates the
current plant, which is why its metadata reads
`train_hover: 0.5` and `action_scale: [4.0, 4.0, 3.0]` against a real plant
that hovers near 0.27.

```bash
make attitude-harness   # writes flightlab/calibration.json + signs.json
```

Run that with the sim in a TRAINING session. The harness is open-loop and
characterizes the attitude and thrust response. Its results tell you whether
the current spec remains valid; they are not automatically embedded into a
checkpoint. Retrain after deliberately updating the spec from measured data.

## Gate-map capture

The sim transmits the track as a short MAVLink burst at race start, nothing
before, nothing after. You must be listening when the race starts or you miss
it entirely.

Open one terminal and start the listener first:

```bash
make capture-gates      # uv run scripts/capture_gates.py (150s window)
```

Then, in the sim, click Race (or restart the race). The listener prints
`CAPTURED N gates -> rl/data/gate_map.json` when it succeeds. If you see
`NO gate burst captured in window`, the race did not restart inside the
window. Run it again and restart the race while it listens.

The deploy loop will look for `rl/data/gate_map.json` and refuses to fly
without a gate map. If the file is missing you get an abort message pointing
you back at `make capture-gates`.

## GateNet data and training

GateNet is the gate-segmentation U-Net. It is optional in the live loop, the
deploy path prefers native PnP, but training it is part of the full pipeline
and the dataset module is how you get labeled frames.

### Collect frames

Module 2 flies the existing vision pilot and records frames plus auto-labeled
masks at camera rate. The labels are pixel-perfect because they come from
projecting the known gate geometry through the camera intrinsics. No manual
labeling.

The sim must be running and on the course before you start.

```bash
make dataset                              # 1200 frames default
uv run -m rl.perception.dataset --frames 1500   # override the count
```

Output lands in `rl/data/gatenet_ds/` with `images/`, `masks/`, and a
`meta.json`. Frame count is controlled with `--frames`; no separate collection
mode is required.

### Train GateNet

Module 3 trains the U-Net on those pairs and writes `rl/data/gatenet.pt`.

```bash
make train-gatenet        # uv run -m rl.perception.gatenet (defaults: 30 epochs, bs 8)
uv run -m rl.perception.gatenet --data rl/data/gatenet_ds --epochs 30 --bs 8
```

The data root defaults to `rl/data/gatenet_ds`, the epochs to 30, the batch
size to 8. Training is pure torch so it runs on CPU. The baseline records
GateNet weights as absent, so a freshly trained `gatenet.pt` is strictly an
addition.

A quick sanity check before committing to a long run:

```bash
uv run -m rl.perception.gatenet --selftest   # synthetic end-to-end smoke
```

## PPO training

Module 8 trains the PPO policy. The entry point is `rl.training.train_ppo`.
Note that `rl.algorithms.ppo` has no CLI of its own, it is a library module.
Always invoke the training runner.

```bash
make rl-train    # uses the variables below
```

That expands to:

```bash
uv run python -m rl.training.train_ppo --config configs/default.yaml --run-name ppo --steps 300000
```

The Makefile variables are overridable, so you can point at a candidate run
or bound the step count without editing anything:

```bash
make rl-train RUN=my-run STEPS=100000
```

- `CONFIG` defaults to `configs/default.yaml`.
- `RUN` defaults to `ppo` and names the checkpoint directory.
- `STEPS` overrides `ppo.total_timesteps_per_stage` from the config.
- `POLICY` has no default. Pass a candidate explicitly to `fly-policy` and
  `rl-eval`; pass `rl/data/policy.pt` only when intentionally evaluating the
  frozen anchor.

Always smoke-test the runner before a full run. It bounds to a few thousand
steps on CPU, stage 0 only, and proves the pipeline end to end:

```bash
uv run python -m rl.training.train_ppo --smoke 2000 --run-name smoke
```

The runner trains across the curriculum, three stages from one gate to the
full six-gate course. It exports two artifacts at the end.

- `rl/data/best/<run>/policy.pt` is the dependency-light standalone actor
  for deploy.
- `rl/data/best/<run>/policy_ppo.zip` is the full SB3 model for resuming.

### BC warm start

If `rl/data/policy_bc.pt` exists, the runner picks it up automatically as a
warm start. Generate it from the GP expert demos:

```bash
make log-demos    # rl/data/gp_demos.npz
make train-bc     # rl/data/policy_bc.pt
```

Both are `.gitignore`d, so they are per-machine. Skip them for a from-scratch
PPO run. The `--bc-init` flag lets you point at a specific checkpoint if you
want to override the default discovery.

### Watching progress

Two streams come out of training.

- `rl/data/best/<run>/progress.csv` gets one row per finished episode with
  `ep_rew`, `gates_cleared`, and `n_steps`.
- `rl/data/tb/<run>/` holds TensorBoard event files.

```bash
uv run tensorboard --logdir rl/data/tb
```

Training also writes periodic atomic checkpoints named
`checkpoint_XXXXXXXX.ckpt` into the run directory. With
`checkpoint.resume: true` in the config, re-running the same `--run-name`
resumes from the latest checkpoint instead of restarting.

## Deterministic evaluation

Before any live flight, evaluate the candidate against the frozen T0
protocol. This reproduces the baseline exactly: raw `GateRacingEnv`, stage 2,
seeds `[0, 1, 2]`, 50 episodes each, deterministic actions, episode seed
`base_seed * 10000 + episode_index`.

```bash
make rl-eval POLICY=rl/data/best/my-run/policy.pt
```

Point it at your candidate instead of the anchor:

```bash
make rl-eval POLICY=rl/data/best/my-run/policy.pt
# or directly:
uv run python -m rl.training.evaluate --policy rl/data/best/my-run/policy.pt --episodes 50 --seeds 0 1 2
```

The harness prints a one-line summary and writes one row per episode to a
CSV, defaulting to `rl/data/best/<stem>/eval.csv`.

The expert reference is also runnable through the same harness, which is how
the baseline numbers were produced:

```bash
uv run python -m rl.training.evaluate --expert gp_expert
```

### Reading the numbers

The frozen baseline in `rl/data/baseline.json` is the reference. Two numbers
matter most.

- The expert clears `2.8066666666666666` gates on average with population
  std `1.1177159249509192`, across 150 episodes, with a near-zero success
  rate. That is the bar.
- The checked-in anchor clears `0.0` gates. Its metadata is the legacy
  pre-calibration plant (`train_hover: 0.5`, `action_scale: [4.0, 4.0, 3.0]`).

So the anchor is a regression floor, not a goal. A candidate is worth
promoting only when its mean gates cleared is strictly greater than zero and
you have compared it honestly against the expert mean of roughly 2.81. Do
not call a candidate converged just because it beats zero. The expert itself
only finishes the full course in well under one percent of episodes, so
matching it is hard and exceeding it is the actual target.

### Verified CPU behavior-cloning candidate

The repository includes `rl/data/best/bc-candidate/policy.pt`, trained from
61,065 freshly generated GP-expert transitions. Under the same 150-episode
protocol it clears `2.1866666666666665` gates on average with population std
`0.9047037575299933`. Its metadata uses hover thrust `0.27` and action scales
`[0.6, 0.6, 0.6]`. The adjacent `manifest.json` and `eval.csv` record its
provenance and per-episode results.

This candidate beats the zero-gate anchor and exceeds the expert-minus-one-std
threshold, but remains below the expert mean and completes no full courses.
Treat it as a validated BC warm start and live-flight candidate behind the
camera-native safety fallback, not as converged PPO. A fresh GPU PPO run should
start from this behavior-cloning path and undergo the same evaluation gate:

```bash
uv run python -m rl.training.train_ppo --config configs/default.yaml \
  --run-name gpu-ppo --bc-init rl/data/best/bc-candidate/policy.pt
```

## Live deploy

Once a candidate clears the evaluation bar, hand it to the live loop. This
is a human-in-the-loop step, never an automated acceptance gate.

```bash
make fly-policy POLICY=rl/data/best/my-run/policy.pt
```

That expands to:

```bash
uv run -m rl.deploy --config configs/default.yaml --policy rl/data/best/my-run/policy.pt
```

`make fly-policy` without `POLICY=...` fails intentionally. Never live-fly
the known zero-gate anchor merely to reproduce its baseline; evaluate it
offline with `make rl-eval POLICY=rl/data/policy.pt` when needed.

### What the deploy loop does

The deploy loop closes the loop on the live sim. The ESKF predicts on
commanded thrust plus gyro, not raw accelerometer, and updates from odometry
and native PnP. The trained policy only flies when the estimator is healthy.
When it is not, a camera-native expert takes over.

GateNet is optional in this loop. Native PnP is the primary pose source. The
loop degrades gracefully if `gatenet.pt` is absent.

### Fallback behavior

The loop watches estimator health, covariance trace, coast frames, and gate
progress. When any of those trip, it engages the camera-native fallback and
logs exactly:

```
EXPERT FALLBACK ENGAGED
```

Recovery requires accepted PnP fusion hysteresis, three consecutive accepted
fresh PnP frames with a healthy EKF. When that holds, control returns to the
policy and the loop logs exactly:

```
POLICY RESUMED
```

If there is no vision at all, the fallback cannot steer. It levels the drone
and holds hover thrust, zero rates plus hover thrust, so the airframe does
not accelerate into the ground. That is the no-vision behavior: a stable
level hover, not navigation.

The deploy also handles flips and race restarts. A tilt over 70 degrees
triggers a two-second level-off at hover thrust, and a race restart reseeds
the EKF and holds hover.

### Live success criteria

Live flight is observed, not automated. A successful handoff flight shows all
of the following.

- The drone arms and climbs on the policy, not stuck in fallback.
- Gate progress lines print: `[deploy] gate N/6 passed`.
- The EKF covariance stays bounded and `POLICY RESUMED` holds without
  immediately dropping back to fallback.
- The candidate clears at least one gate on the live course that the anchor
  does not.

If the loop spends the whole flight in fallback, or never advances past gate
zero, the candidate is not ready. Land, read the log, and retrain. A flight
that only level-hovers is the no-vision fallback doing its job, which means
the vision or estimator pipeline upstream is broken, not the policy.

## Success criteria summary

The full pipeline is done when all of these hold.

1. `make rl-test` passes.
2. `flightlab/calibration.json` exists from `make attitude-harness`.
3. `rl/data/gate_map.json` exists from `make capture-gates`.
4. Native PnP reaches deploy. `rl/data/gatenet.pt` is optional supplemental
   perception and may be produced with `make train-gatenet`.
5. A trained candidate at `rl/data/best/<run>/policy.pt` exists from
   `make rl-train`.
6. `make rl-eval POLICY=rl/data/best/<run>/policy.pt` reports a mean gates
   cleared strictly above zero, compared honestly against the expert mean of
   `2.8066666666666666`.
7. `make fly-policy POLICY=rl/data/best/<run>/policy.pt` flies the candidate
   on the live course and clears at least one gate beyond the anchor.

The checked-in anchor at `rl/data/policy.pt` is never overwritten by this
process. It stays as the regression floor. Candidates live under
`rl/data/best/<run>/` and are selected by passing `POLICY=...` to the
handoff targets.

## Regenerating the baseline

If the contract drifts, or after a deliberate overhaul, regenerate the
frozen baseline. This re-evaluates the anchor and the expert, recomputes the
golden contract and trajectory, and rewrites `rl/data/baseline.json`.

```bash
make rl-baseline    # uv run python scripts/capture_baseline.py
```

It writes to `rl/data/baseline.json` and `.sisyphus/evidence/task-0-baseline.json`.
Do not run this casually. It is the reference everything else compares
against, so regenerating it silently invalidates prior candidate comparisons.

## Troubleshooting

`[deploy] no gate map; aborting`
The race-start burst was never captured. Run `make capture-gates`, click
Race in the sim while it listens, wait for `CAPTURED N gates`, then fly.

`[deploy] no policy at ...`
The policy path does not exist. Either run `make rl-train` to produce one
under `rl/data/best/<run>/policy.pt`, or pass the right `POLICY=` path. For
a throwaway checkpoint, `uv run -m rl.training.train_ppo --smoke 4000`
writes one fast.

`[deploy] WARNING: legacy checkpoint without training metadata`
The policy predates current training metadata. Deploy falls back to legacy
hover and rate-scale assumptions. The attitude harness can validate the
current plant, but does not modify checkpoints. Train a new policy against
the current spec/config so its exported metadata records the current values.

`[deploy] WARNING: checkpoint pitch-rate scale ... != current obs RATE_SCALE`
The policy was trained against a different observation normalization than
the current build. The observation scaling changed under it. Retrain before
trusting it.

`EXPERT FALLBACK ENGAGED` and never resumes
The estimator is unhealthy or PnP is not fusing. Check that the sim is in
TRAINING with odometry, or that vision is reaching the loop. Three accepted
fresh PnP frames are required to resume. If you never see `POLICY RESUMED`,
vision or the estimator is the problem, not the policy.

Loop only level-hovers, no forward motion
That is the no-vision fallback. With no gate detection, the fallback cannot
compute guidance, so it zeroes the rates and holds hover thrust. Fix the
vision pipeline upstream before blaming the policy.

`make capture-gates` prints `NO gate burst captured in window`
The race did not restart inside the 150-second window. Start the listener
first, then restart the race in the sim, and make sure the race actually
restarts rather than resumes.

PPO export parity assertion fires
`export parity failed (max_abs_action_diff=...)` means the exported
standalone actor diverges from the SB3 actor past 1e-4. The exported
`policy.pt` would not match the trained model. This is an internal bug, not
a training issue. Re-run, and if it persists, investigate the weight copy
in `export_policy`, do not fly the artifact.

Candidate beats zero but is far below expert
That is expected early in training and is not convergence. The expert itself
rarely finishes the course. Keep training, watch `progress.csv` for the
gates-cleared trend, and only promote when the candidate is genuinely
competitive with the expert mean, not merely above zero.
