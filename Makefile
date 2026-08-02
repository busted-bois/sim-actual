.PHONY: train-vq2 vq2-observe vq2-fly i install check test sim view auto auto-gp control-flight free-port probe est-selftest doc-context doc-validate doc-update capture-gates fly fly-vision fly-vision-est hover dynamics capture dataset train-gatenet train-ppo fly-policy rl-train rl-eval rl-baseline rl-test attitude-harness log-demos train-bc

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

# --- Documentation (auto-sync on push to main; local: CURSOR_API_KEY required) ----
doc-context:
	bash scripts/doc-context.sh > .doc-context.txt

doc-validate:
	bash scripts/doc-validate.sh docs/main-documentation.md $(MAIN_SHA)

doc-update: doc-context
	cd scripts && npm ci && cd ..
	node scripts/update-main-documentation.mjs

test:
	uv run python -m unittest tests.test_preflight tests.test_pilot_gates_passed tests.test_race_monitor tests.test_auto_flight tests.test_fly2_course tests.test_vision_nav tests.test_vision_nav_pilot tests.test_vq2_pose tests.test_vq2_pilot tests.test_vision_rx_auto_logs tests.test_lap_log tests.test_gp_pilot tests.test_gp_signs tests.test_calibration tests.test_flightlab tests.test_flightlab_bus tests.test_mavlink_client tests.test_gp_expert tests.test_bc_pipeline tests.test_deploy_gate_map tests.test_gate_corners_cv tests.test_gate_pnp tests.test_gate_detector tests.test_vq2_observation tests.test_vq2_env -v

sim:
	uv run main.py

# Manual keyboard flight — WASD move, Q/E turn, R/F up/down, C level, L auto-land.
manual:
	uv run manual.py

# Auto flight — continuous overnight retry; Ctrl+C stops
auto:
	uv run auto.py

# Smooth GP control flight — YOLO/PnP vision -> GPPilot guidance (~8 km/h
# cruise speed loop, collision backoff). Single-shot: arm + WAIT_FOR_* inside
# the pilot. Not overnight `make auto`.
control-flight:
	uv run auto_gp.py

# Back-compat alias for the original AndurilGP-style entry name.
auto-gp: control-flight

# Passive live vision window (camera + YOLO gate detection). No MAVLink, no
# arming -- works under the VQ2 telemetry block. Just watch the CNN detect.
view:
	uv run -m simulator.vision_view

# Kill a stale make auto/make sim client still holding UDP 14550
free-port:
ifeq ($(OS),Windows_NT)
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/free-mavlink-port.ps1
else
	bash scripts/free-mavlink-port.sh
endif

# Passive MAVLink probe: per-message rates + IMU conventions. Run in Training
# AND in VQ2 to see exactly what the event block removes.
probe:
	uv run -m simulator.telemetry_probe

# Offline selftest for the VQ2 state estimator (ESKF + tilt/mag/baro/landmarks).
est-selftest:
	uv run -m simulator.state_estimator --selftest

# --- Fly the course (odometry + gate map, measured-dynamics controller) -------
# Gate map is captured at race START as a one-shot burst. If rl/data/gate_map.json
# is missing, run `make capture-gates` and start the race WHILE it listens.
capture-gates:
	uv run scripts/capture_gates.py

# Fly the full 6-gate course (resets, arms, flies). Start the race first.
fly:
	uv run -m rl.experts.fly2 --mode course

# Fly the course from VISION ONLY -- YOLO gate detection + PnP, no gate map,
# no hardcoded coordinates. Start the race first. Under the VQ2 block the
# IMU+vision estimator takes over automatically (odometry absent).
fly-vision:
	uv run -m rl.experts.fly2 --mode vision

# VQ2 dress rehearsal in Training mode: fly on the estimator even though
# odometry exists; odometry only feeds the shadow error CSV (rl/data/shadow_*).
fly-vision-est:
	uv run -m rl.experts.fly2 --mode vision --est

# Hold a stable hover (sanity check the controller).
hover:
	uv run -m rl.experts.fly2 --mode hover --seconds 8

# Measure the sim's attitude/thrust response (open-loop characterization).
dynamics:
	uv run scripts/rl_diag_dynamics.py

# Attitude inner-loop harness (Spec B). Writes flightlab/calibration.json +
# signs.json consumed by rl/core/spec, rl/environment/env, and the GP expert. Sim must be in
# a TRAINING session.
attitude-harness:
	uv run python -m flightlab.run_attitude

# --- RL pipeline (Modules 1-8) ------------------------------------------------
# Overridable RL handoff settings (see docs/rl-live-handoff.md). POLICY has no
# default because the checked-in zero-gate anchor is unsafe for live flight.
CONFIG ?= configs/default.yaml
POLICY ?=
RUN ?= ppo
STEPS ?= 300000

# Module 1: connect to live sim, dump telemetry snapshot + gate map.
capture:
	uv run -m rl.environment.sim_interface

# Module 2: collect frames + auto-labeled masks from the live sim.
dataset:
	uv run -m rl.perception.dataset

# Module 3: train GateNet U-Net -> rl/data/gatenet.pt
train-gatenet:
	uv run -m rl.perception.gatenet

# GP expert demos: AndurilGP guidance rollouts in the internal env
# -> rl/data/gp_demos.npz (offline, no live sim).
log-demos:
	uv run -m rl.training.log_demos

# BC pretrain on GP demos -> rl/data/policy_bc.pt (train-ppo warm-starts
# from it automatically when present).
train-bc:
	uv run -m rl.training.train_bc

# Module 8: train PPO policy over the curriculum -> rl/data/policy.pt
train-ppo:
	uv run -m rl.training.train_ppo

# Module 8: fly the trained policy on the live sim (see docs/rl-live-handoff.md).
# Pass --config and --policy so candidate selection is explicit.
fly-policy:
	@test -n "$(POLICY)" || (echo "POLICY is required; use rl/data/best/<run>/policy.pt" >&2; exit 2)
	uv run -m rl.deploy --config $(CONFIG) --policy $(POLICY)

# Parameterized PPO training -> rl/data/best/$(RUN)/policy.pt
rl-train:
	uv run python -m rl.training.train_ppo --config $(CONFIG) --run-name $(RUN) --steps $(STEPS)

# Deterministic multi-seed eval against the frozen T0 protocol.
rl-eval:
	@test -n "$(POLICY)" || (echo "POLICY is required; pass the candidate or explicit anchor path" >&2; exit 2)
	uv run python -m rl.training.evaluate --policy $(POLICY)

# Regenerate the frozen rl/data/baseline.json contract (anchor + expert).
rl-baseline:
	uv run python scripts/capture_baseline.py

# Offline self-tests for every module (no live sim needed).
rl-test:
	uv run -m simulator.state_estimator --selftest
	uv run -m rl.perception.dataset --selftest
	uv run -m rl.perception.gatenet --selftest
	uv run -m rl.perception.pnp --selftest
	uv run -m rl.estimation.ekf --selftest
	uv run -m rl.core.observation --selftest
	uv run -m rl.environment.env --selftest
	uv run -m rl.deploy --selftest

# --- VQ2 vision-only RL (17-gate course, no gate map, no odometry) -----------
# Train the recurrent policy in the camera-rendered surrogate. ~1-3 h on this
# box; --quick for a smoke run. Offline: no simulator needed.
train-vq2:
	uv run -m rl.training.train_vq2

# READ-ONLY live check. Runs perception -> VQ2 observation against the live sim
# and logs every field. Never arms, never commands. Run this BEFORE vq2-fly.
vq2-observe:
	uv run -m rl.deploy_vq2 --observe --seconds 30

# Fly the trained policy on the live simulator, vision + IMU only.
# Signs/gain are UNCALIBRATED defaults -- override with RL_SIGN_* / RL_RATE_GAIN.
vq2-fly:
	uv run -m rl.deploy_vq2 --fly
