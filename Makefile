.PHONY: i install check test sim view auto free-port probe est-selftest doc-context doc-validate doc-update capture-gates fly fly-vision fly-vision-est hover dynamics vertical capture dataset train-gatenet train-ppo fly-policy rl-test

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
	uv run python -m unittest tests.test_preflight tests.test_pilot_gates_passed tests.test_race_monitor tests.test_auto_flight tests.test_fly2_course tests.test_flightlab_bus_race tests.test_vision_nav tests.test_vision_nav_pilot tests.test_vq2_pose tests.test_vq2_pilot tests.test_vision_rx_auto_logs tests.test_lap_log -v

sim:
	uv run main.py

# Auto flight — continuous overnight retry; Ctrl+C stops
auto:
	uv run auto.py

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
	uv run -m rl.capture_gates

# Fly the full 6-gate course (resets, arms, flies). Start the race first.
fly:
	uv run -m rl.fly2 --mode course

# Fly the course from VISION ONLY -- YOLO gate detection + PnP, no gate map,
# no hardcoded coordinates. Start the race first. Under the VQ2 block the
# IMU+vision estimator takes over automatically (odometry absent).
fly-vision:
	uv run -m rl.fly2 --mode vision

# VQ2 dress rehearsal in Training mode: fly on the estimator even though
# odometry exists; odometry only feeds the shadow error CSV (rl/data/shadow_*).
fly-vision-est:
	uv run -m rl.fly2 --mode vision --est

# Hold a stable hover (sanity check the controller).
hover:
	uv run -m rl.fly2 --mode hover --seconds 8

# Vertical-axis flight harness (V1–V4). Start sim + TRAINING session first.
vertical:
	uv run python -m flightlab.run_vertical

# Measure the sim's attitude/thrust response (open-loop characterization).
dynamics:
	uv run -m rl.dynamics_id

# --- RL pipeline (Modules 1-8) ------------------------------------------------
# Module 1: connect to live sim, dump telemetry snapshot + gate map.
capture:
	uv run -m rl.sim_interface

# Module 2: collect frames + auto-labeled masks from the live sim.
dataset:
	uv run -m rl.dataset

# Module 3: train GateNet U-Net -> rl/data/gatenet.pt
train-gatenet:
	uv run -m rl.gatenet

# Module 8: train PPO policy over the curriculum -> rl/data/policy.pt
train-ppo:
	uv run -m rl.train_ppo

# Module 8: fly the trained policy on the live sim.
fly-policy:
	uv run -m rl.deploy

# Offline self-tests for every module (no live sim needed).
rl-test:
	uv run -m simulator.state_estimator --selftest
	uv run -m rl.dataset --selftest
	uv run -m rl.gatenet --selftest
	uv run -m rl.pnp --selftest
	uv run -m rl.ekf --selftest
	uv run -m rl.observation --selftest
	uv run -m rl.env --selftest
	uv run -m rl.deploy --selftest
