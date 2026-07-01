.PHONY: i install check test sim free-port doc-context doc-validate doc-update capture-gates fly hover dynamics capture dataset train-gatenet train-ppo fly-policy rl-test

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
	uv run python -m unittest tests.test_preflight tests.test_camera_model tests.test_gate_estimator tests.test_ibvs_comparison tests.test_pilot_profile tests.test_pilot_gates_passed tests.test_race_monitor tests.test_auto_flight tests.test_fly2_course -v

sim:
	uv run main.py

# Auto flight — continuous overnight retry; Ctrl+C stops
auto:
	uv run auto.py

# Kill a stale make auto/make sim client still holding UDP 14550
free-port:
ifeq ($(OS),Windows_NT)
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/free-mavlink-port.ps1
else
	bash scripts/free-mavlink-port.sh
endif

# A/B gate-racing profiles (Windows: powershell -File scripts/sim-ab.ps1 -Profile main)
sim-ab-main:
	PILOT_AB=main uv run main.py

sim-ab-branch:
	PILOT_AB=branch uv run main.py

sim-ab-jacobian:
	PILOT_AB=jacobian uv run main.py

sim-ab-main-jacobian:
	PILOT_AB=main-jacobian uv run main.py

sim-ab-branch-legacy:
	PILOT_AB=branch-legacy uv run main.py

# --- Fly the course (odometry + gate map, measured-dynamics controller) -------
# Gate map is captured at race START as a one-shot burst. If rl/data/gate_map.json
# is missing, run `make capture-gates` and start the race WHILE it listens.
capture-gates:
	uv run -m rl.capture_gates

# Fly the full 6-gate course (resets, arms, flies). Start the race first.
fly:
	uv run -m rl.fly2 --mode course

# Hold a stable hover (sanity check the controller).
hover:
	uv run -m rl.fly2 --mode hover --seconds 8

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
	uv run -m rl.dataset --selftest
	uv run -m rl.gatenet --selftest
	uv run -m rl.pnp --selftest
	uv run -m rl.ekf --selftest
	uv run -m rl.observation --selftest
	uv run -m rl.env --selftest
	uv run -m rl.deploy --selftest
