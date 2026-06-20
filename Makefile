.PHONY: i install check sim capture-gates fly hover dynamics capture dataset train-gatenet train-ppo fly-policy rl-test

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

validate-log:
	uv run python -m scripts.validate_tracking_log

vision-smoke:
	uv run python -m scripts.vision_smoke

mavlink-probe:
	uv run python -m scripts.mavlink_probe

race-timing-probe:
	uv run python -m scripts.race_timing_probe

preflight:
	uv run python -m scripts.preflight

tracking-smoke:
	uv run python -m scripts.tracking_smoke

sim:
	uv run main.py

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
