.PHONY: i install check sim capture-gates fly hover dynamics rate-id thrust-id capture dataset train-gatenet train-ppo train-ppo-quick bc-check sign-check fly-policy rl-test

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

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

# Measure command-rate -> actual-body-rate mapping (for RL env calibration).
rate-id:
	uv run -m rl.rate_id

# Measure thrust -> vertical-accel mapping (for RL env translation calibration).
thrust-id:
	uv run -m rl.thrust_id

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
# (BC warm-start from the geometric expert, then PPO per-stage with success
# reporting). Use train-ppo-quick for a fast end-to-end pipeline smoke.
train-ppo:
	uv run -m rl.train_ppo

train-ppo-quick:
	uv run -m rl.train_ppo --quick

# Cheap pre-flight: BC warm-start only + closed-loop baseline on stages 0-1.
# Run BEFORE train-ppo to confirm PPO starts from a competent base (action MSE
# alone doesn't prove it — per-step error compounds over a trajectory).
bc-check:
	uv run -m rl.train_ppo --bc-only

# Module 8: fly the trained policy on the live sim. Run sign-check FIRST on a
# fresh sim session to confirm roll/yaw signs (a flipped sign crashes flight 1).
sign-check:
	uv run -m rl.deploy --sign-check

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
