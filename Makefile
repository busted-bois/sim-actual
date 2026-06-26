.PHONY: i install check sim capture-gates fly hover dynamics capture dataset train-gatenet train-ppo fly-policy rl-test demo-jacobian fly-jacobian

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

# BC from fly2 --log demos -> rl/data/policy.pt
train-bc:
	uv run -m rl.bc_dataset --demo rl/data/demo.jsonl

# PPO with BC warm-start (record demo first: fly2 --log rl/data/demo.jsonl)
train-hybrid:
	uv run -m rl.train_ppo --bc rl/data/demo.jsonl

# Module 8: fly the trained policy on the live sim.
fly-policy:
	uv run -m rl.deploy

# Jacobian controller demo (no live sim needed).
demo-jacobian:
	uv run -m rl.demo_jacobian --episodes 3 --stage 0 --plot

# Tier-A Jacobian on live FlightSim (gate pursuit + geo_control, no ML).
fly-jacobian:
	uv run -m rl.fly_jacobian --mode course

# Offline self-tests for every module (no live sim needed).
rl-test:
	uv run -m rl.geo_control --selftest
	uv run -m rl.demo_jacobian --selftest
	uv run -m rl.fly_jacobian --selftest
	uv run -m rl.bc_dataset --selftest
	uv run -m rl.dataset --selftest
	uv run -m rl.gatenet --selftest
	uv run -m rl.pnp --selftest
	uv run -m rl.ekf --selftest
	uv run -m rl.observation --selftest
	uv run -m rl.env --selftest
	uv run -m rl.deploy --selftest
