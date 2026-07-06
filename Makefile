.PHONY: i install check sim view probe est-selftest est-replay capture-gates fly fly-vision fly-vision-est hover dynamics capture dataset train-gatenet train-ppo fly-policy rl-test

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

sim:
	uv run main.py

# Passive live vision window (camera + YOLO gate detection). No MAVLink, no
# arming -- works under the VQ2 telemetry block. Just watch the CNN detect.
view:
	uv run -m simulator.vision_view

# Passive MAVLink probe: per-message rates + IMU conventions. Run in Training
# AND in VQ2 to see exactly what the event block removes.
probe:
	uv run -m simulator.telemetry_probe

# Offline selftest for the VQ2 state estimator (ESKF + tilt/mag/baro/landmarks).
est-selftest:
	uv run -m simulator.state_estimator --selftest
	uv run -m simulator.est_replay --selftest

# Replay a recorded flight log offline: metrics + PASS/FAIL + plot PNG.
#   make est-replay LOG=rl/data/est_log_YYYYmmdd_HHMMSS.jsonl
est-replay:
	uv run -m simulator.est_replay $(LOG) --plot

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
	uv run -m simulator.est_replay --selftest
	uv run -m rl.dataset --selftest
	uv run -m rl.gatenet --selftest
	uv run -m rl.pnp --selftest
	uv run -m rl.ekf --selftest
	uv run -m rl.observation --selftest
	uv run -m rl.env --selftest
	uv run -m rl.deploy --selftest
