.PHONY: i install check sim capture-gates fly hover dynamics capture dataset dataset-pose collect-sweep synth-data train-gatepose train-ppo fly-policy fly-vision fly-vision-only fly-servo vision rl-test

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

# Module 2 (pose): collect frames + auto-labeled corner KEYPOINTS from live sim.
# Start the race first (needs the broadcast gate map for auto-labeling).
dataset-pose:
	uv run -m rl.dataset --pose

# Balanced diversity sweep: fly to varied viewpoints around ALL gates, recording
# range-gated auto-labels. Start the race, then run. Appends to the dataset.
collect-sweep:
	uv run -m rl.collect_sweep

# Synthetic gate-image factory (Python stand-in for Matt's Unity renderer):
# render thousands of domain-randomized, auto-labeled gates into the pose dataset.
# No sim needed. This is what makes the detector transfer to the real gate.
synth-data:
	uv run -m rl.synth_data

# Module 3 (pose): fine-tune YOLO11-pose (yolo11n-pose.pt) -> rl/data/gate_pose.pt
train-gatepose:
	uv run -m rl.gatepose --train

# Module 8: train PPO policy over the curriculum -> rl/data/policy.pt
train-ppo:
	uv run -m rl.train_ppo

# Module 8: fly the trained policy on the live sim.
fly-policy:
	uv run -m rl.deploy

# Vision + tuned control: fly2's stable controller, gate target from YOLO->PnP
# ->tracker (vision in a background thread). Start the race, then run this.
fly-vision:
	uv run -m rl.fly_vision

# TRUE vision-only: target comes solely from YOLO->PnP tracks, the broadcast
# gate map is never read for navigation. Proves it flies on vision, not coords.
fly-vision-only:
	uv run -m rl.fly_vision --vision-only

# Box-based visual-servo flight: fly THROUGH the gate the detector SEES (YOLO
# bbox center, no PnP, no broadcast map). Works with the synthetic-trained
# detector before corners converge. Start the race, then run.
fly-servo:
	uv run -m rl.fly_servo

# Standalone live vision window (no flight): watch YOLO detect gates + corners.
# Safe to run anytime the sim is up. Press q to quit.
vision:
	uv run -m rl.vision_view

# Offline self-tests for every module (no live sim needed).
rl-test:
	uv run -m rl.dataset --selftest
	uv run -m rl.gatepose --selftest
	uv run -m rl.pnp --selftest
	uv run -m rl.perception --selftest
	uv run -m rl.gate_tracker --selftest
	uv run -m rl.collect_sweep --selftest
	uv run -m rl.synth_data --selftest
	uv run -m rl.ekf --selftest
	uv run -m rl.observation --selftest
	uv run -m rl.env --selftest
	uv run -m rl.deploy --selftest
