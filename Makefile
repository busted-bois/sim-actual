.PHONY: i install check test sim view auto auto-gp control-flight bluevision free-port probe est-selftest est-validate shadow-validate doc-context doc-validate doc-update capture-gates fly fly-vision fly-vision-est hover dynamics thrust-id capture dataset train-gatenet train-ppo fly-policy eval-policy rl-flight rl-test attitude-harness log-demos train-bc rl2-reset-bench rl2-log-demos rl2-run rl2-diff rl2-train-bc rl2-train rl2-eval rl2-log rl2-fly-gate rl2-gp-smoke rl2-list-demos rl2-reset-demos

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
	uv run python -m unittest tests.test_preflight tests.test_pilot_gates_passed tests.test_race_monitor tests.test_auto_flight tests.test_fly2_course tests.test_vision_nav tests.test_vision_nav_pilot tests.test_vq2_pose tests.test_vq2_pilot tests.test_vision_rx_auto_logs tests.test_lap_log tests.test_gp_pilot tests.test_gp_signs tests.test_calibration tests.test_flightlab tests.test_flightlab_bus tests.test_mavlink_client tests.test_gp_expert tests.test_bc_pipeline tests.test_deploy_gate_map tests.test_gate_corners_cv tests.test_gate_pnp tests.test_gate_detector tests.test_blue_line_vision -v

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

# Same GP flight as control-flight. The dual-cyan corridor detector is always
# on (vision_rx), so this is only a name that says "watch the blue line": it feeds
# the no-gate ribbon fallback and the post-pass SEARCH direction cue.
bluevision: control-flight

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

# Validate the ecl/EKF2 estimator against VQ1 GROUND TRUTH. Run on the legacy
# VQ1 sim (telemetry ON): flies the GP pilot, runs EclEkf as a live shadow fed
# IMU + PnP vision velocity, and logs EKF-fused vs IMU-dead-reckon vs truth
# (LOCAL_POSITION_NED/ATTITUDE). Ctrl+C to stop and print the RMSE report.
# Re-report a saved log:  make est-validate ARGS="--report logs/ecl_validate_<boot>.jsonl"
est-validate:
	uv run -m simulator.ecl_validate $(ARGS)

# SHADOW-MODE validation: run Abhay's known-good telemetry pilot (VQ1 sim)
# UNMODIFIED while our VQ2 estimator observes IMU+camera only, then compare vs
# ground truth. Set ABHAY_DIR if his repo isn't at the default Desktop path.
# Re-report:  make shadow-validate ARGS="--report logs/shadow_<ts>.jsonl"
shadow-validate:
	uv run -m simulator.shadow_validate $(ARGS)

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

# Measure + FIT + persist translational dynamics (thrust_accel, hover, drag,
# v_max) into flightlab/calibration.json so rl/env.py trains on the real thrust
# curve and drag. Sim must be in a TRAINING session (odometry velocity needed).
# `thrust-id` is an alias. --dry-run measures without writing.
dynamics thrust-id:
	uv run -m rl.dynamics_id

# Attitude inner-loop harness (Spec B). Writes flightlab/calibration.json +
# signs.json consumed by rl/spec, rl/env, and the GP expert. Sim must be in
# a TRAINING session.
attitude-harness:
	uv run python -m flightlab.run_attitude

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

# GP expert demos: AndurilGP guidance rollouts in the internal env
# -> rl/data/gp_demos.npz (offline, no live sim).
log-demos:
	uv run -m rl.log_demos

# BC pretrain on GP demos -> rl/data/policy_bc.pt (train-ppo warm-starts
# from it automatically when present).
train-bc:
	uv run -m rl.train_bc

# Module 8: train PPO over the curriculum -> rl/data/policy.pt. Uses the GPU
# when available (--device cpu to force CPU); warm-starts from policy_bc.pt when
# present; TensorBoard logs + best/checkpoint models under rl/data/{tb,best,ckpts}.
#   tensorboard --logdir rl/data/tb
train-ppo:
	uv run -m rl.train_ppo

# VQ2 vision-only RL flight: fly rl/data/policy.pt on the live sim using
# YOLO+PnP gate pose + IMU state (NO gate map, NO odometry). Run the sim in a
# TRAINING session on the VQ2 course. Live rate calibration via env vars:
#   RL_RATE_SCALE=0.4 RL_SIGN_ROLL=-1 RL_SIGN_PITCH=1 RL_SIGN_YAW=-1 make rl-flight
rl-flight:
	uv run auto_rl.py

# Evaluate the trained policy offline (gates chained + completion per stage).
# Headless, no live sim. Prints reproducible per-stage metrics from policy.pt.
eval-policy:
	uv run -m rl.eval_policy

# Module 8: fly the trained policy on the live sim.
fly-policy:
	uv run -m rl.deploy

# --- VQ2 real-sim RL (rl/vq2/) --------------------------------------------
# DE-RISK FIRST: benchmark automated episodic reset speed + re-align reliability
# on the live VQ2 sim. If resets are slow/flaky, real-sim RL is not viable.
rl2-reset-bench:
	uv run -m rl.vq2.reset --cycles $(or $(CYCLES),10)

# Collect BC demos by taping the proven GP pilot flying the real sim. Saves a
# standalone replayable trajectory to rl/data/vq2/saves/<name>.npz AND appends to
# the BC bootstrap (demos.npz). Name it:  make rl2-log-demos ARGS="--name run1"
rl2-log-demos:
	uv run -m rl.vq2.log_demos $(ARGS)

# Replay a saved trajectory's ACTIONS open-loop (schedules on the SIM clock) AND
# record the replay's own trace to <name>_replay.npz. Start/restart the race:
#   make rl2-run ARGS="<name>"
rl2-run:
	uv run -m rl.vq2.run_traj $(ARGS)

# Diff a replay trace vs its original to find WHERE/WHY the drone turned
# differently (first diverging signal: command / sim-time / gyro / gate-pose).
#   make rl2-diff ARGS="<name>"
rl2-diff:
	uv run -m rl.vq2.diff_traj $(ARGS)

# Behaviour-clone a PPO policy on the demos -> rl/data/vq2/policy_bc.zip (no sim).
rl2-train-bc:
	uv run -m rl.vq2.train_bc $(ARGS)

# Incremental closed-loop training in the REAL VQ2 sim. BC-inits the policy from
# ALL accumulated successful segments (+ demos.npz), then flies PPO online; every
# flight that reaches a NEW gate is saved to success/gate{N}/ as it happens.
# Resume a checkpoint:  make rl2-train ARGS="--resume rl/data/vq2/vq2_ppo.zip"
rl2-train:
	uv run -m rl.vq2.train $(ARGS)

# List accumulated successful trajectories per gate.
rl2-list-demos:
	uv run -m rl.vq2.success --list

# Delete ALL stored demonstrations (successes + demos.npz) to rebuild from scratch.
# Keep the expert bootstrap:  make rl2-reset-demos ARGS="--keep-demos"
rl2-reset-demos:
	uv run -m rl.vq2.success --reset $(ARGS)

# Evaluate a trained VQ2 policy in the real sim.
rl2-eval:
	uv run -m rl.vq2.eval $(ARGS)

# Read the training episode log: gate-pass rate + per-term reward breakdown.
# Verifies the drone registers gate passes and the reward function is working.
rl2-log:
	uv run -m rl.vq2.read_log $(ARGS)

# ISOLATION TEST: fly straight at the gate with a hand-coded vision servo (GP
# steering law) through the RL env's command path -- NO neural net. If this
# threads gate 1, the vision->control plumbing is sound and the RL failure is the
# learned policy. Start the race, then run.
rl2-fly-gate:
	uv run -m rl.vq2.fly_to_gate $(ARGS)

# RESIDUAL-RL STAGE 1: does the GP base fly through GPFlightInterface (the class
# the RL env flies with), with NO residual? Should clear gates like control-flight.
rl2-gp-smoke:
	uv run -m rl.vq2.gp_smoke $(ARGS)

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
