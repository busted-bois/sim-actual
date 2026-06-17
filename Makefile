.PHONY: i install check sim test verify-reset rl-sync train eval

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

sim:
	uv run main.py

test:
	uv run pytest -q

# ---- Round-2 residual-RL targets ----
rl-sync:           ## install RL deps (torch / sb3 / gymnasium)
	uv sync --group rl

verify-reset:      ## P0: does sim_reset clear a fly-away? (sim must be running)
	uv run python -m simulator.verify_reset

flight-test:       ## P0.5: reset + timed bare-pilot lap; reports gates cleared + lap time
	uv run python -m simulator.flight_test

train:             ## P2/P3: train the residual policy -> policy.pt (sim + rl deps)
	uv run --group rl python -m simulator.rl_train train

eval:              ## P3: eval policy vs bare pilot, report deploy gate (sim + rl deps)
	uv run --group rl python -m simulator.rl_train eval
