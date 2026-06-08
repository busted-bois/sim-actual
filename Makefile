.PHONY: i install check test validate-log vision-smoke mavlink-probe race-timing-probe preflight tracking-smoke sim

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
	uv run main.py $(if $(COLLISION_RESET),--collision-reset,)
