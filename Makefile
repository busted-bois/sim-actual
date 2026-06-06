.PHONY: i install check test vision-smoke mavlink-probe race-timing-probe preflight tracking-smoke sim

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

vision-smoke:
	uv run python scripts/vision_smoke.py

mavlink-probe:
	uv run python scripts/mavlink_probe.py

race-timing-probe:
	uv run python scripts/race_timing_probe.py

preflight:
	uv run python scripts/preflight.py

tracking-smoke:
	uv run python scripts/tracking_smoke.py

sim:
	uv run main.py
