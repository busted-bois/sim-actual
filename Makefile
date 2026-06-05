.PHONY: i install check test vision-smoke mavlink-probe sim

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

sim:
	uv run main.py
