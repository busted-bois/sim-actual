.PHONY: i install check sim manual probe

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

sim:
	uv run main.py

# Manual keyboard flight — WASD move, Q/E turn, SPACE up, X down, R/F speed. Close window to exit.
manual:
	uv run manual.py

# Diagnostic — print which MAVLink messages the sim streams (run with a session active).
probe:
	uv run python scripts/mavlink_probe.py
