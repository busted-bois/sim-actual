.PHONY: i install check test sim

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

sim:
	uv run main.py
