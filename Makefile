.PHONY: i install check sim

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

sim:
	uv run main.py
