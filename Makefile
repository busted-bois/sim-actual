.PHONY: i install check

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .
