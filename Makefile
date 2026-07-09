.PHONY: i install check sim test

i install:
	uv sync

check:
	uv run ruff check --fix .
	uv run ruff format .

sim:
	uv run main.py

test:
	uv run -m unittest discover -s tests -v
