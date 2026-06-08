# anduril-sim

Autonomous drone racing pilot for the [AI Grand Prix](https://www.theaigrandprix.com/) competition.

## Quickstart

- Requires [uv](https://docs.astral.sh/uv/)

- On Windows, ensure you are using powershell, then install make with `choco install make`.


```bash
make          # install deps
make check    # lint + format
make sim      # run the pilot
```

## Project Structure

```
docs/                   # Competition docs
simulator/              # Simulator package
main.py                 # Entry point
Makefile                # Setup, lint, run targets
pyproject.toml          # Dependencies (uv)
uv.lock                 # Lockfile
skills-lock.json        # Agent skills lockfile
```

## Autonomous color-navigation pilot

A toggleable autonomy stack detects the orange gate, flies through gates in
sequence, follows the blue path as a fallback, and stops safely at the end of the
course. Toggle it in `settings.json` (`autonomy.enabled` / `autonomy.algorithm`).

See [docs/Autonomy.md](docs/Autonomy.md) for the design, all tunables, and full
testing instructions (unit tests, offline color tuning, and in-sim run).

```bash
uv run python -m pytest tests/ -q     # unit tests, no simulator needed
uv run python tools/vision_preview.py # offline HSV tuning preview
```

## More Info

See [Instructions.md](Instructions.md) for full setup details, system requirements, competition timeline, and technical specifications.
