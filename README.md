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

1. Start **FlightSim.exe** from your AI-GP Simulator install (e.g. `C:\Users\trung\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe`), log in, and **start a qualifier / flight session** (not just the main menu).
2. Then run `make sim`.

If you see `No MAVLink heartbeat received`:

- Confirm the sim session is still running (not paused at menus).
- Stop any other `make sim` / python process using port 14550 (`netstat -ano | findstr 14550`).
- With the session active, run `make mavlink-probe` — you should see `HEARTBEAT` lines within 60s.

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

## More Info

See [Instructions.md](docs/Instructions.md) for full setup details, system requirements, competition timeline, and technical specifications.

See [qualifier-playbook.md](docs/qualifier-playbook.md) for live-run troubleshooting and diagnostics.
