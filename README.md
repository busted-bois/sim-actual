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
<img width="2203" height="959" alt="ANDURIL team pic" src="https://github.com/user-attachments/assets/e4d5c707-7f95-4caf-91de-04f9e5022625" />

## More Info

See [Instructions.md](Instructions.md) for full setup details, system requirements, competition timeline, and technical specifications.
