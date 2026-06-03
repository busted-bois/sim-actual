# anduril-sim

Autonomous drone racing pilot for the [AI Grand Prix](https://www.theaigrandprix.com/) competition.

## Quickstart

```bash
uv sync
uv run main.py
```

## Project Structure

```
controller.py    Flight control logic
main.py          Entry point
mavlink_rx.py    MAVLink message receiver
setup.py         Component initialization
timesync.py      Time synchronization
vision_rx.py     Vision data receiver
```

The simulator binaries live in `simulator/` (gitignored).

## More Info

See [Instructions.md](Instructions.md) for full setup details, system requirements, competition timeline, and technical specifications.
