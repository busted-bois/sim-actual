- In all interactions, **be extremely concise and sacrifice grammar for the sake of concision**.

## Git

- Do NOT commit any files unless otherwise instructed. 
- Do NOT assume any file showing as modified on 'git status' (or similar commands) was modified by an Agent. Confirm all file changes with the user before restoring them.
- Do NOT add yourself to the Git History or anything related AT ALL.

## Python

- Your primary method for interacting with anything Python related should be **uv** (https://docs.astral.sh/uv/). **THIS IS IMPORTANT**.
- When running scripts, use **uv run <script_name>** instead of **python3 <script_name>**.

## Plan Mode

- Make the plan extremely concise. **Sacrifice grammar for the sake of concision**.
- At the end of each plan, give me a list of unresolved questions to answer, if any. Make the questions extremely concise. **Sacrifice grammar for the sake of concision**.
- Make all plans multi-phase.
- While working on the plan, ensure that tasks within the plan file are marked as completed once they are completed.

## Diagnosing a flight run

- Every run writes `runs/videos/vision_<RUN_ID>.mp4`, `runs/videos/vision_<RUN_ID>.json` (gates reached, outcome, git sha), and `rl/data/gp_log_<RUN_ID>_a<N>.csv` per attempt. **Same `<RUN_ID>`** — pair by id, not by timestamp.
- **The CSVs are the tuning source of truth**, not the video. Practically every tuned constant in `simulator/gp_pilot.py` cites one by filename.
- `make videos-index` lists the team's shared runs (gates, outcome, length) **without switching branches** and without spending LFS bandwidth. Then `make videos-sync` (sidecars + telemetry), `make videos-get RUN=<run>` (one ~65 MB mp4), `make videos-frames RUN=<run> AT=12.5`.
- Never `git checkout videos` — it swaps the working tree mid-session. The targets above read the branch in place.

## Codebase Rules

- Do NOT use python (.py) for anything other than actual simulator logic. Use shell scripts (ex. bash, powershell) for other tasks, ONLY AS NEEDED.
- Makefile is the centralized calling file with scripts, NOT pyproject.toml. If needed, install it in powershell with `choco install make`. On MacOS / Linux, it should automatically work by default.
- Before commiting any files, use the deslop skill to verify the changes and ensure we aren't going overboard.
- All simulator code logic should be written within the simulator/ directory, and then called from main.py. The main.py should not have command-line arguments for new functionality. The new functionality added should work with the default settings, just from running `make sim` (`uv run sim`).