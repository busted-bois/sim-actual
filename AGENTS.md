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

## Codebase Rules

- Do NOT use python (.py) for anything other than actual simulator logic. Use shell scripts (ex. bash, powershell) for other tasks, ONLY AS NEEDED.
- Makefile is the centralized calling file with scripts, NOT pyproject.toml. If needed, install it in powershell with `choco install make`. On MacOS / Linux, it should automatically work by default.
- Before commiting any files, use the deslop skill to verify the changes and ensure we aren't going overboard.
