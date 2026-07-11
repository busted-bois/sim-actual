# Doc update agent instructions

You update **only** `docs/main-documentation.md` for what is merged on `main` at the commit in context.

## Rules

- Living doc for **`main` only** — not feature branches unless merged.
- Preserve section structure §1–14, TOC, markdown anchors (`#25-choosing-an-entry-point`, etc.).
- Be concise; sacrifice grammar per AGENTS.md.
- Do **not** invent files, modules, or Makefile targets — use context lists only.
- Flag `simulator/` modules committed but **not wired** into `make sim` / `main.py`.
- Update **Last updated** (today UTC from context), **Main commit** (full SHA + subject).
- Add one **Changelog** row at top for this merge.
- Refresh: §3 layout, §5 `shared_data`, §6 `simulator/`, §7 `rl/`, §10 Makefile, §12 capabilities/gaps, §2.5 entry points, §2.6 troubleshooting if new failure modes apply.
- §13: note doc is auto-synced by `.github/workflows/update-main-documentation.yml` on push to `main`.

## Output

Edit `docs/main-documentation.md` in place. Write the complete file. Do not edit other paths.

## Context

The following block is machine-generated repo context (diff, file tree, Makefile targets):
