#!/usr/bin/env bash
# Emit git + repo context for docs/main-documentation.md auto-update.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DOC="${DOC_PATH:-docs/main-documentation.md}"
MAIN_SHA="${MAIN_SHA:-$(git rev-parse HEAD)}"
MAIN_MSG="$(git log -1 --format='%s' "$MAIN_SHA")"
MAIN_DATE="$(git log -1 --format='%ci' "$MAIN_SHA" | cut -d' ' -f1)"

PREV_SHA=""
if [[ -f "$DOC" ]]; then
  PREV_SHA="$(grep -E '^\| \*\*Main commit\*\*' "$DOC" | sed -n 's/.*`\([0-9a-f]\{7,40\}\)`.*/\1/p' | head -1)"
fi

echo "=== MAIN_SHA ==="
echo "$MAIN_SHA"
echo "=== MAIN_SHORT_SHA ==="
echo "${MAIN_SHA:0:7}"
echo "=== MAIN_SUBJECT ==="
echo "$MAIN_MSG"
echo "=== MAIN_DATE ==="
echo "$MAIN_DATE"
echo "=== PREV_DOC_SHA ==="
echo "${PREV_SHA:-none}"
echo "=== TODAY ==="
date -u +%Y-%m-%d

WATCH_PATHS=(simulator/ rl/ main.py auto.py Makefile tests/ pyproject.toml SPEC.md scripts/)

echo "=== DIFF_STAT ==="
if [[ -n "$PREV_SHA" ]] && git cat-file -e "${PREV_SHA}^{commit}" 2>/dev/null; then
  git diff --stat "$PREV_SHA" "$MAIN_SHA" -- "${WATCH_PATHS[@]}" 2>/dev/null || true
else
  git show --stat "$MAIN_SHA" -- "${WATCH_PATHS[@]}" 2>/dev/null || true
fi

echo "=== DIFF_NAMES ==="
if [[ -n "$PREV_SHA" ]] && git cat-file -e "${PREV_SHA}^{commit}" 2>/dev/null; then
  git diff --name-status "$PREV_SHA" "$MAIN_SHA" -- "${WATCH_PATHS[@]}" 2>/dev/null || true
else
  git diff-tree --no-commit-id --name-status -r "$MAIN_SHA" -- "${WATCH_PATHS[@]}" 2>/dev/null || true
fi

echo "=== DIFF_PATCH ==="
if [[ -n "$PREV_SHA" ]] && git cat-file -e "${PREV_SHA}^{commit}" 2>/dev/null; then
  git diff "$PREV_SHA" "$MAIN_SHA" -- "${WATCH_PATHS[@]}" 2>/dev/null | head -c 120000 || true
else
  git show "$MAIN_SHA" -- "${WATCH_PATHS[@]}" 2>/dev/null | head -c 120000 || true
fi

echo "=== SIMULATOR_FILES ==="
git ls-files simulator/*.py 2>/dev/null | sort || true

echo "=== RL_FILES ==="
git ls-files rl/*.py 2>/dev/null | sort || true

echo "=== TEST_FILES ==="
git ls-files tests/*.py 2>/dev/null | sort || true

echo "=== ROOT_FILES ==="
git ls-files main.py auto.py Makefile pyproject.toml SPEC.md README.md AGENTS.md 2>/dev/null | sort || true

echo "=== MAKEFILE_TARGETS ==="
grep -E '^[a-zA-Z0-9_.-]+:' Makefile 2>/dev/null | sed 's/:.*//' | grep -v '^\.' | sort -u || true

echo "=== COMMIT_LOG_SINCE_PREV ==="
if [[ -n "$PREV_SHA" ]] && git cat-file -e "${PREV_SHA}^{commit}" 2>/dev/null; then
  git log --oneline "${PREV_SHA}..${MAIN_SHA}" 2>/dev/null || true
else
  git log --oneline -10 "$MAIN_SHA" 2>/dev/null || true
fi
