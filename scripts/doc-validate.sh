#!/usr/bin/env bash
# Validate docs/main-documentation.md structure and main-commit header.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DOC="${1:-docs/main-documentation.md}"
EXPECTED_SHA="${2:-}"

if [[ ! -f "$DOC" ]]; then
  echo "doc-validate: missing $DOC" >&2
  exit 1
fi

REQUIRED_SECTIONS=(
  "## 1. Purpose"
  "## 2. Quickstart"
  "### 2.5 Choosing an entry point"
  "### 2.6 Troubleshooting"
  "## 3. Repository layout"
  "## 4. System architecture"
  "## 5. \`shared_data\` schema"
  "## 6. Module reference"
  "## 7. Module reference"
  "## 8. External interfaces"
  "## 9. Control modes"
  "## 10. Makefile targets"
  "## 11. Dependencies"
  "## 12. Current capabilities"
  "## 13. How to update this doc"
  "## Changelog"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
  if ! grep -qF "$section" "$DOC"; then
    echo "doc-validate: missing section: $section" >&2
    exit 1
  fi
done

if ! grep -qE '^\| \*\*Last updated\*\* \|' "$DOC"; then
  echo "doc-validate: missing Last updated header" >&2
  exit 1
fi

DOC_SHA="$(grep -E '^\| \*\*Main commit\*\*' "$DOC" | sed -n 's/.*`\([0-9a-f]\{7,40\}\)`.*/\1/p' | head -1)"
if [[ -z "$DOC_SHA" ]]; then
  echo "doc-validate: could not parse Main commit from header" >&2
  exit 1
fi

if [[ -n "$EXPECTED_SHA" ]]; then
  if [[ "$DOC_SHA" != "$EXPECTED_SHA" ]] && [[ "${DOC_SHA:0:7}" != "${EXPECTED_SHA:0:7}" ]]; then
    echo "doc-validate: Main commit '$DOC_SHA' != expected '$EXPECTED_SHA'" >&2
    exit 1
  fi
fi

# Cross-check simulator/*.py files mentioned in layout block exist on disk.
LAYOUT_BLOCK="$(awk '/^```$/{c++} c==2{exit} c==1{print}' "$DOC" | grep 'simulator/' | sed 's/ .*//' || true)"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  path="$(echo "$line" | tr -d '[:space:]')"
  [[ "$path" == simulator/* ]] || continue
  [[ "$path" == *".py" ]] || continue
  if [[ ! -f "$path" ]]; then
    echo "doc-validate: layout lists missing file: $path" >&2
    exit 1
  fi
done <<< "$LAYOUT_BLOCK"

echo "doc-validate: ok ($DOC_SHA)"
