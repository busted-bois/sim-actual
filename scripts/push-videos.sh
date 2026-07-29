#!/usr/bin/env bash
# Publish local run recordings (runs/videos/*.mp4) to the shared `videos`
# branch via Git LFS. All work happens in a throwaway worktree, so the current
# working tree and branch are never touched.
set -euo pipefail

BRANCH="${BRANCH:-videos}"
REMOTE="${REMOTE:-origin}"
DRY_RUN=0
SOURCE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --source) SOURCE="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        --remote) REMOTE="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${SOURCE:=$ROOT/runs/videos}"

# git narrates checkouts on stderr; swallow it unless the command actually fails.
quiet_git() {
    local out
    if ! out="$(git "$@" 2>&1)"; then
        printf '%s\n' "$out" >&2
        return 1
    fi
}

# --- Is there anything to push? ----------------------------------------------

if [ ! -d "$SOURCE" ]; then
    echo "No recordings yet: $SOURCE does not exist. Fly a run first."
    exit 0
fi

videos=()
while IFS= read -r f; do
    videos+=("$f")
done < <(find "$SOURCE" -maxdepth 1 -type f -name '*.mp4' | sort)

if [ "${#videos[@]}" -eq 0 ]; then
    echo "No .mp4 files in $SOURCE. Fly a run first."
    exit 0
fi

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs is not installed. Get it from https://git-lfs.com (or 'brew install git-lfs'), then re-run." >&2
    exit 1
fi
git lfs install --local >/dev/null

member="$(git config user.name || true)"
[ -n "$member" ] || member="${USER:-unknown}"
# The member name ends up in filenames, so keep it boring.
member="$(printf '%s' "$member" | tr -c 'A-Za-z0-9' '-' | tr -s '-' | sed 's/^-//; s/-$//' | tr 'A-Z' 'a-z')"
[ -n "$member" ] || member="unknown"

# mtime as YYYYmmdd_HHMMSS -- GNU date and BSD/macOS stat spell this differently.
file_stamp() {
    date -r "$1" +%Y%m%d_%H%M%S 2>/dev/null && return 0
    stat -f %Sm -t %Y%m%d_%H%M%S "$1" 2>/dev/null && return 0
    date +%Y%m%d_%H%M%S
}

# Size from metadata; `wc -c` would stream 65 MB per video to count bytes.
file_size() {
    stat -c %s "$1" 2>/dev/null && return 0
    stat -f %z "$1" 2>/dev/null && return 0
    wc -c < "$1"
}

# --- Check out the videos branch somewhere harmless ---------------------------

WORK="$(mktemp -d "${TMPDIR:-/tmp}/push-videos-XXXXXXXX")"
rmdir "$WORK"  # git worktree add wants to create it itself
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"

TEMP_BRANCH=""
cleanup() {
    cd "$ROOT"
    git worktree remove --force "$WORK" >/dev/null 2>&1 || true
    git worktree prune >/dev/null 2>&1 || true
    # Only after the worktree is gone, or the branch is still checked out.
    [ -z "$TEMP_BRANCH" ] || git branch -D "$TEMP_BRANCH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Videos already on the branch stay as LFS pointers instead of being downloaded.
# This has to hold for every checkout below -- the rebase in the retry path
# touches teammates' files too, and pushing must never cost a re-download of
# everyone else's runs. It only disables the smudge (checkout) direction; our
# own files still go through the clean filter into LFS on `git add`.
export GIT_LFS_SKIP_SMUDGE=1

if git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
    quiet_git fetch "$REMOTE" "+${BRANCH}:$REMOTE_REF"
    quiet_git worktree add --detach "$WORK" "$REMOTE_REF"
else
    echo "Branch '$BRANCH' does not exist on $REMOTE yet -- creating it."
    quiet_git worktree add --no-checkout --detach "$WORK"
    # An orphan needs a branch name; it is local and thrown away by cleanup,
    # since the push below targets $BRANCH by refspec.
    TEMP_BRANCH="push-videos-$$"
    quiet_git -C "$WORK" switch --orphan "$TEMP_BRANCH"
fi

DEST="$WORK/videos"
mkdir -p "$DEST"

# The videos are LFS payload; the branch is useless without this rule.
ATTR_LINE='videos/*.mp4 filter=lfs diff=lfs merge=lfs -text'
if ! grep -qxF "$ATTR_LINE" "$WORK/.gitattributes" 2>/dev/null; then
    printf '%s\n' "$ATTR_LINE" >> "$WORK/.gitattributes"
fi

[ -f "$DEST/README.md" ] || cp "$ROOT/scripts/push-videos-README.md" "$DEST/README.md"

# --- Copy in whatever is new -------------------------------------------------

added=()
added_bytes=0
skipped=0
for v in "${videos[@]}"; do
    base="$(basename "$v" .mp4)"
    # Pre-timestamp recordings were all called vision.mp4; stamp those from
    # mtime so two different runs cannot collide on one name and vanish.
    case "$base" in
        *_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]) ;;
        *) base="${base}_$(file_stamp "$v")" ;;
    esac
    target="$DEST/${member}_${base}.mp4"
    if [ -e "$target" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    [ "$DRY_RUN" -eq 1 ] || cp "$v" "$target"
    added+=("$(basename "$target")")
    added_bytes=$((added_bytes + $(file_size "$v")))
done

if [ "${#added[@]}" -eq 0 ]; then
    echo "Already published: all ${#videos[@]} local recording(s) are on '$BRANCH'."
    exit 0
fi

mb="$(awk -v b="$added_bytes" 'BEGIN { printf "%.1f", b / 1048576 }')"
echo "Publishing ${#added[@]} recording(s) as '$member' (~${mb} MB; $skipped already on the branch):"
printf '  %s\n' "${added[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "--dry-run: nothing was copied, committed, or pushed."
    exit 0
fi

git -C "$WORK" add -A -- videos .gitattributes
if [ -z "$(git -C "$WORK" status --porcelain)" ]; then
    echo "Nothing to commit."
    exit 0
fi
git -C "$WORK" commit -qm "Add ${#added[@]} run recording(s) from $member"

echo
echo "Uploading to $REMOTE/$BRANCH (${mb} MB through LFS -- this takes a while)..."
if ! git -C "$WORK" push "$REMOTE" "HEAD:refs/heads/$BRANCH"; then
    # Someone else published while we were packing.
    echo "Push rejected, rebasing onto the latest '$BRANCH' and retrying..."
    quiet_git fetch "$REMOTE" "+${BRANCH}:$REMOTE_REF"
    quiet_git -C "$WORK" rebase "$REMOTE_REF"
    git -C "$WORK" push "$REMOTE" "HEAD:refs/heads/$BRANCH"
fi

echo
echo "Done -- ${#added[@]} recording(s) now on '$BRANCH'."
