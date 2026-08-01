#!/usr/bin/env bash
# Publish local run recordings (runs/videos/*.mp4) to the shared `videos`
# branch via Git LFS. All work happens in a throwaway worktree, so the current
# working tree and branch are never touched.
set -euo pipefail

BRANCH="${BRANCH:-videos}"
REMOTE="${REMOTE:-origin}"
DRY_RUN=0
SOURCE=""
# An overnight run can leave hundreds of MB of CSV; publish a sane slice.
MAX_TELEM_MB="${MAX_TELEM_MB:-25}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --max-telemetry-mb) MAX_TELEM_MB="$2"; shift ;;
        --source) SOURCE="$2"; shift ;;
        --telemetry-dir) TELEM_DIR="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        --remote) REMOTE="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${SOURCE:=$ROOT/runs/videos}"
: "${TELEM_DIR:=$ROOT/rl/data}"

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
TELEM="$WORK/telemetry"
mkdir -p "$DEST" "$TELEM"

# Only the mp4s are LFS payload. The sidecars and CSVs deliberately fall through
# to regular git objects: they are small, and that keeps them greppable with
# `git grep` and readable with `git show` without spending LFS bandwidth.
ATTR_LINE='videos/*.mp4 filter=lfs diff=lfs merge=lfs -text'
# gp_pilot opens its CSV with newline="", so it writes CRLF on every platform.
CSV_ATTR='telemetry/*.csv text eol=lf'
for line in "$ATTR_LINE" "$CSV_ATTR"; do
    if ! grep -qxF "$line" "$WORK/.gitattributes" 2>/dev/null; then
        printf '%s\n' "$line" >> "$WORK/.gitattributes"
    fi
done

[ -f "$DEST/README.md" ] || cp "$ROOT/scripts/push-videos-README.md" "$DEST/README.md"

# --- Copy in whatever is new -------------------------------------------------
# Each artifact is judged on its own. Keying the whole run off the mp4 would
# mean a video published before sidecars existed could never gain one.

added=()          # mp4s -> LFS
side=()           # sidecars + telemetry -> regular git
lfs_bytes=0
reg_bytes=0
telem_bytes=0
skipped=0
capped=0
max_telem_bytes=$((MAX_TELEM_MB * 1048576))

copy_in() {  # src dst -> records it as a regular-git artifact
    [ "$DRY_RUN" -eq 1 ] || cp "$1" "$2"
    side+=("${2#"$WORK"/}")
    reg_bytes=$((reg_bytes + $(file_size "$1")))
}

for v in "${videos[@]}"; do
    base="$(basename "$v" .mp4)"
    # Pre-timestamp recordings were all called vision.mp4; stamp those from
    # mtime so two different runs cannot collide on one name and vanish.
    case "$base" in
        *_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]) ;;
        *) base="${base}_$(file_stamp "$v")" ;;
    esac
    stem="${member}_${base}"

    if [ -e "$DEST/${stem}.mp4" ]; then
        skipped=$((skipped + 1))
    else
        [ "$DRY_RUN" -eq 1 ] || cp "$v" "$DEST/${stem}.mp4"
        added+=("${stem}.mp4")
        lfs_bytes=$((lfs_bytes + $(file_size "$v")))
    fi

    # Sidecar: what the run actually did. Written next to the mp4 by run_meta.
    src_json="$(dirname "$v")/${base}.json"
    [ -f "$src_json" ] || src_json="$(dirname "$v")/$(basename "$v" .mp4).json"
    if [ -f "$src_json" ] && [ ! -e "$DEST/${stem}.json" ]; then
        copy_in "$src_json" "$DEST/${stem}.json"
    fi

    # Telemetry: the CSVs this run wrote. Same run id as the video, so the
    # glob is exact rather than a nearest-timestamp guess.
    runid="${base#vision_}"
    for c in "$TELEM_DIR"/gp_log_"${runid}"_a*.csv; do
        [ -f "$c" ] || continue
        dst="$TELEM/${member}_$(basename "$c")"
        [ ! -e "$dst" ] || continue
        csz="$(file_size "$c")"
        if [ $((telem_bytes + csz)) -gt "$max_telem_bytes" ]; then
            capped=$((capped + 1))
            continue
        fi
        telem_bytes=$((telem_bytes + csz))
        copy_in "$c" "$dst"
    done
done

if [ "${#added[@]}" -eq 0 ] && [ "${#side[@]}" -eq 0 ]; then
    echo "Already published: all ${#videos[@]} local recording(s) are on '$BRANCH'."
    exit 0
fi

lfs_mb="$(awk -v b="$lfs_bytes" 'BEGIN { printf "%.1f", b / 1048576 }')"
reg_mb="$(awk -v b="$reg_bytes" 'BEGIN { printf "%.1f", b / 1048576 }')"
mb="$lfs_mb"
echo "Publishing as '$member' ($skipped recording(s) already on the branch):"
[ "${#added[@]}" -eq 0 ] || {
    echo "  ${#added[@]} recording(s), ~${lfs_mb} MB -> Git LFS"
    printf '    %s\n' "${added[@]}"
}
[ "${#side[@]}" -eq 0 ] || {
    echo "  ${#side[@]} sidecar/telemetry file(s), ~${reg_mb} MB -> regular git (no LFS quota)"
    printf '    %s\n' "${side[@]}"
}
[ "$capped" -eq 0 ] || echo "  ($capped CSV(s) skipped: over --max-telemetry-mb ${MAX_TELEM_MB})"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "--dry-run: nothing was copied, committed, or pushed."
    exit 0
fi

git -C "$WORK" add -A -- videos telemetry .gitattributes
if [ -z "$(git -C "$WORK" status --porcelain)" ]; then
    echo "Nothing to commit."
    exit 0
fi
if [ "${#added[@]}" -eq 0 ]; then
    msg="Add sidecars/telemetry for ${#side[@]} file(s) from $member"
else
    msg="Add ${#added[@]} run recording(s) from $member"
fi
git -C "$WORK" commit -qm "$msg"

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
if [ "${#added[@]}" -eq 0 ]; then
    echo "Done -- ${#side[@]} sidecar/telemetry file(s) now on '$BRANCH'."
else
    echo "Done -- ${#added[@]} recording(s) (+${#side[@]} sidecar/telemetry) now on '$BRANCH'."
fi
