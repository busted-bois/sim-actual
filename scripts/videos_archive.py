"""Read the shared run archive without checking out the videos branch.

The archive lives on an orphan `videos` branch. The obvious way in --
`git checkout videos` -- swaps your working tree mid-tuning-session, which is
exactly what you do not want. Everything here reads through `git show` or a
throwaway worktree, so the branch you are on never changes.

    uv run scripts/videos_archive.py index          # what runs exist, and how they went
    uv run scripts/videos_archive.py sync           # copy sidecars + telemetry locally
    uv run scripts/videos_archive.py get <run>      # download one mp4 (~65 MB)

Normally reached through `make videos-index` / `videos-sync` / `videos-get`.

On cost: `git fetch` moves the small regular objects (sidecars, CSVs) and only
LFS *pointers* for the videos, so `index` and `sync` spend zero LFS bandwidth.
That matters -- the org's free tier is 1 GB/month, about 15 videos. Only `get`
spends any, and it prints the size first.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "runs", "archive")
# Never let a checkout pull video payload we did not explicitly ask for.
ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")


def git(*args, check=True, cwd=ROOT):
    r = subprocess.run(
        ("git",) + args, cwd=cwd, env=ENV, capture_output=True, text=True
    )
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout


def fetch(remote, branch):
    ref = f"refs/remotes/{remote}/{branch}"
    if subprocess.run(
        ("git", "ls-remote", "--exit-code", "--heads", remote, branch),
        cwd=ROOT, capture_output=True, env=ENV,
    ).returncode != 0:
        raise SystemExit(
            f"No '{branch}' branch on {remote} yet. Publish a run with `make push-videos`."
        )
    git("fetch", remote, f"+{branch}:{ref}")
    return ref


def ls(ref, prefix):
    out = git("ls-tree", "-r", "--name-only", ref, "--", prefix)
    return [ln for ln in out.splitlines() if ln.strip()]


def load_sidecars(ref):
    runs = []
    for path in ls(ref, "videos/"):
        if not path.endswith(".json"):
            continue
        try:
            runs.append((path, json.loads(git("show", f"{ref}:{path}"))))
        except (ValueError, SystemExit):
            continue
    return runs


def pointer_size(ref, path):
    """Bytes of an LFS-backed file, straight from its pointer -- no download."""
    for line in git("show", f"{ref}:{path}", check=False).splitlines():
        if line.startswith("size "):
            return int(line.split()[1])
    return None


def cmd_index(args):
    ref = fetch(args.remote, args.branch)
    runs = load_sidecars(ref)
    videos = [p for p in ls(ref, "videos/") if p.endswith(".mp4")]
    telem = ls(ref, "telemetry/")

    if not videos:
        print("Archive is empty.")
        return 0

    described = {os.path.basename(p)[: -len(".json")] for p, _ in runs}
    rows = []
    for path, meta in sorted(runs, key=lambda r: r[0]):
        stem = os.path.basename(path)[: -len(".json")]
        tot = meta.get("totals") or {}
        att = meta.get("attempts") or []
        outcomes = [a.get("outcome") for a in att if a.get("outcome")]
        vid = meta.get("video") or {}
        secs = ""
        if vid.get("frames") and vid.get("fps_actual"):
            secs = f"{vid['frames'] / vid['fps_actual']:.0f}s"
        rows.append((
            stem,
            meta.get("target") or "?",
            str(tot.get("best_gates")) if tot.get("best_gates") is not None else "-",
            str(tot.get("attempts") or len(att) or "-"),
            outcomes[-1] if outcomes else "-",
            secs or "-",
            str(sum(1 for t in telem if os.path.basename(t).startswith(stem.split("_vision_")[0]) and stem.split("_vision_")[-1] in t)),
        ))

    head = ("RUN", "TARGET", "GATES", "ATT", "OUTCOME", "LEN", "CSV")
    width = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)] if rows else []
    if rows:
        print("  ".join(h.ljust(width[i]) for i, h in enumerate(head)))
        print("  ".join("-" * width[i] for i in range(len(head))))
        for r in rows:
            print("  ".join(r[i].ljust(width[i]) for i in range(len(head))))
        print()

    bare = [os.path.basename(v)[: -len(".mp4")] for v in videos]
    bare = [b for b in bare if b not in described]
    if bare:
        print(f"{len(bare)} recording(s) with no sidecar (published before run metadata existed):")
        for b in bare:
            print(f"  {b}")
        print()

    total = sum(pointer_size(ref, v) or 0 for v in videos)
    print(
        f"{len(videos)} recording(s), {len(telem)} telemetry file(s). "
        f"Video payload {total / 1048576:.0f} MB -- not downloaded."
    )
    print("Fetch one with: make videos-get RUN=<run>")
    return 0


def cmd_sync(args):
    ref = fetch(args.remote, args.branch)
    n = 0
    for prefix, sub in (("videos/", "videos"), ("telemetry/", "telemetry")):
        for path in ls(ref, prefix):
            if path.endswith(".mp4"):
                continue  # payload only on explicit `get`
            dst = os.path.join(ARCHIVE, sub, os.path.basename(path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8", newline="") as fh:
                fh.write(git("show", f"{ref}:{path}"))
            n += 1
    print(f"{n} sidecar/telemetry file(s) -> {os.path.relpath(ARCHIVE, ROOT)} (0 MB of LFS)")
    return 0


def cmd_get(args):
    ref = fetch(args.remote, args.branch)
    name = args.run if args.run.endswith(".mp4") else args.run + ".mp4"
    path = f"videos/{os.path.basename(name)}"
    if path not in ls(ref, "videos/"):
        print(f"not in the archive: {path}", file=sys.stderr)
        print("Run `make videos-index` to see what is there.", file=sys.stderr)
        return 1

    dst = os.path.join(ARCHIVE, "videos", os.path.basename(path))
    if os.path.exists(dst):
        print(f"already local: {os.path.relpath(dst, ROOT)}")
        return 0

    size = pointer_size(ref, path)
    mb = (size or 0) / 1048576
    print(f"Downloading {os.path.basename(path)} ({mb:.0f} MB) through Git LFS...")
    if size and mb > args.max_mb:
        print(
            f"Refusing: {mb:.0f} MB is over --max-mb {args.max_mb}. "
            f"Re-run with --max-mb {int(mb) + 1} to override.",
            file=sys.stderr,
        )
        return 1

    work = os.path.join(tempfile.gettempdir(), "videos-get-" + uuid.uuid4().hex[:8])
    try:
        git("worktree", "add", "--detach", work, ref)
        subprocess.run(
            ("git", "lfs", "pull", "--include", path),
            cwd=work, env=os.environ, check=True,
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(work, path), dst)
    finally:
        git("worktree", "remove", "--force", work, check=False)
        git("worktree", "prune", check=False)
    print(f"-> {os.path.relpath(dst, ROOT)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("index", "sync", "get"))
    ap.add_argument("run", nargs="?", default="")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="videos")
    ap.add_argument("--max-mb", type=float, default=200.0)
    args = ap.parse_args(argv)
    if args.mode == "get" and not args.run:
        ap.error("get needs a run name: make videos-get RUN=<run>")
    return {"index": cmd_index, "sync": cmd_sync, "get": cmd_get}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
