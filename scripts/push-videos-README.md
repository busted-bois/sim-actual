# Run recordings

Vision recordings from the sim -- one mp4 per run -- kept as shared reference
for review and finetuning. Stored with [Git LFS](https://git-lfs.com); the
files here are pointers until you pull them.

Naming:

    <member>_vision_<YYYYmmdd_HHMMSS>.mp4

The stamp is the run's first frame and uses the same `%Y%m%d_%H%M%S`
convention as `rl/data/gp_log_*.csv`, so a video pairs with its telemetry by
filename. The log stamps at race start, so expect a few seconds of skew --
pair by nearest, not exact.

## Getting the videos

```bash
git lfs install
git fetch origin videos
git checkout videos
```

That gives you pointer files. Then fetch the payload you actually want --
everything here is ~65 MB per video, so prefer a filter:

```bash
git lfs pull --include "videos/ryan_*"        # one member
git lfs pull --include "videos/*_20260728_*"  # one day
git lfs pull                                  # all of it
```

## Adding your own

From your working branch, with recordings sitting in `runs/videos/`:

```bash
make push-videos
```

It copies every local recording that is not already here, commits, and pushes
to this branch. Re-running is safe: files already published are skipped. Use
`make push-videos-dry` first if you want to see what would be uploaded.
