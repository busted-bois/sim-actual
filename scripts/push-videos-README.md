# Run archive

Every run the team publishes, kept as shared reference for review and
finetuning. Three things travel together, all keyed by the same run id:

    videos/<member>_vision_<RUN_ID>.mp4          the recording      (Git LFS)
    videos/<member>_vision_<RUN_ID>.json         what the run did   (regular git)
    telemetry/<member>_gp_log_<RUN_ID>_a<N>.csv  40 Hz trace, per attempt

`RUN_ID` is `%Y%m%d_%H%M%S`, minted once per process, so a video and its
telemetry pair **exactly** -- no nearest-timestamp guessing.

The sidecar is the point: it says how far the run got before you download
anything. Gates reached, outcome, attempt count, the git sha that flew it, and
the video's real frame rate.

## Reading it

Do **not** `git checkout videos` -- it swaps your working tree mid-session.
From your normal branch:

```bash
make videos-index                     # every run: target, gates, outcome, length
make videos-sync                      # sidecars + telemetry -> runs/archive/
make videos-get RUN=<run>             # one recording (~65 MB)
make videos-frames RUN=<run> AT=12.5  # write the frame at t=12.5s
```

`videos-index` and `videos-sync` cost **zero LFS bandwidth** -- they read
sidecars and CSVs, which are regular git objects, never video payload. Only
`videos-get` downloads, and it prints the size first.

Frame extraction uses the sidecar's measured `fps_actual`, not the container's
nominal 30 -- the recorder only writes a frame when the camera delivers one, so
seeking on the header value drifts badly.

## Adding your own

From your working branch, with recordings in `runs/videos/`:

```bash
make push-videos       # or: make push-videos-dry, to see what would upload
```

Publishes every local recording, sidecar, and CSV not already here. Re-running
is safe, and each artifact is judged on its own -- a video published before
sidecars existed picks one up on the next run. Needs
[git-lfs](https://git-lfs.com) installed.

## Quota

GitHub's free LFS tier is **1 GB storage / 1 GB per month bandwidth** for the
org -- roughly 15 videos at ~65 MB each. Only the mp4s count against it;
sidecars and telemetry are ordinary git objects. Prefer publishing runs worth
keeping over every run, and `make videos-index` before `videos-get`.
