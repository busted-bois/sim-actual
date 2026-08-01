# anduril-sim

Autonomous drone racing pilot for the [AI Grand Prix](https://www.theaigrandprix.com/) competition.

## Quickstart

- Requires [uv](https://docs.astral.sh/uv/)

- On Windows, ensure you are using powershell, then install make with `choco install make`.


```bash
make          # install deps
make check    # lint + format
make sim      # run the pilot
```

## Project Structure

```
docs/                   # Competition docs
simulator/              # Simulator package
main.py                 # Entry point
Makefile                # Setup, lint, run targets
pyproject.toml          # Dependencies (uv)
uv.lock                 # Lockfile
skills-lock.json        # Agent skills lockfile
```


## Run recordings

Every run writes `runs/videos/vision_<RUN_ID>.mp4`, a `vision_<RUN_ID>.json`
sidecar recording how it went (gates reached, outcome, git sha), and
`rl/data/gp_log_<RUN_ID>_a<N>.csv` per attempt — all sharing one run id, all
local and gitignored. To add yours to the shared archive:

```bash
make push-videos
```

To read the team's archive without switching branches (`videos-index` and
`videos-sync` spend no LFS bandwidth):

```bash
make videos-index                     # every run: target, gates, outcome, length
make videos-get RUN=<run>             # one recording (~65 MB)
make videos-frames RUN=<run> AT=12.5  # the frame at t=12.5s
```

That publishes them to the **`videos`** branch through [Git LFS](https://git-lfs.com)
(install it first) -- one folder holding every member's runs, for review and
finetuning. Re-running is safe; recordings already on the branch are skipped, and
your current branch and working tree are never touched. `make push-videos-dry`
shows what would upload without pushing.

To pull videos back down, see the README on the [`videos`](https://github.com/busted-bois/sim-actual/tree/videos/videos) branch.

## More Info

See [docs/main-documentation.md](docs/main-documentation.md) for a living overview of what is on **`main`** (updated when features merge).

See [docs/Instructions.md](docs/Instructions.md) for full setup details, system requirements, competition timeline, and technical specifications.

# Team Members
- Ryan Yang, Ram Rao, Samyak Kakatur, Kunal Shrivastav, Trung Ngyuen, David Vayntrub, Yat Chun Wong, Sameer Faisal
<img width="2203" height="959" alt="ANDURIL team pic" src="https://github.com/user-attachments/assets/e4d5c707-7f95-4caf-91de-04f9e5022625" />
