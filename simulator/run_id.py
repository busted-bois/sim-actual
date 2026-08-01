"""One id shared by everything a single run writes.

The mp4 and the flight CSVs used to stamp themselves independently -- the video
at its first camera frame, the log at the first flying tick -- so pairing a
recording with its telemetry meant guessing at timestamps that could be minutes
apart. Both now derive their filename from RUN_ID instead, so a video and its
logs share one exact key.

Minted once at import (process start), never recomputed: importing this a second
time from another module hands back the same value. Keep it that way -- the id
is only useful because every writer in the process agrees on it.

Format is unchanged from what those writers used before (%Y%m%d_%H%M%S), so the
push scripts' stamp regex, the .gitignore globs, and the existing filename
assertions all still match.
"""

import time

#: Stamp for this process, e.g. "20260728_195640".
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
