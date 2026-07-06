"""Flight event recorder for the state estimator — JSONL, replayable.

Records the EXACT event stream the estimator consumed (imu, thrust,
landmark, collision, reset) plus observer events it did not (odometry
truth, 1 Hz est_ckpt live poses), one JSON object per line:

    {"k": "imu", "tw": <monotonic>, "d": {ax..az, gx..gz, ..., time_us}}

log() is a deque append (~µs) so it is safe on the MAVLink RX thread; a
daemon writer thread drains the queue and does json.dumps + file I/O off
the hot path. Volume is ~3-4 MB/min at sim rates.

Replay: simulator/est_replay.py feeds the core events back through
EstimatorCore.apply() in file order — which equals consumption order,
because every log() happens under the estimator lock immediately next to
the core call that consumed the event.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time


def _jsonable(x):
    """numpy scalars/arrays -> plain floats/lists (live payloads are already
    plain, but selftests feed numpy)."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):  # np.ndarray / np scalar
        return x.tolist()
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    return float(x)


class EstRecorder:
    """Append-only JSONL event log with an off-thread writer."""

    def __init__(self, path: str, flush_s: float = 0.25):
        self.path = path
        self._q: collections.deque = collections.deque()
        self._flush_s = flush_s
        self._closed = False
        self._f = None
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def log(self, kind: str, payload: dict | None = None):
        """O(1) enqueue — no serialization, no I/O. Safe on the RX thread."""
        if self._closed:
            return
        self._q.append((kind, time.monotonic(), payload))

    def _drain(self):
        wrote = False
        while self._q:
            kind, tw, payload = self._q.popleft()
            ev = {"k": kind, "tw": round(tw, 6)}
            if payload is not None:
                ev["d"] = _jsonable(payload)
            self._f.write(json.dumps(ev) + "\n")
            wrote = True
        if wrote:
            self._f.flush()

    def _writer(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = open(self.path, "w")
        while not self._closed:
            time.sleep(self._flush_s)
            self._drain()
        self._drain()  # final drain requested by close()

    def close(self):
        """Drain everything and flush. Idempotent — MUST be called explicitly
        before os._exit(), which skips atexit/finally."""
        if self._closed:
            return
        self._closed = True
        self._thread.join(timeout=5.0)  # writer does the final drain
        if self._f is not None:
            self._f.flush()
            self._f.close()


class ListRecorder:
    """In-memory recorder with the same interface — for selftests."""

    def __init__(self):
        self.events: list[dict] = []

    def log(self, kind: str, payload: dict | None = None):
        ev = {"k": kind, "tw": 0.0}
        if payload is not None:
            ev["d"] = _jsonable(payload)
        self.events.append(ev)

    def close(self):
        pass


def read_log(path: str):
    """Yield events from a JSONL log in file order."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
