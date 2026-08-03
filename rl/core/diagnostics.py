"""Console and persistent file diagnostics for long-running RL processes."""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

DEFAULT_LOG_ROOT = Path("logs")
_CONSOLE_FMT = "[%(tag)s] %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s [%(tag)s] %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _TagFilter(logging.Filter):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    def filter(self, record: logging.LogRecord) -> bool:
        record.tag = getattr(record, "tag", None) or self.tag
        return True


class RunnerLog:
    """Write tagged lines to a flushed console and timestamped log file."""

    def __init__(
        self,
        tag: str = "vq2",
        log_dir: str | os.PathLike[str] | None = None,
        stream: TextIO | None = None,
        tick_period_s: float = 2.0,
        file_name: str | None = None,
    ):
        self.tag = tag
        self.tick_period_s = float(tick_period_s)
        self._closed = False
        self._last_tick: float | None = None

        directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_ROOT / tag
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.file_path = str(directory / (file_name or f"{tag}_{stamp}.log"))

        self._logger = logging.getLogger(f"rl.runner.{tag}.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        tag_filter = _TagFilter(tag)
        self._stream = stream if stream is not None else sys.stdout
        console = logging.StreamHandler(self._stream)
        console.setFormatter(logging.Formatter(_CONSOLE_FMT))
        console.addFilter(tag_filter)

        file_handler = logging.FileHandler(self.file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATEFMT))
        file_handler.addFilter(tag_filter)

        self._logger.addHandler(console)
        self._logger.addHandler(file_handler)

    def _emit(self, level: int, message: str, component: str | None) -> None:
        if not self._closed:
            extra = {"tag": component} if component else None
            self._logger.log(level, message, extra=extra)

    def info(self, message: str, *, component: str | None = None) -> None:
        self._emit(logging.INFO, message, component)

    def warn(self, message: str, *, component: str | None = None) -> None:
        self._emit(logging.WARNING, message, component)

    def tick(self, message: str, *, component: str | None = None) -> bool:
        """Emit a periodic diagnostic at most once per configured interval."""
        if self._closed:
            return False
        now = time.monotonic()
        if self._last_tick is not None and now - self._last_tick < self.tick_period_s:
            return False
        self._last_tick = now
        self._emit(logging.INFO, message, component)
        return True

    def fatal(self, message: str) -> None:
        """Log the active exception and its traceback to both outputs."""
        if not self._closed:
            self._logger.error(message, exc_info=True, extra={"tag": self.tag})

    def reset_tick_clock(self) -> None:
        self._last_tick = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handler in tuple(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
        self._stream.flush()

    def __enter__(self) -> RunnerLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.fatal(f"terminating: {exc}")
        self.close()
        return False
