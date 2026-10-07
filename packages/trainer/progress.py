"""Process-aware tqdm positioning for concurrent trainer jobs."""

from __future__ import annotations

import os

from tqdm.auto import tqdm as _tqdm


_POSITION_ENV = "TRAINER_TQDM_POSITION"
_SHARED_LOCK_ENV = "TRAINER_TQDM_SHARED_LOCK"


class tqdm(_tqdm):
    def __init__(self, *args, position: int | None = None, **kwargs):
        configured_position = os.environ.get(_POSITION_ENV)
        if configured_position is None and position is None:
            resolved_position = None
        else:
            base_position = int(configured_position or "0")
            resolved_position = base_position + (0 if position is None else position)
        super().__init__(*args, position=resolved_position, **kwargs)


def configure_job_progress(position: int, lock) -> None:
    os.environ[_POSITION_ENV] = str(position)
    os.environ[_SHARED_LOCK_ENV] = "1"
    tqdm.set_lock(lock)


def has_shared_progress_lock() -> bool:
    return os.environ.get(_SHARED_LOCK_ENV) == "1"
