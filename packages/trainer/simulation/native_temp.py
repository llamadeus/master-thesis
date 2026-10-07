"""Per-process temporary directories for native simulator libraries."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


_WORKER_TEMP: tempfile.TemporaryDirectory[str] | None = None


def isolate_native_worker_temp() -> Path | None:
    """Give a Windows worker a private temp namespace for native libraries.

    EPANET's legacy Windows API uses ``_tempnam(NULL, ...)`` for several
    scratch files.  Concurrent processes can otherwise select the same name
    before either process opens it.  POSIX builds use ``mkstemp`` and do not
    need this process-wide environment change.
    """
    if os.name != "nt":
        return None
    global _WORKER_TEMP
    if _WORKER_TEMP is None:
        _WORKER_TEMP = tempfile.TemporaryDirectory(
            prefix=f"thesis-epanet-{os.getpid()}-"
        )
    path = Path(_WORKER_TEMP.name).resolve()
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(path)
    # Python caches gettempdir(); keep Python-side helpers in the same private
    # namespace even if they were imported before the pool initializer ran.
    tempfile.tempdir = str(path)
    return path
