"""One Penumbra process per data directory.

Opening the service rebuilds index.sqlite and runs lifecycle recovery on the truth files, so a second process on the
same directory would rewrite both under the one already serving (2026-09-25: three `npm run dev` stacks each started
their own Penumbra on ../penumbra/data; the ones that lost port 8790 exited and were restarted every 15 s, each time
rebuilding the live index, until two rebuilds collided). The lock is taken before anything is touched and held until
close(); the operating system drops it if the process dies.
"""
from __future__ import annotations

import os
from pathlib import Path

LOCK_NAME = ".penumbra.lock"


class AlreadyRunning(RuntimeError):
    """Another process holds this data directory."""


def _try_lock(handle) -> bool:
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DataDirLock:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / LOCK_NAME
        self._handle = open(self.path, "a+b")
        if not _try_lock(self._handle):
            self._handle.close()
            self._handle = None
            raise AlreadyRunning(f"another Penombre process is using {data_dir}")

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            _unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None
