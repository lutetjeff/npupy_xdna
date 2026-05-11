from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE = Path("/tmp/npupy_xdna.npu.lock")
LOG_FILE = Path("/home/lutet/ece511/npupy_xdna/results/npu_lock.log")
RETRY_INTERVAL_S = 0.05


class NpuLockTimeoutError(Exception):
    pass


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    pid = os.getpid()
    line = f"{ts} PID={pid} {msg}\n"
    try:
        with LOG_FILE.open("a") as f:
            f.write(line)
    except OSError:
        pass


def _read_lock_pid(fd: int) -> int | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 32).strip()
        return int(data) if data else None
    except (OSError, ValueError):
        return None


def _write_pid(fd: int) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
    except OSError:
        pass


@contextmanager
def npu_exclusive(timeout_s: float = 600.0):
    """Exclusive NPU lock via fcntl.flock. Retries every RETRY_INTERVAL_S up to
    timeout_s. Raises NpuLockTimeoutError if timeout exceeded. Auto-released on
    process death (fd-bound flock semantics)."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o666)
    deadline = time.monotonic() + timeout_s
    acquired = False
    holder_pid: int | None = None

    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                holder_pid = _read_lock_pid(fd)
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise NpuLockTimeoutError(
                        f"PID {holder_pid} holds lock — timed out after {timeout_s}s"
                    )
                time.sleep(RETRY_INTERVAL_S)

        _write_pid(fd)
        _log(f"ACQUIRED npu_exclusive lock (fd={fd})")
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            _log(f"RELEASED npu_exclusive lock (fd={fd})")
        try:
            os.close(fd)
        except OSError:
            pass
