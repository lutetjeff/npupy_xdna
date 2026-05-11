from __future__ import annotations

import multiprocessing
import os
import signal
import time

import pytest

from npupy_xdna.runtime.npu_lock import LOCK_FILE, NpuLockTimeoutError, npu_exclusive


def _hold_lock_for(duration_s: float) -> None:
    with npu_exclusive(timeout_s=60.0):
        time.sleep(duration_s)


def _try_acquire_with_timeout(timeout_s: float, result_queue: multiprocessing.Queue) -> None:
    try:
        with npu_exclusive(timeout_s=timeout_s):
            result_queue.put("acquired")
    except NpuLockTimeoutError as exc:
        result_queue.put(f"timeout:{exc}")


def test_lock_acquired_and_released():
    with npu_exclusive(timeout_s=5.0):
        assert LOCK_FILE.exists()
    assert LOCK_FILE.exists()


def test_two_processes_serialize():
    hold_seconds = 0.2
    p1 = multiprocessing.Process(target=_hold_lock_for, args=(hold_seconds,))
    t_start = time.monotonic()
    p1.start()
    time.sleep(0.05)

    q: multiprocessing.Queue = multiprocessing.Queue()
    p2 = multiprocessing.Process(target=_try_acquire_with_timeout, args=(5.0, q))
    p2.start()

    p1.join(timeout=5.0)
    p2.join(timeout=5.0)
    t_end = time.monotonic()

    wall = t_end - t_start
    assert wall >= hold_seconds, f"Wall time {wall:.3f}s < hold_seconds {hold_seconds}"
    assert q.get(timeout=1.0) == "acquired"


def test_lock_auto_releases_on_process_sigkill():
    hold_for = 30.0
    p = multiprocessing.Process(target=_hold_lock_for, args=(hold_for,))
    p.start()
    time.sleep(0.1)
    os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=2.0)

    acquired = False
    with npu_exclusive(timeout_s=2.0):
        acquired = True
    assert acquired, "Lock should auto-release after SIGKILL"


def test_timeout_raises_npu_lock_timeout_error():
    hold_for = 30.0
    p = multiprocessing.Process(target=_hold_lock_for, args=(hold_for,))
    p.start()
    time.sleep(0.1)

    try:
        with pytest.raises(NpuLockTimeoutError) as exc_info:
            with npu_exclusive(timeout_s=0.15):
                pass
        assert "PID" in str(exc_info.value)
    finally:
        p.terminate()
        p.join(timeout=2.0)
