"""Serialize native numerical work and process-global thread-pool changes.

threadpoolctl changes BLAS/OpenMP pools for the whole process. Changing a pool
while another lane is inside LAPACK can hang native workers. Every BLAS/model
entry point shares this lock; file/network work and Pro inference stay outside.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import wraps

from threadpoolctl import threadpool_limits

_NATIVE_LOCK = threading.RLock()
_LOCAL = threading.local()


@contextmanager
def native_compute(limits=2):
    """Hold ownership through calculation *and* thread-pool restoration.

    Nested calls inherit their outer call's limit without resizing a live pool.
    The reentrant lock supports composite fits and optimizer/IC helper calls.
    """
    if isinstance(limits, bool) or not isinstance(limits, int) or not 1 <= limits <= 8:
        raise ValueError("Native computation requires between one and eight threads")
    with _NATIVE_LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        _LOCAL.depth = depth + 1
        try:
            if depth:
                yield
            else:
                # Construction, computation, and restoration all need ownership.
                with threadpool_limits(limits=limits):
                    yield
        finally:
            _LOCAL.depth = depth


def native_serialized(limits=2):
    """Protect a numerical entry point, including calls made outside the daemon."""
    def decorate(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            with native_compute(limits=limits(*args, **kwargs) if callable(limits) else limits):
                return function(*args, **kwargs)
        guarded.__native_limits__ = limits
        return guarded
    return decorate
