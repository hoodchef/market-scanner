import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

def ttl_cache(ttl_seconds: int):
    """
    Simple per-process TTL cache. Not multi-process safe (fine for now).
    Keyed on function args + sorted kwargs.
    """
    def decorator(fn: Callable):
        store: Dict[Tuple[Any, ...], Tuple[Any, float]] = {}

        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store:
                val, exp = store[key]
                if now < exp:
                    return val
            val = fn(*args, **kwargs)
            store[key] = (val, now + ttl_seconds)
            return val
        return wrapped
    return decorator
