# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Action log: one JSON line per action under ~/.aiida-physbo/log/<kind>-<YYYY-MM>.jsonl.

No aiida import. Writing never raises: the log is a by-product of the action.
"""
import datetime
import json
import os


def log_dir() -> str:
    return os.environ.get("PHYSBO_LOG_DIR") or os.path.join(os.path.expanduser("~"), ".aiida-physbo", "log")


def append_jsonl(kind: str, record: dict):
    """append `record` (with a timestamp) to the log of `kind`; return the path or None on failure."""
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        path = os.path.join(log_dir(), f"{kind}-{now:%Y-%m}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"time": now.isoformat(timespec="seconds"), **record}, default=str) + "\n")
        return path
    except Exception:  # noqa: BLE001  never fail the action because of the log
        return None
