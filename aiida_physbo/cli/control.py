# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Control-side implementations (daemon start/stop, kill). Every action that does something is logged."""
from .. import logdir


def daemon_status():
    from aiida.engine.daemon.client import get_daemon_client

    client = get_daemon_client()
    info = {"running": client.is_daemon_running}
    if info["running"]:
        try:
            info["workers"] = client.get_numprocesses().get("numprocesses")
        except Exception as exc:  # noqa: BLE001
            info["workers_error"] = str(exc)
        try:
            info["pid"] = client.get_daemon_pid()
        except Exception:  # noqa: BLE001
            pass
    return info


def daemon_start(workers=None, caller="cli"):
    from aiida.engine.daemon.client import get_daemon_client

    client = get_daemon_client()
    if client.is_daemon_running:
        return {"already_running": True, **daemon_status()}
    client.start_daemon(number_workers=int(workers or 1))
    logdir.append_jsonl("action", {"action": "daemon_start", "workers": int(workers or 1), "caller": caller})
    return {"started": True, **daemon_status()}


def daemon_stop(caller="cli"):
    from aiida.engine.daemon.client import get_daemon_client

    client = get_daemon_client()
    if not client.is_daemon_running:
        return {"already_stopped": True, "running": False}
    client.stop_daemon(wait=True)
    logdir.append_jsonl("action", {"action": "daemon_stop", "caller": caller})
    return {"stopped": True, "running": client.is_daemon_running}


def kill(pk, caller="cli"):
    from aiida import orm
    from aiida.engine.processes.control import kill_processes

    node = orm.load_node(int(pk))
    if node.is_terminated:
        return {"pk": node.pk, "already_terminated": True, "state": node.process_state.value}
    kill_processes([node], timeout=10)
    logdir.append_jsonl("action", {"action": "kill", "pk": node.pk, "caller": caller})
    return {"pk": node.pk, "killed": node.is_killed, "state": node.process_state.value}
