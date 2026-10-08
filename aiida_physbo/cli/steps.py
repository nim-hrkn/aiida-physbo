# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Node-creating implementations of the CLI (candidates, observe, propose, submit-optimize).

Validation happens here, before anything is stored or submitted. Every action is logged.
"""
import json
import os

import numpy as np

from .. import logdir
from ..calcfunctions import PROPOSE_DEFAULTS


def _node(pk, cls=None, what="node"):
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    try:
        node = orm.load_node(int(pk))
    except NotExistent as exc:
        raise ValueError(f"no node with pk {pk}") from exc
    if cls is not None and not isinstance(node, cls):
        raise ValueError(f"Node<{node.pk}> is a {node.__class__.__name__}, not {what}")
    return node


def _json(text, what):
    if text is None or text == "":
        return None
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{what} must be JSON: {exc}") from exc


def _ints(text, what):
    try:
        return [int(x) for x in str(text).replace(";", ",").split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"{what} must be comma separated integers") from exc


def _label(node, label):
    if label:
        node.label = label
    return node


def _info(node, process, label):
    return {"pk": node.pk, "process_pk": process.pk, "label": label or node.label}


# ---------------------------------------------------------------- candidates
def candidates(file=None, format=None, delimiter=None, skip_header=None, columns=None, grid=None, test_function=None,
               kwargs=None, num=None, names=None, label=None, caller="cli"):
    from aiida import orm

    from .. import objectives
    from ..calcfunctions import candidates_from_file, candidates_from_grid, parse_candidates

    given = [x for x in (file, grid, test_function) if x]
    if len(given) != 1:
        raise ValueError("give exactly one of --file, --grid, --test-function")
    name_list = [n.strip() for n in names.split(",")] if names else None
    meta = {"label": label} if label else {}
    if file:
        path = os.path.abspath(os.path.expanduser(file))
        if not os.path.isfile(path):
            raise ValueError(f"no such file: {path}")
        opts = {k: v for k, v in {"format": format, "delimiter": delimiter, "skip_header": skip_header,
                                  "columns": _ints(columns, "--columns") if columns else None, "names": name_list}.items()
                if v is not None}
        with open(path, "rb") as f:
            X = parse_candidates(f.read(), os.path.basename(path), opts)      # validate before storing anything
        source = orm.SinglefileData(file=path)
        node = candidates_from_file(source, orm.Dict(dict=opts), metadata=meta)
        how = {"source": "file", "file": path}
    else:
        if grid:
            spec = _json(grid, "--grid")
            if not isinstance(spec, dict) or not {"min", "max", "num"} <= set(spec):
                raise ValueError('--grid must be a JSON object with "min", "max", "num"')
        else:
            spec = objectives.grid_spec({"name": test_function, "kwargs": _json(kwargs, "--kwargs") or {}}, int(num or 21))
        if name_list:
            spec["names"] = name_list
        node = candidates_from_grid(orm.Dict(dict=spec), metadata=meta)
        how = {"source": "grid", "spec": spec}
        X = node.X
    _label(node, label)
    if name_list and len(name_list) != X.shape[1]:
        raise ValueError(f"{len(name_list)} names for {X.shape[1]} columns")
    logdir.append_jsonl("action", {"action": "candidates", "pk": node.pk, "n": int(X.shape[0]), "caller": caller, **how})
    return {**_info(node, node.creator, label), "num_candidates": int(X.shape[0]), "dim": int(X.shape[1]), **how}


# ---------------------------------------------------------------- observe
def _parse_values(actions, values, file):
    """(actions (M,), t (M, k)) from --actions/--values or a csv file."""
    if file:
        path = os.path.abspath(os.path.expanduser(file))
        data = np.atleast_2d(np.genfromtxt(path, delimiter=",", comments="#"))
        if data.shape[1] < 2:
            raise ValueError("the observations file needs an action column and at least one value column")
        return data[:, 0], data[:, 1:]
    if actions is None or values is None:
        raise ValueError("give --actions and --values, or --file")
    a = np.asarray(_ints(actions, "--actions"))
    v = values if isinstance(values, list) else None
    if v is None:
        text = str(values).strip()
        if text.startswith("["):
            v = _json(text, "--values")
        else:
            try:
                v = [float(x) for x in text.replace(";", ",").split(",") if x.strip() != ""]
            except ValueError as exc:
                raise ValueError("--values must be comma separated numbers or a JSON array") from exc
    t = np.asarray(v, dtype=float)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    if t.shape[0] != a.shape[0]:
        raise ValueError(f"{a.shape[0]} actions but {t.shape[0]} value rows")
    return a, t


def observe(candidates_pk, observations_pk=None, actions=None, values=None, file=None, label=None, caller="cli"):
    from aiida import orm

    from ..calcfunctions import observe as _observe
    from ..data import CandidatesData, ObservationsData

    cand = _node(candidates_pk, CandidatesData, "CandidatesData")
    prev = _node(observations_pk, ObservationsData, "ObservationsData") if observations_pk else None
    a, t = _parse_values(actions, values, file)
    N = cand.num_candidates
    bad = [int(x) for x in a if x < 0 or x >= N]
    if bad:
        raise ValueError(f"actions {bad} are outside [0, {N})")
    if prev is not None:
        dup = sorted(set(int(x) for x in a) & set(int(x) for x in prev.actions))
        if dup:
            raise ValueError(f"actions {dup} are already observed in ObservationsData<{prev.pk}>")
    new = orm.ArrayData()
    new.set_array("actions", np.asarray(a, dtype=np.int64))
    new.set_array("t", t)
    kw = {"candidates": cand, "new": new}
    if prev is not None:
        kw["observations"] = prev
    node = _observe(**kw, metadata={"label": label} if label else {})
    _label(node, label)
    logdir.append_jsonl("action", {"action": "observe", "pk": node.pk, "candidates_pk": cand.pk,
                                   "previous_pk": prev.pk if prev else None, "num_new": int(a.shape[0]), "caller": caller})
    return {**_info(node, node.creator, label), "candidates_pk": cand.pk, "previous_observations_pk": prev.pk if prev else None,
            "num_new": int(a.shape[0]), "num_observations": node.num_observations, "num_objectives": node.num_objectives}


# ---------------------------------------------------------------- propose
def _propose_parameters(score=None, num_rand_basis=None, num_search_each_probe=None, seed=None, num_objectives=None,
                        minimize=False, random=False, posterior=False, interval=None, maximize=None):
    """the parameters Dict of propose from CLI options (only what was given; defaults live in PROPOSE_DEFAULTS)."""
    p = {}
    for key, val in (("score", score), ("num_rand_basis", num_rand_basis), ("num_search_each_probe", num_search_each_probe),
                     ("seed", seed), ("num_objectives", num_objectives), ("interval", interval)):
        if val is not None:
            p[key] = val
    if minimize:
        p["maximize"] = False
    if maximize is not None:
        p["maximize"] = bool(maximize)
    if random:
        p["random"] = True
    if posterior:
        p["posterior"] = True
    assert set(p) <= set(PROPOSE_DEFAULTS)
    return p


def propose(candidates_pk, observations_pk=None, score=None, num_rand_basis=None, num_search_each_probe=None, seed=None,
            num_objectives=None, minimize=False, random=False, posterior=False, interval=None, label=None, caller="cli"):
    from aiida import orm

    from ..calcfunctions import propose as _propose
    from ..data import CandidatesData, ObservationsData

    cand = _node(candidates_pk, CandidatesData, "CandidatesData")
    obs = _node(observations_pk, ObservationsData, "ObservationsData") if observations_pk else None
    params = _propose_parameters(score, num_rand_basis, num_search_each_probe, seed, num_objectives, minimize, random,
                                 posterior, interval)
    kw = {"candidates": cand, "parameters": orm.Dict(dict=params)}
    if obs is not None:
        kw["observations"] = obs
    out = _propose(**kw, metadata={"label": label} if label else {})
    prop, summary = out["proposal"], out["summary"].get_dict()
    _label(prop, label)
    logdir.append_jsonl("action", {"action": "propose", "pk": prop.pk, "candidates_pk": cand.pk,
                                   "observations_pk": obs.pk if obs else None, "mode": summary["mode"],
                                   "actions": summary["proposed_actions"], "caller": caller})
    result = {**_info(prop, prop.creator, label), "actions": prop.get_array("actions").tolist(),
              "X": prop.get_array("X").tolist(), "summary": summary}
    if "posterior" in out:
        result["posterior_pk"] = out["posterior"].pk
    if obs is None:
        result["note"] = "no observations given: random proposals. Record values with `observe`, then propose again with --observations-pk."
    return result


# ---------------------------------------------------------------- submit-optimize
def submit_optimize(test_function, kwargs=None, maximize=False, candidates_pk=None, num=None, observations_pk=None,
                    num_random=None, num_bayes=None, score=None, num_rand_basis=None, num_search_each_probe=None,
                    seed=None, label=None, caller="cli"):
    from aiida import orm
    from aiida.engine import submit
    from aiida.plugins import WorkflowFactory

    from .. import objectives
    from ..calcfunctions import candidates_from_grid, run_propose
    from ..data import CandidatesData, ObservationsData
    from .control import daemon_status

    objective = {"name": test_function, "kwargs": _json(kwargs, "--kwargs") or {}, "maximize": bool(maximize)}
    fn = objectives.make(objective)                                   # raises on an unknown name / bad kwargs
    if candidates_pk:
        cand = _node(candidates_pk, CandidatesData, "CandidatesData")
    else:
        cand = candidates_from_grid(orm.Dict(dict=objectives.grid_spec(objective, int(num or 21))),
                                    metadata={"label": f"{label or test_function}_grid"})
        cand.label = f"{label or test_function}_grid"
    if cand.dim != fn.dim:
        raise ValueError(f"candidates have dimension {cand.dim} but {test_function} has {fn.dim}")
    obs = _node(observations_pk, ObservationsData, "ObservationsData") if observations_pk else None
    params = _propose_parameters(score, num_rand_basis, num_search_each_probe, seed, maximize=bool(maximize))
    params["num_objectives"] = fn.nobj
    # validate the acquisition before submitting (the WorkChain would only fail minutes later in the daemon)
    run_propose(cand.X[:3], np.zeros(0, dtype=np.int64), np.zeros((0, fn.nobj)), dict(params, random=True))
    if score and (score.upper() not in (("TS", "EI", "PI") if fn.nobj == 1 else ("TS", "EHVI", "HVPI"))):
        raise ValueError(f"score {score!r} is not available for {fn.nobj} objective(s)")
    n_random, n_bayes = int(num_random if num_random is not None else 10), int(num_bayes if num_bayes is not None else 20)
    if n_random <= 0 and obs is None:
        raise ValueError("--num-random 0 needs --observations-pk to start from")
    if n_random + n_bayes * int(params.get("num_search_each_probe", 1)) > cand.num_candidates:
        raise ValueError(f"{n_random} + {n_bayes} evaluations exceed the {cand.num_candidates} candidates")
    ds = daemon_status()
    inputs = {"candidates": cand, "objective": orm.Dict(dict=objective), "parameters": orm.Dict(dict=params),
              "num_random": orm.Int(n_random), "num_bayes": orm.Int(n_bayes), "label": orm.Str(label or test_function),
              "metadata": {"label": label or test_function}}
    if obs is not None:
        inputs["observations"] = obs
    node = submit(WorkflowFactory("physbo.optimize"), **inputs)
    logdir.append_jsonl("action", {"action": "submit-optimize", "pk": node.pk, "objective": objective,
                                   "candidates_pk": cand.pk, "num_random": n_random, "num_bayes": n_bayes, "caller": caller})
    out = {"pk": node.pk, "label": node.label, "candidates_pk": cand.pk, "objective": objective, "parameters": params,
           "num_random": n_random, "num_bayes": n_bayes, "daemon": ds}
    if not ds.get("running"):
        out["hint"] = "the daemon is not running: the WorkChain stays in Created until `daemon-start`"
    return out
