# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Node-creating implementations of the CLI (candidates, search-box, observe, propose, submit-optimize).

Validation happens here, before anything is stored or submitted. Every action is logged.
"""
import json
import os

import numpy as np

from .. import logdir
from ..calcfunctions import PROPOSE_DEFAULTS, RANGE_ONLY


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


def _space(pk):
    from ..data import CandidatesData, SearchBoxData

    return _node(pk, (CandidatesData, SearchBoxData), "CandidatesData or SearchBoxData")


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


def _floats(text, what):
    try:
        return [float(x) for x in str(text).split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"{what} must be comma separated numbers") from exc


def _rows(text, what, column=False):
    """rows of numbers from JSON [[...], ...] or 'a,b;c,d' (';' separates rows, ',' entries of a row).

    column=True (objective values): a plain 'a,b,c' is M rows of one value each, not one row of M values;
    column=False (coordinates): a plain 'a,b' is one point with d coordinates."""
    if isinstance(text, list):
        v = text
    else:
        s = str(text).strip()
        if s.startswith("["):
            v = _json(s, what)
        elif ";" in s:
            v = [_floats(r, what) for r in s.split(";") if r.strip()]
        else:
            v = _floats(s, what)
            if column:
                v = [[x] for x in v]
    a = np.asarray(v, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1) if column else a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"{what} must be a list of numbers or of rows")
    return a


def _label(node, label):
    if label:
        node.label = label
    return node


def _info(node, process, label):
    return {"pk": node.pk, "process_pk": process.pk, "label": label or node.label}


# ---------------------------------------------------------------- search spaces
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
        if name_list and len(name_list) != X.shape[1]:
            raise ValueError(f"{len(name_list)} names for {X.shape[1]} columns")
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
            if len(name_list) != len(spec["min"]):
                raise ValueError(f"{len(name_list)} names for {len(spec['min'])} dimensions")
            spec["names"] = name_list
        node = candidates_from_grid(orm.Dict(dict=spec), metadata=meta)
        how = {"source": "grid", "spec": spec}
        X = node.X
    _label(node, label)
    logdir.append_jsonl("action", {"action": "candidates", "pk": node.pk, "n": int(X.shape[0]), "caller": caller, **how})
    return {**_info(node, node.creator, label), "space": "discrete", "num_candidates": int(X.shape[0]), "dim": int(X.shape[1]), **how}


def search_box(min=None, max=None, test_function=None, kwargs=None, names=None, label=None, caller="cli"):
    from aiida import orm

    from .. import objectives
    from ..calcfunctions import search_box as _search_box

    if test_function:
        if min or max:
            raise ValueError("give --min/--max or --test-function, not both")
        spec = objectives.grid_spec({"name": test_function, "kwargs": _json(kwargs, "--kwargs") or {}}, 1)
        spec.pop("num", None)
    else:
        if not (min and max):
            raise ValueError("give --min and --max (comma separated), or --test-function")
        spec = {"min": _floats(min, "--min"), "max": _floats(max, "--max")}
        if len(spec["min"]) != len(spec["max"]):
            raise ValueError("--min and --max must have the same number of entries")
        if not all(lo < hi for lo, hi in zip(spec["min"], spec["max"])):
            raise ValueError("every --min entry must be smaller than the --max entry")
    if names:
        name_list = [n.strip() for n in names.split(",")]
        if len(name_list) != len(spec["min"]):
            raise ValueError(f"{len(name_list)} names for {len(spec['min'])} dimensions")
        spec["names"] = name_list
    node = _search_box(orm.Dict(dict=spec), metadata={"label": label} if label else {})
    _label(node, label)
    logdir.append_jsonl("action", {"action": "search-box", "pk": node.pk, "spec": spec, "caller": caller})
    return {**_info(node, node.creator, label), "space": "range", "dim": node.dim, "min": node.min_X.tolist(),
            "max": node.max_X.tolist()}


# ---------------------------------------------------------------- observe
def _parse_values(values):
    if values is None:
        raise ValueError("give --values")
    return _rows(values, "--values", column=True)


def _new_observations(space, actions, x, values, file):
    """ArrayData (actions | X, t) for `observe`, validated against the space."""
    from aiida import orm

    from ..data import DISCRETE, space_of

    kind = space_of(space)
    if file:
        path = os.path.abspath(os.path.expanduser(file))
        data = np.atleast_2d(np.genfromtxt(path, delimiter=",", comments="#"))
        if kind == DISCRETE:
            if data.shape[1] < 2:
                raise ValueError("the observations file needs an action column and at least one value column")
            a, X, t = data[:, 0], None, data[:, 1:]
        else:
            if data.shape[1] < space.dim + 1:
                raise ValueError(f"the observations file needs {space.dim} coordinate columns and at least one value column")
            a, X, t = None, data[:, :space.dim], data[:, space.dim:]
    else:
        t = _parse_values(values)
        if kind == DISCRETE:
            if actions is None:
                raise ValueError("a discrete space needs --actions (candidate indices)")
            if x is not None:
                raise ValueError("--x is for a range space; a discrete space takes --actions")
            a, X = np.asarray(_ints(actions, "--actions")), None
        else:
            if x is None:
                raise ValueError("a range space needs --x (coordinates)")
            if actions is not None:
                raise ValueError("--actions is for a discrete space; a range space takes --x")
            a, X = None, _rows(x, "--x")
    M = t.shape[0]
    if kind == DISCRETE:
        if a.shape[0] != M:
            raise ValueError(f"{a.shape[0]} actions but {M} value rows")
        N = space.num_candidates
        bad = [int(v) for v in a if v < 0 or v >= N]
        if bad:
            raise ValueError(f"actions {bad} are outside [0, {N})")
    else:
        if X.shape[0] != M:
            raise ValueError(f"{X.shape[0]} coordinate rows but {M} value rows")
        if X.shape[1] != space.dim:
            raise ValueError(f"coordinates have {X.shape[1]} entries but the box has dimension {space.dim}")
    new = orm.ArrayData()
    if a is not None:
        new.set_array("actions", np.asarray(a, dtype=np.int64))
    if X is not None:
        new.set_array("X", np.asarray(X, dtype=float))
    new.set_array("t", t)
    return new


def observe(space_pk, observations_pk=None, actions=None, x=None, values=None, file=None, label=None, caller="cli"):
    from ..calcfunctions import observe as _observe
    from ..data import DISCRETE, ObservationsData, space_of

    space = _space(space_pk)
    prev = _node(observations_pk, ObservationsData, "ObservationsData") if observations_pk else None
    new = _new_observations(space, actions, x, values, file)
    if prev is not None and space_of(space) == DISCRETE and prev.actions is not None:
        dup = sorted(set(int(v) for v in new.get_array("actions")) & set(int(v) for v in prev.actions))
        if dup:
            raise ValueError(f"actions {dup} are already observed in ObservationsData<{prev.pk}>")
    kw = {"space": space, "new": new}
    if prev is not None:
        kw["observations"] = prev
    node = _observe(**kw, metadata={"label": label} if label else {})
    _label(node, label)
    n_new = int(new.get_array("t").shape[0])
    logdir.append_jsonl("action", {"action": "observe", "pk": node.pk, "space_pk": space.pk,
                                   "previous_pk": prev.pk if prev else None, "num_new": n_new, "caller": caller})
    return {**_info(node, node.creator, label), "space": node.space, "space_pk": space.pk,
            "previous_observations_pk": prev.pk if prev else None, "num_new": n_new,
            "num_observations": node.num_observations, "num_objectives": node.num_objectives}


# ---------------------------------------------------------------- propose
def _propose_parameters(score=None, num_rand_basis=None, num_search_each_probe=None, seed=None, num_objectives=None,
                        minimize=False, random=False, posterior=False, interval=None, maximize=None, optimizer=None,
                        optimizer_nsamples=None, odatse_algorithm=None, odatse_params=None):
    """the parameters Dict of propose from CLI options (only what was given; defaults live in PROPOSE_DEFAULTS)."""
    p = {}
    for key, val in (("score", score), ("num_rand_basis", num_rand_basis), ("num_search_each_probe", num_search_each_probe),
                     ("seed", seed), ("num_objectives", num_objectives), ("interval", interval), ("optimizer", optimizer),
                     ("optimizer_nsamples", optimizer_nsamples), ("odatse_algorithm", odatse_algorithm)):
        if val is not None:
            p[key] = val
    if odatse_params is not None:
        d = _json(odatse_params, "--odatse-params")
        if not isinstance(d, dict):
            raise ValueError("--odatse-params must be a JSON object")
        p["odatse_params"] = d
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


def _check_range_opts(space, params):
    from ..data import DISCRETE, space_of

    if space_of(space) == DISCRETE:
        given = [k for k in RANGE_ONLY if k in params]
        if given:
            raise ValueError(f"{given} apply to a range space (SearchBoxData); Node<{space.pk}> is a CandidatesData")


def propose(space_pk, observations_pk=None, score=None, num_rand_basis=None, num_search_each_probe=None, seed=None,
            num_objectives=None, minimize=False, random=False, posterior=False, interval=None, optimizer=None,
            optimizer_nsamples=None, odatse_algorithm=None, odatse_params=None, label=None, caller="cli"):
    from aiida import orm

    from ..calcfunctions import propose as _propose
    from ..data import ObservationsData

    space = _space(space_pk)
    obs = _node(observations_pk, ObservationsData, "ObservationsData") if observations_pk else None
    params = _propose_parameters(score, num_rand_basis, num_search_each_probe, seed, num_objectives, minimize, random,
                                 posterior, interval, optimizer=optimizer, optimizer_nsamples=optimizer_nsamples,
                                 odatse_algorithm=odatse_algorithm, odatse_params=odatse_params)
    _check_range_opts(space, params)
    kw = {"space": space, "parameters": orm.Dict(dict=params)}
    if obs is not None:
        kw["observations"] = obs
    out = _propose(**kw, metadata={"label": label} if label else {})
    prop, summary = out["proposal"], out["summary"].get_dict()
    _label(prop, label)
    logdir.append_jsonl("action", {"action": "propose", "pk": prop.pk, "space_pk": space.pk,
                                   "observations_pk": obs.pk if obs else None, "mode": summary["mode"],
                                   "actions": summary["proposed_actions"], "X": summary["proposed_X"], "caller": caller})
    result = {**_info(prop, prop.creator, label), "space": summary["space"], "actions": summary["proposed_actions"],
              "X": summary["proposed_X"], "summary": summary}
    if "posterior" in out:
        result["posterior_pk"] = out["posterior"].pk
    if obs is None:
        result["note"] = "no observations given: random proposals. Record values with `observe`, then propose again with --observations-pk."
    return result


# ---------------------------------------------------------------- submit-optimize
def submit_optimize(test_function, kwargs=None, maximize=False, space_pk=None, space=None, num=None, observations_pk=None,
                    num_random=None, num_bayes=None, score=None, num_rand_basis=None, num_search_each_probe=None,
                    seed=None, optimizer=None, optimizer_nsamples=None, odatse_algorithm=None, odatse_params=None,
                    label=None, caller="cli"):
    from aiida import orm
    from aiida.engine import submit
    from aiida.plugins import WorkflowFactory

    from .. import objectives
    from ..calcfunctions import candidates_from_grid, run_propose
    from ..calcfunctions import search_box as _search_box
    from ..data import DISCRETE, RANGE, CandidatesData, ObservationsData, space_of
    from .control import daemon_status

    objective = {"name": test_function, "kwargs": _json(kwargs, "--kwargs") or {}, "maximize": bool(maximize)}
    fn = objectives.make(objective)                                   # raises on an unknown name / bad kwargs
    name = label or test_function
    if space_pk:
        if space:
            raise ValueError("give --space-pk or --space, not both")
        sp = _space(space_pk)
    else:
        kind = (space or DISCRETE).lower()
        if kind == DISCRETE:
            sp = candidates_from_grid(orm.Dict(dict=objectives.grid_spec(objective, int(num or 21))),
                                      metadata={"label": f"{name}_grid"})
        elif kind == RANGE:
            spec = objectives.grid_spec(objective, 1)
            spec.pop("num")
            sp = _search_box(orm.Dict(dict=spec), metadata={"label": f"{name}_box"})
        else:
            raise ValueError("--space must be discrete or range")
        sp.label = f"{name}_{'grid' if kind == DISCRETE else 'box'}"
    if sp.dim != fn.dim:
        raise ValueError(f"the space has dimension {sp.dim} but {test_function} has {fn.dim}")
    obs = _node(observations_pk, ObservationsData, "ObservationsData") if observations_pk else None
    if obs is not None and obs.space != space_of(sp):
        raise ValueError(f"ObservationsData<{obs.pk}> is of a {obs.space} space but the given space is {space_of(sp)}")
    params = _propose_parameters(score, num_rand_basis, num_search_each_probe, seed, maximize=bool(maximize),
                                 optimizer=optimizer, optimizer_nsamples=optimizer_nsamples,
                                 odatse_algorithm=odatse_algorithm, odatse_params=odatse_params)
    _check_range_opts(sp, params)
    params["num_objectives"] = fn.nobj
    # validate the parameters before submitting (the WorkChain would only fail minutes later in the daemon)
    probe = dict(params, random=True)
    if space_of(sp) == DISCRETE:
        run_propose(sp, np.zeros(0, dtype=np.int64), np.zeros((0, sp.dim)), np.zeros((0, fn.nobj)), probe)
    else:
        run_propose(sp, None, np.zeros((0, sp.dim)), np.zeros((0, fn.nobj)), probe)
    if score and (score.upper() not in (("TS", "EI", "PI") if fn.nobj == 1 else ("TS", "EHVI", "HVPI"))):
        raise ValueError(f"score {score!r} is not available for {fn.nobj} objective(s)")
    n_random = int(num_random if num_random is not None else 10)
    n_bayes = int(num_bayes if num_bayes is not None else 20)
    per_step = int(params.get("num_search_each_probe", 1))
    if n_random <= 0 and obs is None:
        raise ValueError("--num-random 0 needs --observations-pk to start from")
    if isinstance(sp, CandidatesData) and n_random + n_bayes * per_step > sp.num_candidates:
        raise ValueError(f"{n_random} + {n_bayes}x{per_step} evaluations exceed the {sp.num_candidates} candidates")
    if space_of(sp) == RANGE and per_step > 1 and int(params.get("num_rand_basis", 0)) > 0:
        raise ValueError("a range space cannot propose several points per step with num_rand_basis > 0 (PHYSBO 3.2.1)")
    ds = daemon_status()
    inputs = {"space": sp, "objective": orm.Dict(dict=objective), "parameters": orm.Dict(dict=params),
              "num_random": orm.Int(n_random), "num_bayes": orm.Int(n_bayes), "label": orm.Str(name),
              "metadata": {"label": name}}
    if obs is not None:
        inputs["observations"] = obs
    node = submit(WorkflowFactory("physbo.optimize"), **inputs)
    logdir.append_jsonl("action", {"action": "submit-optimize", "pk": node.pk, "objective": objective, "space_pk": sp.pk,
                                   "space": space_of(sp), "num_random": n_random, "num_bayes": n_bayes, "caller": caller})
    out = {"pk": node.pk, "label": node.label, "space_pk": sp.pk, "space": space_of(sp), "objective": objective,
           "parameters": params, "num_random": n_random, "num_bayes": n_bayes, "daemon": ds}
    if not ds.get("running"):
        out["hint"] = "the daemon is not running: the WorkChain stays in Created until `daemon-start`"
    return out
