# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""calcfunctions: every step of a PHYSBO campaign is a node in the provenance graph.

    candidates_from_file(source, options)        SinglefileData (csv/npy/npz) -> CandidatesData (discrete space)
    candidates_from_grid(spec)                   Dict {min, max, num} -> CandidatesData (physbo make_grid)
    search_box(spec)                             Dict {min, max} -> SearchBoxData (range / continuous space)
    observe(space, new, observations=None)       append (actions | X, t) -> new ObservationsData
    propose(space, parameters, observations=None)
                                                 PHYSBO policy on the observations -> proposal (actions, X | X),
                                                 summary Dict, optional posterior (discrete: fmean/fstd on every candidate)
    evaluate_test_function(space, proposal, objective)
                                                 PHYSBO test function on the proposed rows -> ArrayData (actions?, X, t)
    summarize(space, observations, settings)     best / Pareto front / best-so-far sequence -> Dict

`space` is a CandidatesData (discrete policies, proposals are row indices = actions) or a SearchBoxData
(range policies, proposals are coordinates). `propose` is stateless: it rebuilds the policy from
(space, observations) every call (`initial_data`), learns the hyperparameters, and returns the next
points without a simulator (PHYSBO's interactive mode). The whole state of a campaign is therefore the
ObservationsData chain.
"""
import io
import os
import tempfile
import time

import numpy as np
from aiida import orm
from aiida.engine import calcfunction

from .data import DISCRETE, RANGE, CandidatesData, ObservationsData, SearchBoxData, normalize_observations, space_of

SINGLE_SCORES = ("TS", "EI", "PI")
MULTI_SCORES = ("TS", "EHVI", "HVPI")
OPTIMIZERS = ("random", "odatse")
# "mapper" is left out: with ODAT-SE 4 its ColorMap.txt starts with a header line that PHYSBO 3.2.1 does not skip
ODATSE_ALGORITHMS = ("exchange", "pamc", "minsearch", "bayes")

# the keys of the `parameters` Dict of propose and their defaults (the single source for cli/steps.py)
PROPOSE_DEFAULTS = {
    "score": "TS",                 # acquisition: TS/EI/PI (single), TS/EHVI/HVPI (multi)
    "num_rand_basis": 0,           # 0: exact Gaussian process; >0: random-feature Bayesian linear model
    "num_search_each_probe": 1,    # how many points to propose
    "num_objectives": None,        # default: from the observations (1 when there are none)
    "seed": None,                  # numpy seed (None: not seeded)
    "maximize": True,              # False: the stored t is minimized (propose sees -t)
    "random": False,               # True: random proposals (also used automatically when nothing is observed yet)
    "interval": 0,                 # hyperparameter learning interval passed to bayes_search (0: learn once)
    "posterior": False,            # discrete: also return fmean / fstd on every candidate (costly for large N)
    # range space only: how the acquisition function is maximized over the box
    "optimizer": "random",         # random (uniform samples) | odatse (ODAT-SE algorithm)
    "optimizer_nsamples": 1000,    # random: number of uniform samples
    "odatse_algorithm": "exchange",  # odatse: exchange | pamc | minsearch | bayes
    "odatse_params": None,         # odatse: dict merged over physbo's default_alg_dict (e.g. {"exchange": {"numsteps": 200}})
}
RANGE_ONLY = ("optimizer", "optimizer_nsamples", "odatse_algorithm", "odatse_params")


def _nondominated_mask(t: np.ndarray) -> np.ndarray:
    """rows of t (M, k) that are not dominated (maximization of every column)."""
    M = t.shape[0]
    mask = np.ones(M, dtype=bool)
    for i in range(M):
        if not mask[i]:
            continue
        ge = np.all(t >= t[i], axis=1)
        gt = np.any(t > t[i], axis=1)
        if np.any(ge & gt):
            mask[i] = False
    return mask


def best_of(t: np.ndarray, maximize: bool = True, actions=None, X=None, max_front: int = 200) -> dict:
    """best observation (single objective) or Pareto front (multi) of raw values t (M, k), with the action / X when given."""
    if t.shape[0] == 0:
        return {"num_observations": 0}
    sign = 1.0 if maximize else -1.0
    d = {"num_observations": int(t.shape[0])}
    if t.shape[1] == 1:
        i = int(np.argmax(sign * t[:, 0]))
        d.update({"best_index": i, "best_value": float(t[i, 0]), "best_step": i + 1})
        if actions is not None:
            d["best_action"] = int(actions[i])
        if X is not None:
            d["best_X"] = np.asarray(X)[i].tolist()
        return d
    idx = np.nonzero(_nondominated_mask(sign * t))[0]
    d.update({"pareto_size": int(idx.size), "pareto_indices": [int(i) for i in idx[:max_front]],
              "pareto_values": t[idx][:max_front].tolist(), "pareto_truncated": bool(idx.size > max_front)})
    if actions is not None:
        d["pareto_actions"] = [int(a) for a in np.asarray(actions)[idx][:max_front]]
    if X is not None:
        d["pareto_X"] = np.asarray(X)[idx][:max_front].tolist()
    return d


def best_sequence(t: np.ndarray, maximize: bool = True) -> list:
    """best value after each observation (single objective)."""
    if t.shape[0] == 0 or t.shape[1] != 1:
        return []
    acc = np.maximum.accumulate(t[:, 0]) if maximize else np.minimum.accumulate(t[:, 0])
    return [float(v) for v in acc]


# ------------------------------------------------------------------ search spaces
def parse_candidates(content: bytes, filename: str, options: dict) -> np.ndarray:
    """X from csv / npy / npz content (format from options["format"] or the file name)."""
    fmt = (options.get("format") or filename.rsplit(".", 1)[-1]).lower()
    if fmt == "npy":
        X = np.load(io.BytesIO(content), allow_pickle=False)
    elif fmt == "npz":
        with np.load(io.BytesIO(content), allow_pickle=False) as z:
            key = options.get("key") or ("X" if "X" in z.files else z.files[0])
            X = z[key]
    elif fmt in ("csv", "txt", "tsv", "dat"):
        delimiter = options.get("delimiter") or ("\t" if fmt == "tsv" else ",")
        X = np.genfromtxt(io.StringIO(content.decode("utf-8")), delimiter=delimiter,
                          skip_header=int(options.get("skip_header") or 0), comments="#")
    else:
        raise ValueError(f"unknown candidates format {fmt!r} (csv, tsv, npy, npz)")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cols = options.get("columns")
    if cols:
        X = X[:, [int(c) for c in cols]]
    if X.shape[0] == 0:
        raise ValueError("no candidates in the file")
    return X


@calcfunction
def candidates_from_file(source: orm.SinglefileData, options: orm.Dict) -> CandidatesData:
    """CandidatesData from a csv / npy / npz file. options: format, delimiter, skip_header, columns (indices), key, names."""
    opts = options.get_dict()
    with source.open(mode="rb") as f:
        content = f.read()
    X = parse_candidates(content, source.filename, opts)
    return CandidatesData(X=X, columns=opts.get("names"))


@calcfunction
def candidates_from_grid(spec: orm.Dict) -> CandidatesData:
    """CandidatesData on a regular grid: spec = {min: [..], max: [..], num: int | [..], names: [..]}."""
    from physbo.search.utility import make_grid

    d = spec.get_dict()
    for key in ("min", "max", "num"):
        if key not in d:
            raise ValueError(f"grid spec needs {key!r}")
    X = make_grid(d["min"], d["max"], d["num"])
    return CandidatesData(X=X, columns=d.get("names"))


@calcfunction
def search_box(spec: orm.Dict) -> SearchBoxData:
    """SearchBoxData (continuous space): spec = {min: [..], max: [..], names: [..]}."""
    d = spec.get_dict()
    for key in ("min", "max"):
        if key not in d:
            raise ValueError(f"search box spec needs {key!r}")
    return SearchBoxData(min_X=d["min"], max_X=d["max"], columns=d.get("names"))


# ------------------------------------------------------------------ observe
def _arrays(node, *names):
    have = set(node.get_arraynames())
    return [node.get_array(n) if n in have else None for n in names]


@calcfunction
def observe(space: orm.ArrayData, new: orm.ArrayData, observations: ObservationsData = None) -> ObservationsData:
    """Append new observations to `observations` (or start a chain).

    `new` carries `t` and, for a CandidatesData space, `actions` (row indices; X is filled from the candidates,
    duplicates are rejected) or, for a SearchBoxData space, `X` (coordinates; must have the box dimension)."""
    kind = space_of(space)
    actions, X, t = _arrays(new, "actions", "X", "t")
    if t is None:
        raise ValueError("new observations need the array t")
    if kind == DISCRETE:
        if actions is None:
            raise ValueError("a discrete space needs the array actions (row indices of the candidates)")
        actions, _, t = normalize_observations(actions, None, t)
        N = space.num_candidates
        if np.any(actions < 0) or np.any(actions >= N):
            raise ValueError(f"actions must be in [0, {N})")
        X = space.X[actions]
    else:
        if X is None:
            raise ValueError("a range space needs the array X (coordinates)")
        actions, X, t = normalize_observations(None, X, t)
        if X.shape[1] != space.dim:
            raise ValueError(f"X has dimension {X.shape[1]} but the search box has {space.dim}")
    if t.shape[0] == 0:
        raise ValueError("no new observations")
    if observations is not None:
        if observations.space != kind:
            raise ValueError(f"the observations are of a {observations.space} space, the given space is {kind}")
        prev_t = observations.t
        if prev_t.shape[1] != t.shape[1]:
            raise ValueError(f"number of objectives changed: {prev_t.shape[1]} -> {t.shape[1]}")
        prev_a, prev_X = observations.actions, observations.X
        if prev_X is None and kind == DISCRETE:          # 0.1.0 node without X
            prev_X = space.X[prev_a]
    else:
        prev_t = np.zeros((0, t.shape[1]))
        prev_a = np.zeros(0, dtype=np.int64) if kind == DISCRETE else None
        prev_X = np.zeros((0, X.shape[1]))
    all_X = np.concatenate([prev_X, X])
    all_t = np.concatenate([prev_t, t])
    if kind == DISCRETE:
        all_a = np.concatenate([prev_a, actions])
        uniq, counts = np.unique(all_a, return_counts=True)
        dup = uniq[counts > 1]
        if dup.size:
            raise ValueError(f"actions observed more than once: {dup.tolist()}")
        return ObservationsData(actions=all_a, X=all_X, t=all_t)
    return ObservationsData(X=all_X, t=all_t)


# ------------------------------------------------------------------ propose
def _check_params(params: dict, kind: str) -> dict:
    p = dict(PROPOSE_DEFAULTS, **params)
    unknown = set(params) - set(PROPOSE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown propose parameters {sorted(unknown)}; known {sorted(PROPOSE_DEFAULTS)}")
    if kind == DISCRETE:
        given = [k for k in RANGE_ONLY if k in params and params[k] != PROPOSE_DEFAULTS[k]]
        if given:
            raise ValueError(f"{given} apply to a range (SearchBoxData) space only")
    if p["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"optimizer {p['optimizer']!r} unknown; choose from {OPTIMIZERS}")
    if p["odatse_algorithm"] not in ODATSE_ALGORITHMS:
        raise ValueError(f"odatse_algorithm {p['odatse_algorithm']!r} unknown; choose from {ODATSE_ALGORITHMS}")
    if int(p["num_search_each_probe"]) < 1:
        raise ValueError("num_search_each_probe must be >= 1")
    if int(p["optimizer_nsamples"]) < 1:
        raise ValueError("optimizer_nsamples must be >= 1")
    return p


def _deep_update(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _make_optimizer(p: dict, min_X, max_X):
    """the acquisition optimizer of a range policy; (optimizer, description dict)."""
    if p["optimizer"] == "random":
        from physbo.search.optimize.random import Optimizer

        n = int(p["optimizer_nsamples"])
        return Optimizer(min_X=min_X, max_X=max_X, nsamples=n), {"optimizer": "random", "nsamples": n}
    from physbo.search.optimize import odatse as od

    alg = od.default_alg_dict(min_X, max_X, p["odatse_algorithm"])
    alg = _deep_update(alg, p["odatse_params"] or {})
    desc = {"optimizer": "odatse", "algorithm": p["odatse_algorithm"],
            "params": {k: v for k, v in alg.items() if k not in ("param", "name")}}
    return od.Optimizer(alg), desc


class _in_tempdir:
    """ODAT-SE writes odatse_output/ into the working directory; run it in a scratch directory."""

    def __enter__(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="physbo_odatse_")
        os.chdir(self.tmp)
        return self.tmp

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        return False


def _policy(space, kind, k, obs_actions, obs_X, t_fit):
    import physbo

    have = t_fit.shape[0] > 0
    if kind == DISCRETE:
        initial = (obs_actions, t_fit if k > 1 else t_fit[:, 0]) if have else None
        if k == 1:
            return physbo.search.discrete.Policy(test_X=space.X, initial_data=initial)
        return physbo.search.discrete_multi.Policy(test_X=space.X, num_objectives=k, initial_data=initial)
    initial = (obs_X, t_fit if k > 1 else t_fit[:, 0]) if have else None
    if k == 1:
        return physbo.search.range.Policy(min_X=space.min_X, max_X=space.max_X, initial_data=initial)
    return physbo.search.range_multi.Policy(num_objectives=k, min_X=space.min_X, max_X=space.max_X, initial_data=initial)


def run_propose(space, obs_actions, obs_X, t: np.ndarray, params: dict):
    """the PHYSBO call itself: returns (proposed actions | None, proposed X, summary dict, posterior dict | None)."""
    import physbo

    kind = space_of(space)
    p = _check_params(params, kind)
    k = int(p["num_objectives"] or (t.shape[1] if t.size else 1))
    if t.size and t.shape[1] != k:
        raise ValueError(f"num_objectives={k} but the observations have {t.shape[1]} columns")
    scores = SINGLE_SCORES if k == 1 else MULTI_SCORES
    score = str(p["score"]).upper()
    if score not in scores:
        raise ValueError(f"score {score!r} is not available for {k} objective(s); choose from {scores}")
    n = int(p["num_search_each_probe"])
    maximize = bool(p["maximize"])
    sign = 1.0 if maximize else -1.0
    M = t.shape[0]
    mode = "random" if (p["random"] or M == 0) else "bayes"
    summary = {"space": kind, "dim": int(space.dim), "num_objectives": k, "num_observed": int(M), "maximize": maximize,
               "score": score if mode == "bayes" else None, "num_rand_basis": int(p["num_rand_basis"]),
               "seed": p["seed"], "physbo_version": physbo.__version__}
    if kind == DISCRETE:
        remaining = space.num_candidates - M
        if remaining <= 0:
            raise ValueError("every candidate has been observed already")
        n = min(n, remaining)
        summary.update({"num_candidates": int(space.num_candidates), "num_remaining_after": int(remaining - n)})
    elif n > 1 and int(p["num_rand_basis"]) > 0 and mode == "bayes":
        raise ValueError("PHYSBO 3.2.1 range policies cannot propose several points per step with num_rand_basis > 0 "
                         "(Variable.add shape error); use num_search_each_probe=1 or num_rand_basis=0")
    summary["num_search_each_probe"] = n
    summary["mode"] = mode

    t0 = time.time()
    policy = _policy(space, kind, k, obs_actions, obs_X, sign * t)
    if p["seed"] is not None:
        policy.set_seed(int(p["seed"]))
    posterior = None
    if mode == "random":
        out = policy.random_search(max_num_probes=1, num_search_each_probe=n, simulator=None, is_disp=False)
    elif kind == DISCRETE:
        out = policy.bayes_search(max_num_probes=1, num_search_each_probe=n, simulator=None, score=score,
                                  interval=int(p["interval"]), num_rand_basis=int(p["num_rand_basis"]), is_disp=False)
    else:
        optimizer, desc = _make_optimizer(p, space.min_X, space.max_X)
        summary.update(desc)
        with _in_tempdir():
            out = policy.bayes_search(max_num_probes=1, num_search_each_probe=n, simulator=None, score=score,
                                      interval=int(p["interval"]), num_rand_basis=int(p["num_rand_basis"]),
                                      optimizer=optimizer, is_disp=False)
    if kind == DISCRETE:
        proposed_a = np.asarray(out, dtype=np.int64).reshape(-1)
        proposed_X = space.X[proposed_a]
    else:
        proposed_a = None
        proposed_X = np.asarray(out, dtype=float).reshape(-1, space.dim)
    if mode == "bayes":
        fm = np.asarray(policy.get_post_fmean(proposed_X), dtype=float).reshape(proposed_X.shape[0], -1)
        fv = np.asarray(policy.get_post_fcov(proposed_X, diag=True), dtype=float).reshape(proposed_X.shape[0], -1)
        summary["posterior_at_proposal"] = {"fmean": (sign * fm).tolist(), "fstd": np.sqrt(np.clip(fv, 0, None)).tolist()}
        if p["posterior"] and kind == DISCRETE:
            N = space.num_candidates
            fmean = np.asarray(policy.get_post_fmean(space.X), dtype=float).reshape(N, -1)
            fvar = np.asarray(policy.get_post_fcov(space.X, diag=True), dtype=float).reshape(N, -1)
            posterior = {"fmean": sign * fmean, "fstd": np.sqrt(np.clip(fvar, 0.0, None))}
    summary.update({"proposed_actions": proposed_a.tolist() if proposed_a is not None else None,
                    "proposed_X": proposed_X.tolist(),
                    "best_so_far": best_of(t, maximize, actions=obs_actions, X=obs_X),
                    "elapsed_seconds": round(time.time() - t0, 3)})
    return proposed_a, proposed_X, summary, posterior


def _obs_arrays(space, observations):
    """(actions | None, X, t) of an observations node (X filled from the candidates for 0.1.0 nodes)."""
    if observations is None:
        return (np.zeros(0, dtype=np.int64) if space_of(space) == DISCRETE else None), np.zeros((0, space.dim)), np.zeros((0, 1))
    actions, X, t = observations.actions, observations.X, observations.t
    if X is None and actions is not None:
        X = space.X[actions]
    return actions, X, t


@calcfunction
def propose(space: orm.ArrayData, parameters: orm.Dict, observations: ObservationsData = None) -> dict:
    """Next points to evaluate. parameters: see PROPOSE_DEFAULTS. Outputs: proposal (actions?, X), summary, [posterior]."""
    actions, X, t = _obs_arrays(space, observations)
    proposed_a, proposed_X, summary, posterior = run_propose(space, actions, X, t, parameters.get_dict())
    proposal = orm.ArrayData()
    if proposed_a is not None:
        proposal.set_array("actions", proposed_a)
    proposal.set_array("X", proposed_X)
    out = {"proposal": proposal, "summary": orm.Dict(dict=summary)}
    if posterior is not None:
        post = orm.ArrayData()
        post.set_array("fmean", posterior["fmean"])
        post.set_array("fstd", posterior["fstd"])
        out["posterior"] = post
    return out


# ------------------------------------------------------------------ test functions
@calcfunction
def evaluate_test_function(space: orm.ArrayData, proposal: orm.ArrayData, objective: orm.Dict) -> orm.ArrayData:
    """Evaluate a PHYSBO test function ({name, kwargs, maximize}) on the proposed points -> ArrayData (actions?, X, t)."""
    from . import objectives

    space_of(space)
    actions, X = _arrays(proposal, "actions", "X")
    if X is None:
        X = space.X[np.asarray(actions, dtype=np.int64)]
    t = objectives.evaluate(objective.get_dict(), X)
    out = orm.ArrayData()
    if actions is not None:
        out.set_array("actions", np.asarray(actions, dtype=np.int64).reshape(-1))
    out.set_array("X", np.asarray(X, dtype=float))
    out.set_array("t", t)
    return out


# ------------------------------------------------------------------ summary
@calcfunction
def summarize(space: orm.ArrayData, observations: ObservationsData, settings: orm.Dict) -> orm.Dict:
    """Best value (or Pareto front), best X and best-so-far sequence of an observations node. settings: {maximize, ...extra}."""
    s = settings.get_dict()
    maximize = bool(s.get("maximize", True))
    actions, X, t = _obs_arrays(space, observations)
    d = {**s, "space": space_of(space), "observations_pk": observations.pk, "num_observations": observations.num_observations,
         "num_objectives": observations.num_objectives, "best": best_of(t, maximize, actions=actions, X=X),
         "best_sequence": best_sequence(t, maximize)}
    if isinstance(space, CandidatesData):
        d["num_candidates"] = space.num_candidates
    return orm.Dict(dict=d)
