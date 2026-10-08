# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""calcfunctions: every step of a PHYSBO campaign is a node in the provenance graph.

    candidates_from_file(source, options)        SinglefileData (csv/npy/npz) -> CandidatesData
    candidates_from_grid(spec)                   Dict {min, max, num} -> CandidatesData (physbo make_grid)
    observe(candidates, new, observations=None)  append (actions, t) -> new ObservationsData
    propose(candidates, parameters, observations=None)
                                                 PHYSBO policy on the observations -> proposal (actions, X),
                                                 summary Dict, optional posterior (fmean, fstd on every candidate)
    evaluate_test_function(candidates, proposal, objective)
                                                 PHYSBO test function on the proposed rows -> ArrayData (actions, t)

`propose` is stateless: it rebuilds the policy from (candidates, observations) every call
(`initial_data`), learns the hyperparameters, and returns the next actions without a simulator
(PHYSBO's interactive mode). The whole state of a campaign is therefore the ObservationsData chain.
"""
import io
import time

import numpy as np
from aiida import orm
from aiida.engine import calcfunction

from .data import CandidatesData, ObservationsData, normalize_observations

SINGLE_SCORES = ("TS", "EI", "PI")
MULTI_SCORES = ("TS", "EHVI", "HVPI")

# the keys of the `parameters` Dict of propose and their defaults (the single source for cli/steps.py)
PROPOSE_DEFAULTS = {
    "score": "TS",                 # acquisition: TS/EI/PI (single), TS/EHVI/HVPI (multi)
    "num_rand_basis": 0,           # 0: exact Gaussian process; >0: random-feature Bayesian linear model
    "num_search_each_probe": 1,    # how many candidates to propose
    "num_objectives": None,        # default: from the observations (1 when there are none)
    "seed": None,                  # numpy seed (None: not seeded)
    "maximize": True,              # False: the stored t is minimized (propose sees -t)
    "random": False,               # True: random proposals (also used automatically when nothing is observed yet)
    "interval": 0,                 # hyperparameter learning interval passed to bayes_search (0: learn once)
    "posterior": False,            # also return fmean / fstd on every candidate (costly for large N)
}


def _nondominated_mask(t: np.ndarray) -> np.ndarray:
    """rows of t (M, k) that are not dominated (maximization of every column)."""
    M = t.shape[0]
    mask = np.ones(M, dtype=bool)
    for i in range(M):
        if not mask[i]:
            continue
        ge = np.all(t >= t[i], axis=1)
        gt = np.any(t > t[i], axis=1)
        dominated_by = ge & gt
        if np.any(dominated_by):
            mask[i] = False
    return mask


def best_of(actions: np.ndarray, t: np.ndarray, maximize: bool = True, max_front: int = 200) -> dict:
    """best observation (single objective) or Pareto front (multi) of raw values t (M, k)."""
    if actions.shape[0] == 0:
        return {"num_observations": 0}
    sign = 1.0 if maximize else -1.0
    if t.shape[1] == 1:
        i = int(np.argmax(sign * t[:, 0]))
        return {"num_observations": int(actions.shape[0]), "best_action": int(actions[i]),
                "best_value": float(t[i, 0]), "best_step": i + 1}
    mask = _nondominated_mask(sign * t)
    idx = np.nonzero(mask)[0]
    return {"num_observations": int(actions.shape[0]), "pareto_size": int(idx.size),
            "pareto_actions": [int(a) for a in actions[idx][:max_front]],
            "pareto_values": t[idx][:max_front].tolist(),
            "pareto_truncated": bool(idx.size > max_front)}


def best_sequence(t: np.ndarray, maximize: bool = True) -> list:
    """best value after each observation (single objective)."""
    if t.shape[0] == 0 or t.shape[1] != 1:
        return []
    acc = np.maximum.accumulate(t[:, 0]) if maximize else np.minimum.accumulate(t[:, 0])
    return [float(v) for v in acc]


# ------------------------------------------------------------------ candidates
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


# ------------------------------------------------------------------ observe
@calcfunction
def observe(candidates: CandidatesData, new: orm.ArrayData, observations: ObservationsData = None) -> ObservationsData:
    """Append new observations (arrays `actions`, `t`) to `observations` (or start a chain). Duplicated actions are rejected."""
    N = candidates.num_candidates
    actions, t = normalize_observations(new.get_array("actions"), new.get_array("t"))
    if actions.size == 0:
        raise ValueError("no new observations")
    if np.any(actions < 0) or np.any(actions >= N):
        raise ValueError(f"actions must be in [0, {N})")
    if observations is not None:
        prev_a, prev_t = observations.actions, observations.t
        if prev_t.shape[1] != t.shape[1]:
            raise ValueError(f"number of objectives changed: {prev_t.shape[1]} -> {t.shape[1]}")
    else:
        prev_a, prev_t = np.zeros(0, dtype=np.int64), np.zeros((0, t.shape[1]))
    all_a = np.concatenate([prev_a, actions])
    uniq, counts = np.unique(all_a, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        raise ValueError(f"actions observed more than once: {dup.tolist()}")
    return ObservationsData(actions=all_a, t=np.concatenate([prev_t, t]))


# ------------------------------------------------------------------ propose
def _policy(X, k, actions, t_fit):
    import physbo

    initial = (actions, t_fit if k > 1 else t_fit[:, 0]) if actions.shape[0] else None
    if k == 1:
        return physbo.search.discrete.Policy(test_X=X, initial_data=initial)
    return physbo.search.discrete_multi.Policy(test_X=X, num_objectives=k, initial_data=initial)


def run_propose(X: np.ndarray, actions: np.ndarray, t: np.ndarray, params: dict):
    """the PHYSBO call itself (no aiida): returns (proposed actions, summary dict, posterior dict | None)."""
    import physbo

    p = dict(PROPOSE_DEFAULTS, **params)
    unknown = set(params) - set(PROPOSE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown propose parameters {sorted(unknown)}; known {sorted(PROPOSE_DEFAULTS)}")
    N = X.shape[0]
    k = int(p["num_objectives"] or (t.shape[1] if t.size else 1))
    if t.size and t.shape[1] != k:
        raise ValueError(f"num_objectives={k} but the observations have {t.shape[1]} columns")
    scores = SINGLE_SCORES if k == 1 else MULTI_SCORES
    score = str(p["score"]).upper()
    if score not in scores:
        raise ValueError(f"score {score!r} is not available for {k} objective(s); choose from {scores}")
    n = int(p["num_search_each_probe"])
    if n < 1:
        raise ValueError("num_search_each_probe must be >= 1")
    maximize = bool(p["maximize"])
    sign = 1.0 if maximize else -1.0
    remaining = N - actions.shape[0]
    if remaining <= 0:
        raise ValueError("every candidate has been observed already")
    n = min(n, remaining)
    mode = "random" if (p["random"] or actions.shape[0] == 0) else "bayes"

    t0 = time.time()
    policy = _policy(X, k, actions, sign * t)
    if p["seed"] is not None:
        policy.set_seed(int(p["seed"]))
    if mode == "random":
        proposed = policy.random_search(max_num_probes=1, num_search_each_probe=n, simulator=None, is_disp=False)
    else:
        proposed = policy.bayes_search(max_num_probes=1, num_search_each_probe=n, simulator=None, score=score,
                                       interval=int(p["interval"]), num_rand_basis=int(p["num_rand_basis"]),
                                       is_disp=False)
    proposed = np.asarray(proposed, dtype=np.int64).reshape(-1)
    posterior = None
    if mode == "bayes" and p["posterior"]:
        fmean = np.asarray(policy.get_post_fmean(X), dtype=float)
        fvar = np.asarray(policy.get_post_fcov(X, diag=True), dtype=float)
        fmean = fmean.reshape(N, -1)
        fstd = np.sqrt(np.clip(fvar.reshape(N, -1), 0.0, None))
        posterior = {"fmean": sign * fmean, "fstd": fstd}
    summary = {"mode": mode, "score": score if mode == "bayes" else None, "num_rand_basis": int(p["num_rand_basis"]),
               "num_search_each_probe": n, "seed": p["seed"], "maximize": maximize, "num_objectives": k,
               "num_candidates": int(N), "num_observed": int(actions.shape[0]), "num_remaining_after": int(remaining - n),
               "proposed_actions": proposed.tolist(), "best_so_far": best_of(actions, t, maximize),
               "elapsed_seconds": round(time.time() - t0, 3), "physbo_version": physbo.__version__}
    return proposed, summary, posterior


@calcfunction
def propose(candidates: CandidatesData, parameters: orm.Dict, observations: ObservationsData = None) -> dict:
    """Next candidates to evaluate. parameters: see PROPOSE_DEFAULTS. Outputs: proposal (actions, X), summary, [posterior]."""
    X = candidates.X
    if observations is not None:
        actions, t = observations.actions, observations.t
    else:
        actions, t = np.zeros(0, dtype=np.int64), np.zeros((0, 1))
    proposed, summary, posterior = run_propose(X, actions, t, parameters.get_dict())
    proposal = orm.ArrayData()
    proposal.set_array("actions", proposed)
    proposal.set_array("X", X[proposed])
    out = {"proposal": proposal, "summary": orm.Dict(dict=summary)}
    if posterior is not None:
        post = orm.ArrayData()
        post.set_array("fmean", posterior["fmean"])
        post.set_array("fstd", posterior["fstd"])
        out["posterior"] = post
    return out


# ------------------------------------------------------------------ test functions
@calcfunction
def evaluate_test_function(candidates: CandidatesData, proposal: orm.ArrayData, objective: orm.Dict) -> orm.ArrayData:
    """Evaluate a PHYSBO test function ({name, kwargs, maximize}) on the proposed rows -> ArrayData (actions, t)."""
    from . import objectives

    actions = np.asarray(proposal.get_array("actions"), dtype=np.int64).reshape(-1)
    t = objectives.evaluate(objective.get_dict(), candidates.X[actions])
    out = orm.ArrayData()
    out.set_array("actions", actions)
    out.set_array("t", t)
    return out


# ------------------------------------------------------------------ summary
@calcfunction
def summarize(candidates: CandidatesData, observations: ObservationsData, settings: orm.Dict) -> orm.Dict:
    """Best value (or Pareto front), best X and best-so-far sequence of an observations node. settings: {maximize, ...extra}."""
    s = settings.get_dict()
    maximize = bool(s.get("maximize", True))
    best = best_of(observations.actions, observations.t, maximize)
    if "best_action" in best:
        best["best_X"] = candidates.X[best["best_action"]].tolist()
    if "pareto_actions" in best:
        best["pareto_X"] = candidates.X[np.asarray(best["pareto_actions"], dtype=int)].tolist()
    return orm.Dict(dict={**s, "observations_pk": observations.pk, "num_observations": observations.num_observations,
                          "num_objectives": observations.num_objectives, "num_candidates": candidates.num_candidates,
                          "best": best, "best_sequence": best_sequence(observations.t, maximize)})
