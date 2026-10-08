# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""PHYSBO test functions as objectives of the closed-loop WorkChain.

An objective spec is a plain dict: {"name": "Sphere", "kwargs": {"dim": 2}, "maximize": false,
"noise": 0.0, "noise_seed": null, "transform": null}. Names come from physbo.test_functions and from extra_functions.py
(Branin, Hartmann6, Levy, Forrester, DTLZ2, ...). PHYSBO defines every test function as a minimization
problem; here it is evaluated with `test_maximizer=False`, so the stored values are the function values
themselves, and `propose` is told `maximize=False`. Set "maximize": true to store -f instead (then propose
maximizes). "noise" adds Gaussian observation noise of that standard deviation (reproducible per call with
"noise_seed"; the seed is combined with the evaluated coordinates so that every call draws fresh noise).
"""
import inspect

import numpy as np


def _classes(module, base):
    out = {}
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, base) and cls is not base and cls.__module__ == module.__name__:
            out[name] = cls
    return out


def registry():
    """name -> (kind, class) for every single- and multi-objective test function."""
    from physbo.test_functions import multi_objective, single_objective

    from . import extra_functions

    reg = {}
    for name, cls in _classes(single_objective, single_objective.SingleTestFunction).items():
        reg[name] = ("single", cls)
    for name, cls in _classes(multi_objective, multi_objective.MultiTestFunction).items():
        reg[name] = ("multi", cls)
    for name, cls in extra_functions.SINGLE.items():
        reg.setdefault(name, ("single", cls))        # PHYSBO's own definition wins if it ever ships one
    for name, cls in extra_functions.MULTI.items():
        reg.setdefault(name, ("multi", cls))
    return reg


def available():
    """list of {name, kind, parameters, and (when constructible with defaults) dim, nobj, min_X, max_X}."""
    rows = []
    for name, (kind, cls) in sorted(registry().items()):
        params = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "test_maximizer")]
        row = {"name": name, "kind": kind, "parameters": params, "source": cls.__module__.split(".")[0]}
        try:
            fn = cls(test_maximizer=False)
            row.update({"dim": fn.dim, "nobj": fn.nobj, "min_X": fn.min_X.tolist(), "max_X": fn.max_X.tolist()})
            if hasattr(cls, "global_minimum"):
                row["global_minimum"] = float(cls.global_minimum)
        except Exception as exc:  # noqa: BLE001  needs arguments (e.g. Gaussian centers)
            row["needs_kwargs"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def make(spec: dict):
    """instantiate the test function of an objective spec (raises ValueError on an unknown name)."""
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError('objective must be a dict with "name" (see test-functions)')
    reg = registry()
    name = spec["name"]
    if name not in reg:
        raise ValueError(f"unknown test function {name!r}; known: {sorted(reg)}")
    kwargs = dict(spec.get("kwargs") or {})
    cls = reg[name][1]
    return cls(test_maximizer=False, **kwargs)


def evaluate(spec: dict, X: np.ndarray) -> np.ndarray:
    """f(X) as an (n, k) array (+ Gaussian noise of std spec["noise"]); negated when spec["maximize"] is true."""
    fn = make(spec)
    X = np.asarray(X, dtype=float)
    t = np.asarray(fn(X), dtype=float)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    noise = float(spec.get("noise") or 0.0)
    if noise < 0:
        raise ValueError("noise must be >= 0")
    if noise > 0:
        # reproducible: the seed mixes noise_seed with the coordinates, so repeated calls on new points get new draws
        mix = int(abs(hash(np.round(X, 12).tobytes())) % (2**32))
        rng = np.random.default_rng([int(spec.get("noise_seed") or 0), mix])
        t = t + noise * rng.standard_normal(t.shape)
    t = apply_transform(spec, t)
    return -t if spec.get("maximize", False) else t


TRANSFORMS = (None, "log")


def apply_transform(spec: dict, t: np.ndarray) -> np.ndarray:
    """the value transform of an objective spec: None (identity) or "log" (natural log, values must be > 0)."""
    tr = spec.get("transform")
    if tr in (None, "", "none"):
        return t
    if tr == "log":
        if np.any(t <= 0):
            raise ValueError("transform 'log' needs positive function values")
        return np.log(t)
    raise ValueError(f"unknown transform {tr!r}; choose from {TRANSFORMS}")


def known_minimum(spec: dict):
    """(f*, points) of a single-objective function whose optimum is known, else None."""
    fn = make(spec)
    try:
        pts = np.atleast_2d(fn.global_minimum_point())
    except (NotImplementedError, AttributeError):      # multi-objective functions have no single minimum
        return None
    f_star = getattr(type(fn), "global_minimum", None)
    if f_star is None:
        f_star = float(np.min(fn(pts)))
    f_star = float(apply_transform(spec, np.asarray([[f_star]]))[0, 0])
    return f_star, pts.tolist()


def grid_spec(spec: dict, num) -> dict:
    """grid spec (min/max/num) covering the default search box of the test function."""
    fn = make(spec)
    return {"min": fn.min_X.tolist(), "max": fn.max_X.tolist(), "num": num}
