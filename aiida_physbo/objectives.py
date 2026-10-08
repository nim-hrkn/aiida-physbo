# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""PHYSBO test functions as objectives of the closed-loop WorkChain.

An objective spec is a plain dict: {"name": "Sphere", "kwargs": {"dim": 2}, "maximize": false}.
PHYSBO defines every test function as a minimization problem; here it is evaluated with
`test_maximizer=False`, so the stored values are the function values themselves, and `propose`
is told `maximize=False`. Set "maximize": true to store -f instead (then propose maximizes).
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

    reg = {}
    for name, cls in _classes(single_objective, single_objective.SingleTestFunction).items():
        reg[name] = ("single", cls)
    for name, cls in _classes(multi_objective, multi_objective.MultiTestFunction).items():
        reg[name] = ("multi", cls)
    return reg


def available():
    """list of {name, kind, parameters, and (when constructible with defaults) dim, nobj, min_X, max_X}."""
    rows = []
    for name, (kind, cls) in sorted(registry().items()):
        params = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "test_maximizer")]
        row = {"name": name, "kind": kind, "parameters": params}
        try:
            fn = cls(test_maximizer=False)
            row.update({"dim": fn.dim, "nobj": fn.nobj, "min_X": fn.min_X.tolist(), "max_X": fn.max_X.tolist()})
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
    """f(X) as an (n, k) array; negated when spec["maximize"] is true."""
    fn = make(spec)
    t = np.asarray(fn(np.asarray(X, dtype=float)), dtype=float)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    return -t if spec.get("maximize", False) else t


def grid_spec(spec: dict, num) -> dict:
    """grid spec (min/max/num) covering the default search box of the test function."""
    fn = make(spec)
    return {"min": fn.min_X.tolist(), "max": fn.max_X.tolist(), "num": num}
