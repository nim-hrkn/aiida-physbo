# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Typed data nodes (entry points `physbo.candidates` and `physbo.observations` in pyproject.toml).

CandidatesData   array `X` (N, d): the discrete search space, one row per candidate (PHYSBO test_X).
ObservationsData arrays `actions` (M,) int and `t` (M, k) float: evaluated candidates (row indices of X)
                 and their objective values, in the order they were observed. `t` is always 2-D (k objectives).

Both are ArrayData subclasses so that `QueryBuilder.append(CandidatesData)` finds them by type.
"""
import numpy as np
from aiida import orm


class CandidatesData(orm.ArrayData):
    """The set of candidates: rows of `X`."""

    def __init__(self, X=None, columns=None, **kwargs):
        super().__init__(**kwargs)
        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.ndim != 2:
                raise ValueError("X must be a 2-D array (N candidates x d features)")
            if not np.all(np.isfinite(X)):
                raise ValueError("X contains NaN or inf")
            self.set_array("X", X)
        if columns is not None:
            self.base.attributes.set("columns", [str(c) for c in columns])

    @property
    def X(self) -> np.ndarray:
        return self.get_array("X")

    @property
    def num_candidates(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def columns(self):
        return self.base.attributes.get("columns", None)


class ObservationsData(orm.ArrayData):
    """Observed (action, t) pairs."""

    def __init__(self, actions=None, t=None, **kwargs):
        super().__init__(**kwargs)
        if actions is not None or t is not None:
            actions, t = normalize_observations(actions, t)
            self.set_array("actions", actions)
            self.set_array("t", t)

    @property
    def actions(self) -> np.ndarray:
        return self.get_array("actions")

    @property
    def t(self) -> np.ndarray:
        return self.get_array("t")

    @property
    def num_observations(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_objectives(self) -> int:
        return int(self.t.shape[1])


def normalize_observations(actions, t):
    """actions -> (M,) int64, t -> (M, k) float; raises on shape mismatch or non-finite values."""
    actions = np.asarray(actions)
    if actions.ndim != 1:
        actions = actions.reshape(-1)
    if actions.size and not np.all(np.equal(np.mod(actions, 1), 0)):
        raise ValueError("actions must be integers (row indices of the candidates)")
    actions = actions.astype(np.int64)
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        t = t.reshape(1, 1)
    elif t.ndim == 1:
        t = t.reshape(-1, 1)
    elif t.ndim != 2:
        raise ValueError("t must be 1-D (M,) or 2-D (M, k)")
    if t.shape[0] != actions.shape[0]:
        raise ValueError(f"len(actions)={actions.shape[0]} but t has {t.shape[0]} rows")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains NaN or inf")
    return actions, t
