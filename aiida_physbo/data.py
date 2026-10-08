# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Typed data nodes (entry points `physbo.candidates`, `physbo.search_box`, `physbo.observations`).

Two kinds of search space:

CandidatesData  array `X` (N, d): a discrete set of candidates, one row each (PHYSBO `discrete` policies,
                proposals are row indices = actions).
SearchBoxData   arrays `min_X`, `max_X` (d,): a continuous box (PHYSBO `range` policies, proposals are
                coordinates).

ObservationsData arrays `t` (M, k) float, always 2-D, in observation order; `X` (M, d) the observed
                coordinates; and, for a discrete space only, `actions` (M,) int. Attribute `space` is
                "discrete" or "range". Nodes written by 0.1.0 have `actions` and `t` only (X is then None).

All are ArrayData subclasses so that `QueryBuilder.append(CandidatesData)` finds them by type.
"""
import numpy as np
from aiida import orm

DISCRETE, RANGE = "discrete", "range"


def _columns(node, columns):
    if columns is not None:
        node.base.attributes.set("columns", [str(c) for c in columns])


class CandidatesData(orm.ArrayData):
    """The discrete search space: rows of `X`."""

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
        _columns(self, columns)

    space = DISCRETE

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


class SearchBoxData(orm.ArrayData):
    """The continuous search space: the box [min_X, max_X]."""

    def __init__(self, min_X=None, max_X=None, columns=None, **kwargs):
        super().__init__(**kwargs)
        if min_X is not None or max_X is not None:
            lo = np.asarray(min_X, dtype=float).reshape(-1)
            hi = np.asarray(max_X, dtype=float).reshape(-1)
            if lo.shape != hi.shape or lo.size == 0:
                raise ValueError("min_X and max_X must be 1-D arrays of the same length")
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                raise ValueError("min_X / max_X contain NaN or inf")
            if not np.all(lo < hi):
                raise ValueError("min_X must be smaller than max_X in every dimension")
            self.set_array("min_X", lo)
            self.set_array("max_X", hi)
        _columns(self, columns)

    space = RANGE

    @property
    def min_X(self) -> np.ndarray:
        return self.get_array("min_X")

    @property
    def max_X(self) -> np.ndarray:
        return self.get_array("max_X")

    @property
    def dim(self) -> int:
        return int(self.min_X.shape[0])

    @property
    def columns(self):
        return self.base.attributes.get("columns", None)

    def contains(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, self.dim)
        return np.all((X >= self.min_X) & (X <= self.max_X), axis=1)


def space_of(node):
    """'discrete' | 'range' for a CandidatesData / SearchBoxData (raises otherwise)."""
    if isinstance(node, CandidatesData):
        return DISCRETE
    if isinstance(node, SearchBoxData):
        return RANGE
    raise ValueError(f"Node<{node.pk}> is a {node.__class__.__name__}, not CandidatesData or SearchBoxData")


class ObservationsData(orm.ArrayData):
    """Observed points and their objective values."""

    def __init__(self, actions=None, X=None, t=None, **kwargs):
        super().__init__(**kwargs)
        if t is not None:
            actions, X, t = normalize_observations(actions, X, t)
            self.set_array("t", t)
            if X is not None:
                self.set_array("X", X)
            if actions is not None:
                self.set_array("actions", actions)
            self.base.attributes.set("space", DISCRETE if actions is not None else RANGE)

    @property
    def space(self) -> str:
        return self.base.attributes.get("space", DISCRETE if "actions" in self.get_arraynames() else RANGE)

    @property
    def actions(self):
        return self.get_array("actions") if "actions" in self.get_arraynames() else None

    @property
    def X(self):
        return self.get_array("X") if "X" in self.get_arraynames() else None

    @property
    def t(self) -> np.ndarray:
        return self.get_array("t")

    @property
    def num_observations(self) -> int:
        return int(self.t.shape[0])

    @property
    def num_objectives(self) -> int:
        return int(self.t.shape[1])

    @property
    def dim(self):
        return int(self.X.shape[1]) if self.X is not None else None


def normalize_observations(actions, X, t):
    """actions -> (M,) int64 | None, X -> (M, d) float | None, t -> (M, k) float; raises on mismatch / non-finite."""
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        t = t.reshape(1, 1)
    elif t.ndim == 1:
        t = t.reshape(-1, 1)
    elif t.ndim != 2:
        raise ValueError("t must be 1-D (M,) or 2-D (M, k)")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains NaN or inf")
    M = t.shape[0]
    if actions is not None:
        actions = np.asarray(actions).reshape(-1)
        if actions.size and not np.all(np.equal(np.mod(actions, 1), 0)):
            raise ValueError("actions must be integers (row indices of the candidates)")
        actions = actions.astype(np.int64)
        if actions.shape[0] != M:
            raise ValueError(f"len(actions)={actions.shape[0]} but t has {M} rows")
    if X is not None:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(M, -1) if M else X.reshape(0, max(X.size, 1))
        if X.ndim != 2 or X.shape[0] != M:
            raise ValueError(f"X must be (M, d) with M={M} rows; got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
    if actions is None and X is None:
        raise ValueError("observations need actions (discrete space) or X (range space)")
    return actions, X, t
