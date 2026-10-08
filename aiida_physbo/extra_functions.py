# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Benchmark functions that PHYSBO 3.2.1 does not ship, in PHYSBO's TestFunction form (minimization problems).

Single objective: Branin, GoldsteinPrice, SixHumpCamel, Levy, Hartmann3, Hartmann6, Forrester, GramacyLee.
Multi objective:  DTLZ2 (nobj objectives, dim >= nobj).
All are registered by objectives.registry() next to physbo.test_functions; `global_minimum_point()` and
`global_minimum` give the known optimum used by the tests and the benchmark table.
"""
import numpy as np
from physbo.test_functions.multi_objective import MultiTestFunction
from physbo.test_functions.single_objective import SingleTestFunction


class Branin(SingleTestFunction):
    """Branin-Hoo (2-D): three equal global minima f = 0.397887 at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)."""

    global_minimum = 0.39788735772973816

    def __init__(self, min_X=None, max_X=None, test_maximizer=True):
        super().__init__(dim=2, min_X=[-5.0, 0.0] if min_X is None else min_X,
                         max_X=[10.0, 15.0] if max_X is None else max_X, test_maximizer=test_maximizer)

    def f(self, x):
        a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
        x1, x2 = x[:, 0], x[:, 1]
        return (a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)[:, None]

    def global_minimum_point(self):
        return np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])


class GoldsteinPrice(SingleTestFunction):
    """Goldstein-Price (2-D): f = 3 at (0, -1); values span 3 .. 1e6 on [-2, 2]^2."""

    global_minimum = 3.0

    def __init__(self, min_X=-2.0, max_X=2.0, test_maximizer=True):
        super().__init__(dim=2, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def f(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
        b = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
        return (a * b)[:, None]

    def global_minimum_point(self):
        return np.array([[0.0, -1.0]])


class SixHumpCamel(SingleTestFunction):
    """Six-hump camel (2-D): two global minima f = -1.0316 at (0.0898, -0.7126) and (-0.0898, 0.7126)."""

    global_minimum = -1.0316284534898774

    def __init__(self, min_X=None, max_X=None, test_maximizer=True):
        super().__init__(dim=2, min_X=[-3.0, -2.0] if min_X is None else min_X,
                         max_X=[3.0, 2.0] if max_X is None else max_X, test_maximizer=test_maximizer)

    def f(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return ((4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2)[:, None]

    def global_minimum_point(self):
        return np.array([[0.0898, -0.7126], [-0.0898, 0.7126]])


class Levy(SingleTestFunction):
    """Levy (d-D): many local minima, f = 0 at (1, ..., 1)."""

    global_minimum = 0.0

    def __init__(self, dim=2, min_X=-10.0, max_X=10.0, test_maximizer=True):
        super().__init__(dim=dim, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def f(self, x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[:, 0]) ** 2
        term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        mid = (w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2)
        return (term1 + np.sum(mid, axis=1) + term3)[:, None]

    def global_minimum_point(self):
        return np.ones((1, self.dim))


class _Hartmann(SingleTestFunction):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def f(self, x):
        inner = np.sum(self.A[None, :, :] * (x[:, None, :] - self.P[None, :, :]) ** 2, axis=2)
        return (-np.sum(self.alpha[None, :] * np.exp(-inner), axis=1))[:, None]


class Hartmann3(_Hartmann):
    """Hartmann 3-D on [0, 1]^3: f = -3.86278 at (0.114614, 0.555649, 0.852547)."""

    global_minimum = -3.86278214782076
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])

    def __init__(self, min_X=0.0, max_X=1.0, test_maximizer=True):
        super().__init__(dim=3, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def global_minimum_point(self):
        return np.array([[0.114614, 0.555649, 0.852547]])


class Hartmann6(_Hartmann):
    """Hartmann 6-D on [0, 1]^6: f = -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)."""

    global_minimum = -3.322368011391339
    A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])

    def __init__(self, min_X=0.0, max_X=1.0, test_maximizer=True):
        super().__init__(dim=6, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def global_minimum_point(self):
        return np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])


class Forrester(SingleTestFunction):
    """Forrester (1-D) on [0, 1]: f = (6x - 2)^2 sin(12x - 4), minimum -6.02074 at x = 0.757249."""

    global_minimum = -6.020740055767083

    def __init__(self, min_X=0.0, max_X=1.0, test_maximizer=True):
        super().__init__(dim=1, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def f(self, x):
        return ((6 * x[:, 0] - 2) ** 2 * np.sin(12 * x[:, 0] - 4))[:, None]

    def global_minimum_point(self):
        return np.array([[0.757249]])


class GramacyLee(SingleTestFunction):
    """Gramacy & Lee (1-D) on [0.5, 2.5]: f = sin(10 pi x) / (2x) + (x - 1)^4, minimum -0.869011 at x = 0.548563."""

    global_minimum = -0.8690111349894997

    def __init__(self, min_X=0.5, max_X=2.5, test_maximizer=True):
        super().__init__(dim=1, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def f(self, x):
        x1 = x[:, 0]
        return (np.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4)[:, None]

    def global_minimum_point(self):
        return np.array([[0.548563]])


class DTLZ2(MultiTestFunction):
    """DTLZ2: nobj objectives on [0, 1]^dim (dim >= nobj); the Pareto front is the positive orthant of the unit sphere
    (sum f_i^2 = 1), reached when x_i = 0.5 for i >= nobj."""

    def __init__(self, nobj=2, dim=3, min_X=0.0, max_X=1.0, test_maximizer=True):
        if dim < nobj:
            raise ValueError("DTLZ2 needs dim >= nobj")
        super().__init__(nobj=nobj, dim=dim, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer)

    def f(self, x):
        k = self.nobj - 1
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1)
        out = np.empty((x.shape[0], self.nobj))
        for i in range(self.nobj):
            v = 1 + g
            for j in range(k - i):
                v = v * np.cos(0.5 * np.pi * x[:, j])
            if i > 0:
                v = v * np.sin(0.5 * np.pi * x[:, k - i])
            out[:, i] = v
        return out

    def _ref_min(self):
        return np.zeros(self.nobj)

    def _ref_max(self):
        return np.full(self.nobj, 1.0 + 0.25 * (self.dim - self.nobj + 1))


SINGLE = {c.__name__: c for c in (Branin, GoldsteinPrice, SixHumpCamel, Levy, Hartmann3, Hartmann6, Forrester, GramacyLee)}
MULTI = {"DTLZ2": DTLZ2}
