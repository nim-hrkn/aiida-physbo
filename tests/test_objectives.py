"""Layer 1 (needs physbo, no profile): the extra benchmark functions and the objective evaluation."""
import numpy as np
import pytest

pytest.importorskip("physbo")


@pytest.mark.parametrize("name", ["Branin", "GoldsteinPrice", "SixHumpCamel", "Levy", "Hartmann3", "Hartmann6", "Forrester", "GramacyLee"])
def test_known_minimum_is_attained(name):
    from aiida_physbo import objectives

    fn = objectives.make({"name": name})
    pts = np.atleast_2d(fn.global_minimum_point())
    vals = fn(pts)[:, 0]
    assert vals.shape[0] == pts.shape[0]
    assert np.allclose(vals, type(fn).global_minimum, atol=2e-4), (name, vals)
    assert (pts >= fn.min_X - 1e-9).all() and (pts <= fn.max_X + 1e-9).all()
    # nowhere on a coarse random sample is f below the stated minimum
    rng = np.random.default_rng(0)
    X = fn.min_X + (fn.max_X - fn.min_X) * rng.random((2000, fn.dim))
    assert fn(X).min() >= type(fn).global_minimum - 1e-9
    km = objectives.known_minimum({"name": name})
    assert km[0] == pytest.approx(type(fn).global_minimum) and len(km[1]) == pts.shape[0]


def test_dtlz2_front_is_unit_sphere():
    from aiida_physbo import objectives

    fn = objectives.make({"name": "DTLZ2", "kwargs": {"nobj": 3, "dim": 5}})
    assert fn.nobj == 3 and fn.dim == 5
    X = np.random.default_rng(1).random((50, 5))
    X[:, 2:] = 0.5                                     # on the front: g = 0
    F = fn(X)
    assert np.allclose(np.sum(F**2, axis=1), 1.0)
    X[:, 2:] = 0.0                                     # off the front: g > 0, every objective larger
    assert (np.sum(fn(X) ** 2, axis=1) > 1.0).all()
    assert fn.reference_min.tolist() == [0, 0, 0] and len(fn.reference_max) == 3
    with pytest.raises(ValueError):
        objectives.make({"name": "DTLZ2", "kwargs": {"nobj": 3, "dim": 2}})


def test_registry_and_available():
    from aiida_physbo import objectives

    rows = {r["name"]: r for r in objectives.available()}
    assert {"Sphere", "ZDT1", "Branin", "Hartmann6", "Forrester", "DTLZ2"} <= set(rows)
    assert rows["Branin"]["source"] == "aiida_physbo" and rows["Sphere"]["source"] == "physbo"
    assert rows["Branin"]["global_minimum"] == pytest.approx(0.397887, abs=1e-5)
    assert rows["Hartmann6"]["dim"] == 6 and rows["DTLZ2"]["kind"] == "multi"


def test_noise_is_reproducible_and_fresh_per_point():
    from aiida_physbo import objectives

    X = np.array([[0.2], [0.7]])
    clean = objectives.evaluate({"name": "Forrester"}, X)
    a = objectives.evaluate({"name": "Forrester", "noise": 0.5, "noise_seed": 3}, X)
    b = objectives.evaluate({"name": "Forrester", "noise": 0.5, "noise_seed": 3}, X)
    c = objectives.evaluate({"name": "Forrester", "noise": 0.5, "noise_seed": 4}, X)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, clean)
    assert np.abs(a - clean).max() < 3.0
    d = objectives.evaluate({"name": "Forrester", "noise": 0.5, "noise_seed": 3}, X[:1])
    assert d.shape == (1, 1) and not np.isclose(d[0, 0], a[0, 0])     # a different call draws different noise
    with pytest.raises(ValueError, match="noise"):
        objectives.evaluate({"name": "Forrester", "noise": -1}, X)
    assert objectives.known_minimum({"name": "DTLZ2"}) is None            # multi: no single minimum
