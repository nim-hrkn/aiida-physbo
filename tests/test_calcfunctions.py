"""Layer-2 tests: temporary sqlite profile from aiida-core's pytest fixtures.

    python -m pytest -p aiida.tools.pytest_fixtures tests/test_calcfunctions.py
"""
import numpy as np
import pytest

pytest.importorskip("aiida")
pytest.importorskip("physbo")


@pytest.fixture
def grid(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid

    return candidates_from_grid(orm.Dict(dict={"min": [-1.0, -1.0], "max": [1.0, 1.0], "num": 11}))


def test_candidates_from_grid_is_typed_and_in_provenance(grid):
    from aiida import orm

    from aiida_physbo.data import CandidatesData

    assert isinstance(grid, CandidatesData)
    assert grid.num_candidates == 121 and grid.dim == 2
    assert grid.creator is not None and grid.creator.process_label == "candidates_from_grid"
    loaded = orm.load_node(grid.pk)
    assert isinstance(loaded, CandidatesData), type(loaded)          # the entry point is registered
    assert orm.QueryBuilder().append(CandidatesData).count() == 1


def test_candidates_from_file_csv(aiida_profile_clean, tmp_path):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_file

    path = tmp_path / "cands.csv"
    path.write_text("# x, y, name\n0,0,a\n1,0,b\n0,1,c\n")
    node = candidates_from_file(orm.SinglefileData(file=str(path)), orm.Dict(dict={"columns": [0, 1], "names": ["x", "y"]}))
    assert node.X.tolist() == [[0, 0], [1, 0], [0, 1]]
    assert node.columns == ["x", "y"]


def test_observe_chain_and_rejections(grid):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe
    from aiida_physbo.data import ObservationsData

    def new(actions, t):
        a = orm.ArrayData()
        a.set_array("actions", np.asarray(actions))
        a.set_array("t", np.asarray(t))
        return a

    obs1 = observe(grid, new([0, 5], [0.1, 0.2]))
    assert isinstance(obs1, ObservationsData) and obs1.num_observations == 2 and obs1.num_objectives == 1
    assert obs1.t.shape == (2, 1)
    obs2 = observe(grid, new([7], [0.3]), obs1)
    assert obs2.actions.tolist() == [0, 5, 7]
    with pytest.raises(ValueError, match="more than once"):
        observe(grid, new([5], [9.0]), obs2)
    with pytest.raises(ValueError, match=r"in \[0, 121\)"):
        observe(grid, new([121], [1.0]))
    with pytest.raises(ValueError, match="objectives changed"):
        observe(grid, new([8], [[1.0, 2.0]]), obs2)


def test_propose_random_then_bayes_single(grid):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose

    out = propose(grid, orm.Dict(dict={"seed": 1, "num_search_each_probe": 5}))
    s = out["summary"].get_dict()
    assert s["mode"] == "random" and len(out["proposal"].get_array("actions")) == 5
    assert out["proposal"].get_array("X").shape == (5, 2)
    X = grid.X
    actions = out["proposal"].get_array("actions")
    t = -np.sum(X[actions] ** 2, axis=1)          # maximize -|x|^2
    new = orm.ArrayData()
    new.set_array("actions", actions)
    new.set_array("t", t)
    obs = observe(grid, new)
    for score in ("TS", "EI", "PI"):
        out = propose(grid, orm.Dict(dict={"seed": 2, "score": score, "posterior": True}), obs)
        s = out["summary"].get_dict()
        assert s["mode"] == "bayes" and s["score"] == score and s["num_observed"] == 5
        a = int(out["proposal"].get_array("actions")[0])
        assert a not in actions.tolist(), "an observed candidate must not be proposed again"
        post = out["posterior"]
        assert post.get_array("fmean").shape == (121, 1) and post.get_array("fstd").shape == (121, 1)
        assert "best_action" in s["best_so_far"]
    # random-feature BLM
    out = propose(grid, orm.Dict(dict={"seed": 3, "score": "TS", "num_rand_basis": 50, "num_search_each_probe": 2}), obs)
    assert len(out["proposal"].get_array("actions")) == 2
    # minimize: best_so_far is the smallest raw value
    out = propose(grid, orm.Dict(dict={"seed": 3, "maximize": False}), obs)
    assert out["summary"].get_dict()["best_so_far"]["best_value"] == float(t.min())
    with pytest.raises(ValueError, match="unknown propose parameters"):
        propose(grid, orm.Dict(dict={"nope": 1}), obs)
    with pytest.raises(ValueError, match="not available"):
        propose(grid, orm.Dict(dict={"score": "EHVI"}), obs)


def test_propose_multi_objective(grid):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose

    X = grid.X
    actions = np.array([0, 10, 60, 110, 120])
    t = np.stack([-np.sum((X[actions] - 0.5) ** 2, axis=1), -np.sum((X[actions] + 0.5) ** 2, axis=1)], axis=1)
    new = orm.ArrayData()
    new.set_array("actions", actions)
    new.set_array("t", t)
    obs = observe(grid, new)
    assert obs.num_objectives == 2
    for score in ("EHVI", "HVPI", "TS"):
        out = propose(grid, orm.Dict(dict={"seed": 1, "score": score, "posterior": True}), obs)
        s = out["summary"].get_dict()
        assert s["num_objectives"] == 2 and s["score"] == score
        assert "pareto_actions" in s["best_so_far"] and s["best_so_far"]["pareto_size"] >= 1
        assert out["posterior"].get_array("fmean").shape == (121, 2)


def test_propose_exhausted(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid, observe, propose

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0], "max": [1.0], "num": 3}))
    new = orm.ArrayData()
    new.set_array("actions", np.array([0, 1, 2]))
    new.set_array("t", np.array([1.0, 2.0, 3.0]))
    obs = observe(cand, new)
    with pytest.raises(ValueError, match="observed already"):
        propose(cand, orm.Dict(dict={}), obs)


def test_evaluate_test_function_and_objectives(grid):
    from aiida import orm

    from aiida_physbo import objectives
    from aiida_physbo.calcfunctions import evaluate_test_function, propose

    names = {r["name"] for r in objectives.available()}
    assert {"Sphere", "Rastrigin", "ZDT1"} <= names
    out = propose(grid, orm.Dict(dict={"seed": 1, "num_search_each_probe": 3}))
    ev = evaluate_test_function(grid, out["proposal"], orm.Dict(dict={"name": "Sphere", "kwargs": {"dim": 2}}))
    a = ev.get_array("actions")
    assert np.allclose(ev.get_array("t")[:, 0], np.sum(grid.X[a] ** 2, axis=1))      # f itself (minimization)
    ev2 = evaluate_test_function(grid, out["proposal"], orm.Dict(dict={"name": "Sphere", "maximize": True}))
    assert np.allclose(ev2.get_array("t"), -ev.get_array("t"))
    with pytest.raises(ValueError, match="unknown test function"):
        objectives.make({"name": "NoSuchFunction"})
