"""Layer-2 tests: temporary sqlite profile from aiida-core's pytest fixtures.

    python -m pytest -p aiida.tools.pytest_fixtures tests/test_calcfunctions.py
"""
import numpy as np
import pytest

pytest.importorskip("aiida")
pytest.importorskip("physbo")


def _new(actions=None, X=None, t=None):
    from aiida import orm

    a = orm.ArrayData()
    if actions is not None:
        a.set_array("actions", np.asarray(actions))
    if X is not None:
        a.set_array("X", np.asarray(X, dtype=float))
    a.set_array("t", np.asarray(t, dtype=float))
    return a


@pytest.fixture
def grid(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid

    return candidates_from_grid(orm.Dict(dict={"min": [-1.0, -1.0], "max": [1.0, 1.0], "num": 11}))


@pytest.fixture
def box(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import search_box

    return search_box(orm.Dict(dict={"min": [-1.0, -1.0], "max": [1.0, 1.0], "names": ["x", "y"]}))


# ------------------------------------------------------------------ spaces
def test_candidates_from_grid_is_typed_and_in_provenance(grid):
    from aiida import orm

    from aiida_physbo.data import CandidatesData

    assert isinstance(grid, CandidatesData) and grid.space == "discrete"
    assert grid.num_candidates == 121 and grid.dim == 2
    assert grid.creator is not None and grid.creator.process_label == "candidates_from_grid"
    assert isinstance(orm.load_node(grid.pk), CandidatesData)          # the entry point is registered
    assert orm.QueryBuilder().append(CandidatesData).count() == 1


def test_search_box_is_typed(box):
    from aiida import orm

    from aiida_physbo.calcfunctions import search_box
    from aiida_physbo.data import CandidatesData, SearchBoxData

    assert isinstance(orm.load_node(box.pk), SearchBoxData) and box.space == "range"
    assert box.dim == 2 and box.min_X.tolist() == [-1, -1] and box.columns == ["x", "y"]
    assert box.contains([[0, 0], [2, 0]]).tolist() == [True, False]
    assert orm.QueryBuilder().append(CandidatesData).count() == 0     # a different type
    with pytest.raises(ValueError, match="smaller than max_X"):
        search_box(orm.Dict(dict={"min": [0.0], "max": [0.0]}))


def test_candidates_from_file_csv(aiida_profile_clean, tmp_path):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_file

    path = tmp_path / "cands.csv"
    path.write_text("# x, y, name\n0,0,a\n1,0,b\n0,1,c\n")
    node = candidates_from_file(orm.SinglefileData(file=str(path)), orm.Dict(dict={"columns": [0, 1], "names": ["x", "y"]}))
    assert node.X.tolist() == [[0, 0], [1, 0], [0, 1]]
    assert node.columns == ["x", "y"]


# ------------------------------------------------------------------ observe
def test_observe_discrete_chain_and_rejections(grid):
    from aiida_physbo.calcfunctions import observe
    from aiida_physbo.data import ObservationsData

    obs1 = observe(grid, _new(actions=[0, 5], t=[0.1, 0.2]))
    assert isinstance(obs1, ObservationsData) and obs1.space == "discrete"
    assert obs1.num_observations == 2 and obs1.num_objectives == 1 and obs1.t.shape == (2, 1)
    assert obs1.X.tolist() == grid.X[[0, 5]].tolist()                  # X is filled from the candidates
    obs2 = observe(grid, _new(actions=[7], t=[0.3]), obs1)
    assert obs2.actions.tolist() == [0, 5, 7] and obs2.X.shape == (3, 2)
    with pytest.raises(ValueError, match="more than once"):
        observe(grid, _new(actions=[5], t=[9.0]), obs2)
    with pytest.raises(ValueError, match=r"in \[0, 121\)"):
        observe(grid, _new(actions=[121], t=[1.0]))
    with pytest.raises(ValueError, match="objectives changed"):
        observe(grid, _new(actions=[8], t=[[1.0, 2.0]]), obs2)
    with pytest.raises(ValueError, match="needs the array actions"):
        observe(grid, _new(X=[[0.0, 0.0]], t=[1.0]))


def test_observe_range_chain_and_rejections(grid, box):
    from aiida_physbo.calcfunctions import observe

    obs1 = observe(box, _new(X=[[0.1, 0.2], [0.3, -0.4]], t=[1.0, 2.0]))
    assert obs1.space == "range" and obs1.actions is None and obs1.X.shape == (2, 2) and obs1.dim == 2
    obs2 = observe(box, _new(X=[[0.1, 0.2]], t=[1.5]), obs1)          # repeated coordinates are allowed (noise)
    assert obs2.num_observations == 3
    with pytest.raises(ValueError, match="dimension 3"):
        observe(box, _new(X=[[0.0, 0.0, 0.0]], t=[1.0]))
    with pytest.raises(ValueError, match="needs the array X"):
        observe(box, _new(actions=[1], t=[1.0]))
    with pytest.raises(ValueError, match="different kind|given space is"):
        observe(grid, _new(actions=[1], t=[1.0]), obs2)


# ------------------------------------------------------------------ propose: discrete
def test_propose_discrete_random_then_bayes_single(grid):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose

    out = propose(grid, orm.Dict(dict={"seed": 1, "num_search_each_probe": 5}))
    s = out["summary"].get_dict()
    assert s["mode"] == "random" and s["space"] == "discrete" and len(out["proposal"].get_array("actions")) == 5
    assert out["proposal"].get_array("X").shape == (5, 2)
    actions = out["proposal"].get_array("actions")
    t = -np.sum(grid.X[actions] ** 2, axis=1)          # maximize -|x|^2
    obs = observe(grid, _new(actions=actions, t=t))
    for score in ("TS", "EI", "PI"):
        out = propose(grid, orm.Dict(dict={"seed": 2, "score": score, "posterior": True}), obs)
        s = out["summary"].get_dict()
        assert s["mode"] == "bayes" and s["score"] == score and s["num_observed"] == 5
        a = int(out["proposal"].get_array("actions")[0])
        assert a not in actions.tolist(), "an observed candidate must not be proposed again"
        post = out["posterior"]
        assert post.get_array("fmean").shape == (121, 1) and post.get_array("fstd").shape == (121, 1)
        assert "best_action" in s["best_so_far"] and "best_X" in s["best_so_far"]
        assert len(s["posterior_at_proposal"]["fmean"]) == 1
    out = propose(grid, orm.Dict(dict={"seed": 3, "score": "TS", "num_rand_basis": 50, "num_search_each_probe": 2}), obs)
    assert len(out["proposal"].get_array("actions")) == 2
    out = propose(grid, orm.Dict(dict={"seed": 3, "maximize": False}), obs)
    assert out["summary"].get_dict()["best_so_far"]["best_value"] == pytest.approx(float(t.min()))
    with pytest.raises(ValueError, match="unknown propose parameters"):
        propose(grid, orm.Dict(dict={"nope": 1}), obs)
    with pytest.raises(ValueError, match="not available"):
        propose(grid, orm.Dict(dict={"score": "EHVI"}), obs)
    with pytest.raises(ValueError, match="range .*space only"):
        propose(grid, orm.Dict(dict={"optimizer": "odatse"}), obs)


def test_propose_discrete_multi_objective(grid):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose

    X = grid.X
    actions = np.array([0, 10, 60, 110, 120])
    t = np.stack([-np.sum((X[actions] - 0.5) ** 2, axis=1), -np.sum((X[actions] + 0.5) ** 2, axis=1)], axis=1)
    obs = observe(grid, _new(actions=actions, t=t))
    assert obs.num_objectives == 2
    for score in ("EHVI", "HVPI", "TS"):
        out = propose(grid, orm.Dict(dict={"seed": 1, "score": score, "posterior": True}), obs)
        s = out["summary"].get_dict()
        assert s["num_objectives"] == 2 and s["score"] == score
        assert "pareto_actions" in s["best_so_far"] and s["best_so_far"]["pareto_size"] >= 1
        assert out["posterior"].get_array("fmean").shape == (121, 2)


def test_propose_discrete_exhausted(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid, observe, propose

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0], "max": [1.0], "num": 3}))
    obs = observe(cand, _new(actions=[0, 1, 2], t=[1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="observed already"):
        propose(cand, orm.Dict(dict={}), obs)


# ------------------------------------------------------------------ propose: range
def test_propose_range_random_then_bayes(box):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose

    out = propose(box, orm.Dict(dict={"seed": 1, "num_search_each_probe": 5}))
    s = out["summary"].get_dict()
    prop = out["proposal"]
    assert s["mode"] == "random" and s["space"] == "range" and s["proposed_actions"] is None
    assert "actions" not in prop.get_arraynames() and prop.get_array("X").shape == (5, 2)
    X0 = prop.get_array("X")
    assert box.contains(X0).all()
    obs = observe(box, _new(X=X0, t=-np.sum(X0 ** 2, axis=1)))
    for score in ("TS", "EI", "PI"):
        out = propose(box, orm.Dict(dict={"seed": 2, "score": score, "optimizer_nsamples": 200}), obs)
        s = out["summary"].get_dict()
        X = out["proposal"].get_array("X")
        assert s["mode"] == "bayes" and s["optimizer"] == "random" and s["nsamples"] == 200 and X.shape == (1, 2)
        assert box.contains(X).all()
        assert "best_X" in s["best_so_far"] and "best_action" not in s["best_so_far"]
        assert len(s["posterior_at_proposal"]["fmean"]) == 1
    # several points per step with the exact GP; BLM + several points is refused (PHYSBO 3.2.1 limitation)
    out = propose(box, orm.Dict(dict={"seed": 2, "score": "EI", "num_search_each_probe": 3, "optimizer_nsamples": 100}), obs)
    assert out["proposal"].get_array("X").shape == (3, 2)
    out = propose(box, orm.Dict(dict={"seed": 2, "score": "TS", "num_rand_basis": 30, "optimizer_nsamples": 100}), obs)
    assert out["proposal"].get_array("X").shape == (1, 2)
    with pytest.raises(ValueError, match="several points per step"):
        propose(box, orm.Dict(dict={"num_search_each_probe": 2, "num_rand_basis": 30}), obs)
    with pytest.raises(ValueError, match="optimizer 'nope'"):
        propose(box, orm.Dict(dict={"optimizer": "nope"}), obs)
    with pytest.raises(ValueError, match="odatse_algorithm 'mapper'"):
        propose(box, orm.Dict(dict={"optimizer": "odatse", "odatse_algorithm": "mapper"}), obs)


def test_propose_range_odatse_optimizer(box, tmp_path, monkeypatch):
    from aiida import orm

    pytest.importorskip("odatse")
    from aiida_physbo.calcfunctions import observe, propose

    monkeypatch.chdir(tmp_path)
    X0 = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [0.9, 0.0]])
    obs = observe(box, _new(X=X0, t=-np.sum(X0 ** 2, axis=1)))
    out = propose(box, orm.Dict(dict={"seed": 1, "score": "EI", "optimizer": "odatse", "odatse_algorithm": "minsearch"}), obs)
    s = out["summary"].get_dict()
    X = out["proposal"].get_array("X")
    assert s["optimizer"] == "odatse" and s["algorithm"] == "minsearch" and X.shape == (1, 2)
    assert box.contains(X).all()
    assert not (tmp_path / "odatse_output").exists(), "ODAT-SE must run in a scratch directory, not the cwd"
    out = propose(box, orm.Dict(dict={"seed": 1, "score": "EI", "optimizer": "odatse", "odatse_algorithm": "exchange",
                                      "odatse_params": {"exchange": {"numsteps": 50}}}), obs)
    assert out["summary"].get_dict()["params"]["exchange"]["numsteps"] == 50


def test_propose_range_multi_objective(box):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose

    X0 = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    t = np.stack([-np.sum((X0 - 0.5) ** 2, axis=1), -np.sum((X0 + 0.5) ** 2, axis=1)], axis=1)
    obs = observe(box, _new(X=X0, t=t))
    for score in ("HVPI", "EHVI", "TS"):
        out = propose(box, orm.Dict(dict={"seed": 1, "score": score, "optimizer_nsamples": 100}), obs)
        s = out["summary"].get_dict()
        assert s["num_objectives"] == 2 and out["proposal"].get_array("X").shape == (1, 2)
        assert "pareto_X" in s["best_so_far"] and len(s["posterior_at_proposal"]["fmean"][0]) == 2


# ------------------------------------------------------------------ test functions / summary
def test_evaluate_test_function_and_objectives(grid, box):
    from aiida import orm

    from aiida_physbo import objectives
    from aiida_physbo.calcfunctions import evaluate_test_function, propose, summarize

    names = {r["name"] for r in objectives.available()}
    assert {"Sphere", "Rastrigin", "ZDT1"} <= names
    out = propose(grid, orm.Dict(dict={"seed": 1, "num_search_each_probe": 3}))
    ev = evaluate_test_function(grid, out["proposal"], orm.Dict(dict={"name": "Sphere", "kwargs": {"dim": 2}}))
    a = ev.get_array("actions")
    assert np.allclose(ev.get_array("t")[:, 0], np.sum(grid.X[a] ** 2, axis=1))      # f itself (minimization)
    assert ev.get_array("X").shape == (3, 2)
    ev2 = evaluate_test_function(grid, out["proposal"], orm.Dict(dict={"name": "Sphere", "maximize": True}))
    assert np.allclose(ev2.get_array("t"), -ev.get_array("t"))
    out = propose(box, orm.Dict(dict={"seed": 1, "num_search_each_probe": 3}))
    ev = evaluate_test_function(box, out["proposal"], orm.Dict(dict={"name": "Sphere"}))
    assert "actions" not in ev.get_arraynames() and ev.get_array("t").shape == (3, 1)
    with pytest.raises(ValueError, match="unknown test function"):
        objectives.make({"name": "NoSuchFunction"})
    from aiida_physbo.calcfunctions import observe

    obs = observe(box, ev)
    s = summarize(box, obs, orm.Dict(dict={"maximize": False})).get_dict()
    assert s["space"] == "range" and s["best"]["best_value"] == pytest.approx(float(obs.t.min())) and "best_X" in s["best"]


# ------------------------------------------------------------------ posterior on a grid (range) and the posterior query
def test_posterior_grid_range_and_query(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose, search_box
    from aiida_physbo.query.nodes import posterior

    box = search_box(orm.Dict(dict={"min": [0.0], "max": [1.0]}))
    X0 = np.array([[0.1], [0.5], [0.9]])
    obs = observe(box, _new(X=X0, t=(6 * X0[:, 0] - 2) ** 2 * np.sin(12 * X0[:, 0] - 4)))
    out = propose(box, orm.Dict(dict={"seed": 1, "score": "EI", "maximize": False, "posterior": True, "posterior_num": 51,
                                      "optimizer_nsamples": 200}), obs)
    post = out["posterior"]
    assert set(post.get_arraynames()) == {"X", "fmean", "fstd", "score"}
    assert post.get_array("X").shape == (51, 1) and post.get_array("fmean").shape == (51, 1) and post.get_array("score").shape == (51,)
    assert out["summary"].get_dict()["posterior_points"] == 51
    d = posterior(out["proposal"].creator.pk, max_points=20)
    assert d["dim"] == 1 and d["num_points"] == 51 and d["thinned_to"] == 20 and len(d["X"]) == 20
    assert d["observations"]["pk"] == obs.pk and len(d["observations"]["t"]) == 3
    assert d["proposal"]["X"] == out["proposal"].get_array("X").tolist() and d["proposal"]["actions"] is None
    assert d["summary"]["score"] == "EI" and d["summary"]["maximize"] is False
    assert posterior(post.pk)["pk"] == post.pk                      # the posterior node itself is accepted
    with pytest.raises(ValueError, match="stored no posterior"):
        out2 = propose(box, orm.Dict(dict={"seed": 1, "optimizer_nsamples": 50}), obs)
        posterior(out2["proposal"].pk)
    box3 = search_box(orm.Dict(dict={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}))
    obs3 = observe(box3, _new(X=[[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]], t=[1.0, 2.0]))
    with pytest.raises(ValueError, match="dimension 1 or 2"):
        propose(box3, orm.Dict(dict={"posterior": True, "optimizer_nsamples": 50}), obs3)


def test_posterior_discrete_has_X_and_score(grid):
    from aiida import orm

    from aiida_physbo.calcfunctions import observe, propose
    from aiida_physbo.plot import plot_cli

    obs = observe(grid, _new(actions=[0, 60, 120], t=[1.0, 3.0, 2.0]))
    out = propose(grid, orm.Dict(dict={"seed": 1, "score": "PI", "posterior": True}), obs)
    post = out["posterior"]
    assert post.get_array("X").shape == (121, 2) and post.get_array("score").shape == (121,)
    # the 2-D space is drawn as observed points, not as a posterior curve
    pytest.importorskip("matplotlib")
    files = plot_cli(out["proposal"].creator.pk, outdir=str(__import__("tempfile").mkdtemp()))["files"]
    assert any(f.endswith("_points.png") for f in files)


def test_plot_posterior_1d(aiida_profile_clean, tmp_path):
    from aiida import orm

    pytest.importorskip("matplotlib")
    from aiida_physbo.calcfunctions import candidates_from_grid, observe, propose
    from aiida_physbo.plot import plot_cli

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0], "max": [1.0], "num": 41}))
    obs = observe(cand, _new(actions=[0, 20, 40], t=[1.0, -2.0, 0.5]))
    out = propose(cand, orm.Dict(dict={"seed": 1, "score": "EI", "maximize": False, "posterior": True}), obs)
    r = plot_cli(out["proposal"].creator.pk, outdir=str(tmp_path), minimize=True)
    assert len(r["files"]) == 1 and r["files"][0].endswith("_posterior.png") and (tmp_path / r["files"][0].split("/")[-1]).exists()
    assert r["num_observations"] == 3 and r["proposed_X"] == out["proposal"].get_array("X").tolist()
