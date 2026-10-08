"""Layer-2: the closed-loop WorkChain on a temporary sqlite profile (runs in-process with run_get_node)."""
import pytest

pytest.importorskip("aiida")
pytest.importorskip("physbo")


def _run(space, objective, params=None, num_random=5, num_bayes=3, label="t"):
    from aiida import orm
    from aiida.engine import run_get_node
    from aiida.plugins import WorkflowFactory

    return run_get_node(WorkflowFactory("physbo.optimize"), space=space, objective=orm.Dict(dict=objective),
                        parameters=orm.Dict(dict=params or {}), num_random=orm.Int(num_random),
                        num_bayes=orm.Int(num_bayes), label=orm.Str(label))


def test_optimize_workchain_sphere_discrete(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid

    cand = candidates_from_grid(orm.Dict(dict={"min": [-1.0, -1.0], "max": [1.0, 1.0], "num": 11}))
    result, node = _run(cand, {"name": "Sphere"}, {"seed": 1, "score": "EI"})
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    obs = result["observations"]
    assert obs.num_observations == 8 and obs.space == "discrete" and obs.X.shape == (8, 2)
    s = result["summary"].get_dict()
    assert s["steps"] == 4 and s["maximize"] is False and s["space"] == "discrete"
    assert s["best"]["best_value"] == pytest.approx(min(obs.t[:, 0])) and len(s["best_sequence"]) == 8
    assert s["best"]["best_X"] == pytest.approx(cand.X[s["best"]["best_action"]].tolist())
    labels = [c.process_label for c in node.called]
    assert labels.count("propose") == 4 and labels.count("observe") == 4 and labels.count("summarize") == 1
    assert result["summary"].creator.process_label == "summarize"
    from aiida_physbo.query.nodes import history, list_processes, process, results, space_info

    r = results(node.pk)
    assert r["observations_pk"] == obs.pk and r["num_steps_done"] == 4
    h = history(node.pk, minimize=True)
    assert h["num_observations"] == 8 and len(h["chain"]) == 4 and h["space_pk"] == cand.pk
    assert h["best"]["best_value"] == pytest.approx(s["best"]["best_value"]) and len(h["X"]) == 8
    assert process(node.pk)["num_children"] == 13
    assert any(p["pk"] == node.pk for p in list_processes()["processes"])
    assert space_info(cand.pk)["space"] == "discrete"


def test_optimize_workchain_sphere_range(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import search_box
    from aiida_physbo.query.nodes import history, proposal, space_info

    box = search_box(orm.Dict(dict={"min": [-1.0, -1.0], "max": [1.0, 1.0]}))
    result, node = _run(box, {"name": "Sphere"}, {"seed": 1, "score": "EI", "optimizer_nsamples": 200})
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    obs = result["observations"]
    assert obs.space == "range" and obs.actions is None and obs.X.shape == (8, 2)
    assert box.contains(obs.X).all()
    s = result["summary"].get_dict()
    assert s["space"] == "range" and "best_X" in s["best"] and "best_action" not in s["best"]
    h = history(node.pk, minimize=True)
    assert h["actions"] is None and len(h["X"]) == 8 and h["space_pk"] == box.pk
    first_propose = [c for c in node.called if c.process_label == "propose"][0]
    assert proposal(first_propose.pk)["actions"] is None
    assert space_info(box.pk)["space"] == "range"


def test_optimize_workchain_errors(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid, search_box

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0], "max": [1.0], "num": 5}))
    _, node = _run(cand, {"name": "Nope"}, num_random=2, num_bayes=1)
    assert node.exit_status == 410
    _, node = _run(cand, {"name": "Sphere", "kwargs": {"dim": 1}}, num_random=3, num_bayes=5)
    assert node.exit_status == 420      # 5 candidates, 3 random + 5 bayes -> runs out
    _, node = _run(cand, {"name": "Sphere", "kwargs": {"dim": 1}}, {"optimizer": "odatse"}, num_random=2, num_bayes=1)
    assert node.exit_status == 411      # range-only parameter on a discrete space
    box = search_box(orm.Dict(dict={"min": [0.0], "max": [1.0]}))
    _, node = _run(box, {"name": "Sphere", "kwargs": {"dim": 1}}, {"optimizer_nsamples": 50}, num_random=3, num_bayes=4)
    assert node.is_finished_ok, (node.exit_status, node.exit_message)   # a box never runs out


def test_optimize_workchain_multi_objective(aiida_profile_clean):
    from aiida import orm

    from aiida_physbo.calcfunctions import candidates_from_grid

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0, 0.0], "max": [1.0, 1.0], "num": 8}))
    result, node = _run(cand, {"name": "ZDT1", "kwargs": {"dim": 2}}, {"seed": 1, "score": "HVPI"}, num_random=4, num_bayes=2)
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    s = result["summary"].get_dict()
    assert result["observations"].num_objectives == 2 and s["best"]["pareto_size"] >= 1 and "pareto_X" in s["best"]
