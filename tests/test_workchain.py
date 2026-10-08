"""Layer-2: the closed-loop WorkChain on a temporary sqlite profile (runs in-process with run_get_node)."""
import pytest

pytest.importorskip("aiida")
pytest.importorskip("physbo")


def test_optimize_workchain_sphere(aiida_profile_clean):
    from aiida import orm
    from aiida.engine import run_get_node
    from aiida.plugins import WorkflowFactory

    from aiida_physbo.calcfunctions import candidates_from_grid

    cand = candidates_from_grid(orm.Dict(dict={"min": [-1.0, -1.0], "max": [1.0, 1.0], "num": 11}))
    WorkChain = WorkflowFactory("physbo.optimize")
    result, node = run_get_node(WorkChain, candidates=cand, objective=orm.Dict(dict={"name": "Sphere"}),
                                parameters=orm.Dict(dict={"seed": 1, "score": "EI"}),
                                num_random=orm.Int(5), num_bayes=orm.Int(3), label=orm.Str("t"))
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    obs = result["observations"]
    assert obs.num_observations == 8
    s = result["summary"].get_dict()
    assert s["steps"] == 4 and s["maximize"] is False
    assert s["best"]["best_value"] == pytest.approx(min(obs.t[:, 0])) and len(s["best_sequence"]) == 8
    assert s["best"]["best_X"] == pytest.approx(cand.X[s["best"]["best_action"]].tolist())
    labels = [c.process_label for c in node.called]
    assert labels.count("propose") == 4 and labels.count("observe") == 4 and labels.count("summarize") == 1
    assert result["summary"].creator.process_label == "summarize"
    # queries used by the CLI
    from aiida_physbo.query.nodes import history, list_processes, process, results

    r = results(node.pk)
    assert r["observations_pk"] == obs.pk and r["num_steps_done"] == 4
    h = history(node.pk, minimize=True)
    assert h["num_observations"] == 8 and len(h["chain"]) == 4 and h["best"]["best_value"] == pytest.approx(s["best"]["best_value"])
    assert process(node.pk)["num_children"] == 13
    assert any(p["pk"] == node.pk for p in list_processes()["processes"])


def test_optimize_workchain_bad_objective(aiida_profile_clean):
    from aiida import orm
    from aiida.engine import run_get_node
    from aiida.plugins import WorkflowFactory

    from aiida_physbo.calcfunctions import candidates_from_grid

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0], "max": [1.0], "num": 5}))
    _, node = run_get_node(WorkflowFactory("physbo.optimize"), candidates=cand,
                           objective=orm.Dict(dict={"name": "Nope"}), num_random=orm.Int(2), num_bayes=orm.Int(1))
    assert node.exit_status == 410
    _, node = run_get_node(WorkflowFactory("physbo.optimize"), candidates=cand,
                           objective=orm.Dict(dict={"name": "Sphere", "kwargs": {"dim": 1}}),
                           num_random=orm.Int(3), num_bayes=orm.Int(5))
    assert node.exit_status == 420      # 5 candidates, 3 random + 5 bayes -> runs out


def test_optimize_workchain_multi_objective(aiida_profile_clean):
    from aiida import orm
    from aiida.engine import run_get_node
    from aiida.plugins import WorkflowFactory

    from aiida_physbo.calcfunctions import candidates_from_grid

    cand = candidates_from_grid(orm.Dict(dict={"min": [0.0, 0.0], "max": [1.0, 1.0], "num": 8}))
    result, node = run_get_node(WorkflowFactory("physbo.optimize"), candidates=cand,
                                objective=orm.Dict(dict={"name": "ZDT1", "kwargs": {"dim": 2}}),
                                parameters=orm.Dict(dict={"seed": 1, "score": "HVPI"}),
                                num_random=orm.Int(4), num_bayes=orm.Int(2))
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    s = result["summary"].get_dict()
    assert result["observations"].num_objectives == 2 and s["best"]["pareto_size"] >= 1 and "pareto_X" in s["best"]
