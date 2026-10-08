# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Read-only queries over the AiiDA database (aiida imported lazily)."""
import datetime
import os
import time

import numpy as np

OPTIMIZE_LABEL = "PhysboOptimizeWorkChain"
CALCFUNCTION_LABELS = ("candidates_from_file", "candidates_from_grid", "observe", "propose",
                       "evaluate_test_function", "summarize")
PROCESS_LABELS = CALCFUNCTION_LABELS + (OPTIMIZE_LABEL,)


def input_nodes(node) -> dict:
    """link label -> node of the inputs (INPUT_CALC / INPUT_WORK)."""
    from aiida.common.links import LinkType

    links = node.base.links.get_incoming(link_type=(LinkType.INPUT_CALC, LinkType.INPUT_WORK)).all()
    return {link.link_label: link.node for link in links}


def output_nodes(node) -> dict:
    """link label -> node of the outputs (CREATE / RETURN)."""
    from aiida.common.links import LinkType

    links = node.base.links.get_outgoing(link_type=(LinkType.CREATE, LinkType.RETURN)).all()
    return {link.link_label: link.node for link in links}


def _node(pk):
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    try:
        return orm.load_node(int(pk))
    except NotExistent as exc:
        raise ValueError(f"no node with pk {pk}") from exc


def _state(node):
    return node.process_state.value if getattr(node, "process_state", None) else None


def _creator(node):
    c = node.creator
    return {"pk": c.pk, "process_label": c.process_label, "label": c.label} if c is not None else None


def _rows(a: np.ndarray, max_rows=None):
    a = np.asarray(a)
    if max_rows is not None and a.shape[0] > max_rows:
        return a[:max_rows].tolist(), True
    return a.tolist(), False


# ---------------------------------------------------------------- environment
def status():
    from aiida import __version__ as aiida_version
    from aiida import orm
    from aiida.manage import get_manager

    import physbo

    from ..cli.control import daemon_status
    from ..data import CandidatesData, ObservationsData

    profile = get_manager().get_profile()
    info = {"profile": profile.name, "storage_backend": profile.storage_backend,
            "broker_backend": profile.process_control_backend, "aiida_version": aiida_version,
            "physbo_version": physbo.__version__}
    try:
        info["daemon"] = daemon_status()
    except Exception as exc:  # noqa: BLE001
        info["daemon"] = {"error": str(exc)}
    info["counts"] = {
        "candidates": orm.QueryBuilder().append(CandidatesData).count(),
        "observations": orm.QueryBuilder().append(ObservationsData).count(),
        "optimize_workchains": orm.QueryBuilder().append(
            orm.WorkChainNode, filters={"attributes.process_label": OPTIMIZE_LABEL}).count(),
    }
    return info


def daemon_status():
    from ..cli.control import daemon_status as _ds

    return _ds()


def test_functions():
    from .. import objectives

    return {"test_functions": objectives.available(),
            "note": "PHYSBO test functions are minimization problems; the WorkChain stores f and minimizes "
                    "unless the objective says maximize: true"}


# ---------------------------------------------------------------- processes
def process(pk):
    from aiida import orm

    node = _node(pk)
    if not isinstance(node, orm.ProcessNode):
        raise ValueError(f"Node<{node.pk}> is a {node.__class__.__name__}, not a process")
    d = {"pk": node.pk, "label": node.label, "process_label": node.process_label, "state": _state(node),
         "exit_status": node.exit_status, "exit_message": node.exit_message,
         "ctime": node.ctime.isoformat(timespec="seconds"),
         "inputs": {k: v.pk for k, v in input_nodes(node).items()},
         "outputs": {k: v.pk for k, v in output_nodes(node).items()}}
    logs = orm.Log.collection.get_logs_for(node)
    d["report_tail"] = [f"{log.levelname}: {log.message}" for log in logs[-5:]]
    if node.exception:
        d["exception_tail"] = node.exception.strip().splitlines()[-3:]
    if isinstance(node, orm.WorkflowNode):
        d["children"] = [{"pk": c.pk, "label": c.label, "process_label": c.process_label, "state": _state(c),
                          "exit_status": c.exit_status} for c in node.called][-10:]
        d["num_children"] = len(node.called)
    return d


def list_processes(label_prefix=None, days=None, state=None, limit=None):
    from aiida import orm

    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days or 7)
    filters = {"ctime": {">": since}, "attributes.process_label": {"in": list(PROCESS_LABELS)}}
    if label_prefix:
        filters["label"] = {"like": f"{label_prefix}%"}
    if state:
        filters["attributes.process_state"] = state
    qb = orm.QueryBuilder().append(orm.ProcessNode, filters=filters, tag="p")
    qb.order_by({"p": {"ctime": "desc"}}).limit(limit or 50)
    rows = [{"pk": n.pk, "label": n.label, "process_label": n.process_label, "state": _state(n),
             "exit_status": n.exit_status, "ctime": n.ctime.isoformat(timespec="seconds")} for n in qb.all(flat=True)]
    return {"count": len(rows), "processes": rows}


def wait(pk, wait_seconds=None):
    node = _node(pk)
    seconds = min(int(wait_seconds or 30), 45)
    t0 = time.time()
    while not node.is_terminated and time.time() - t0 < seconds:
        time.sleep(2)
    return {"pk": node.pk, "terminated": node.is_terminated, "state": _state(node),
            "exit_status": node.exit_status, "waited_seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- data
def candidates_info(pk, head=None):
    from ..data import CandidatesData

    node = _node(pk)
    if not isinstance(node, CandidatesData):
        raise ValueError(f"Node<{node.pk}> is a {node.__class__.__name__}, not CandidatesData")
    X = node.X
    head_rows, _ = _rows(X, int(head or 5))
    info = {"pk": node.pk, "label": node.label, "num_candidates": node.num_candidates, "dim": node.dim,
            "columns": node.columns, "min": X.min(axis=0).tolist(), "max": X.max(axis=0).tolist(),
            "head": head_rows, "creator": _creator(node)}
    if node.creator is not None:
        ins = input_nodes(node.creator)
        if "source" in ins:
            info["source_file"] = ins["source"].filename
        if "spec" in ins:
            info["grid_spec"] = ins["spec"].get_dict()
    return info


def _observations_of(node):
    """the ObservationsData meant by a pk: itself, the output of an optimize WorkChain, the input of a propose."""
    from aiida import orm

    from ..data import ObservationsData

    if isinstance(node, ObservationsData):
        return node
    if isinstance(node, orm.ProcessNode):
        outs = output_nodes(node)
        if "observations" in outs:
            return outs["observations"]
        ins = input_nodes(node)
        if "observations" in ins:
            return ins["observations"]
        if node.process_label == "observe":
            return outs.get("result")
    raise ValueError(f"Node<{node.pk}> ({node.__class__.__name__}) has no observations")


def _chain(obs):
    """the observe calcfunctions that built this observations node, oldest first."""
    chain = []
    node = obs
    while node is not None and node.creator is not None and node.creator.process_label == "observe":
        creator = node.creator
        ins = input_nodes(creator)
        new = ins.get("new")
        chain.append({"process_pk": creator.pk, "observations_pk": node.pk,
                      "num_new": int(new.get_array("actions").shape[0]) if new is not None else None,
                      "new_from": _creator(new) if new is not None else None})
        node = ins.get("observations")
    return list(reversed(chain))


def history(pk, minimize=False, max_rows=None):
    """observations in order, the best-so-far sequence (or Pareto front), and the observe chain."""
    from ..calcfunctions import best_of, best_sequence

    node = _node(pk)
    obs = _observations_of(node)
    maximize = not minimize
    actions, t = obs.actions, obs.t
    cap = int(max_rows or 500)
    a_rows, truncated = _rows(actions, cap)
    t_rows, _ = _rows(t, cap)
    d = {"pk": obs.pk, "label": obs.label, "num_observations": obs.num_observations,
         "num_objectives": obs.num_objectives, "maximize": maximize,
         "actions": a_rows, "t": t_rows, "truncated": truncated,
         "best": best_of(actions, t, maximize), "best_sequence": best_sequence(t, maximize)[:cap],
         "chain": _chain(obs)}
    creator = obs.creator
    if creator is not None:
        ins = input_nodes(creator)
        if "candidates" in ins:
            cand = ins["candidates"]
            d["candidates_pk"] = cand.pk
            if "best_action" in d["best"]:
                d["best"]["best_X"] = cand.X[d["best"]["best_action"]].tolist()
    return d


def proposal(pk):
    """actions / X of a proposal (the ArrayData or the propose process), with the summary and posterior stats."""
    from aiida import orm

    node = _node(pk)
    if isinstance(node, orm.ProcessNode):
        proc = node
        outs = output_nodes(proc)
        prop = outs.get("proposal")
        if prop is None:
            raise ValueError(f"Node<{node.pk}> is not a propose process (no proposal output)")
    else:
        prop, proc = node, node.creator
        outs = output_nodes(proc) if proc is not None else {}
    d = {"pk": prop.pk, "process_pk": proc.pk if proc is not None else None,
         "actions": prop.get_array("actions").tolist(), "X": prop.get_array("X").tolist()}
    if "summary" in outs:
        d["summary"] = outs["summary"].get_dict()
    if "posterior" in outs:
        post = outs["posterior"]
        fmean, fstd = post.get_array("fmean"), post.get_array("fstd")
        d["posterior"] = {"pk": post.pk, "shape": list(fmean.shape),
                          "fmean_at_proposal": fmean[d["actions"]].tolist(), "fstd_at_proposal": fstd[d["actions"]].tolist(),
                          "fmean_argmax": int(np.argmax(fmean[:, 0])), "fmean_max": float(fmean[:, 0].max()),
                          "fstd_max": float(fstd.max())}
    if proc is not None:
        ins = input_nodes(proc)
        d["inputs"] = {k: v.pk for k, v in ins.items()}
        if "observations" not in ins:
            d["note"] = "no observations were given: the proposal is random"
    return d


def results(pk):
    """summary of an optimize WorkChain (or of a propose / observe process)."""
    from aiida import orm

    node = _node(pk)
    if not isinstance(node, orm.ProcessNode):
        raise ValueError(f"Node<{node.pk}> is not a process; use history / proposal / candidates-info for data nodes")
    d = {"pk": node.pk, "label": node.label, "process_label": node.process_label, "state": _state(node),
         "exit_status": node.exit_status, "exit_message": node.exit_message}
    outs = output_nodes(node)
    if node.process_label == OPTIMIZE_LABEL:
        d["num_steps_done"] = sum(1 for c in node.called if c.process_label == "observe")
        d["inputs"] = {k: v.pk for k, v in input_nodes(node).items()}
        if "summary" in outs:
            d["summary"] = outs["summary"].get_dict()
        if "observations" in outs:
            d["observations_pk"] = outs["observations"].pk
        elif node.is_terminated:
            d["note"] = "no observations output (failed before the first step?)"
        else:
            last = [c for c in node.called if c.process_label == "observe"]
            if last:
                d["observations_pk_so_far"] = output_nodes(last[-1])["result"].pk
    elif node.process_label == "propose":
        d["proposal"] = proposal(node.pk)
    elif node.process_label == "observe" and "result" in outs:
        d["observations_pk"] = outs["result"].pk
        d["num_observations"] = outs["result"].num_observations
    elif node.process_label == "summarize" and "result" in outs:
        d["summary"] = outs["result"].get_dict()
    else:
        d["outputs"] = {k: v.pk for k, v in outs.items()}
    return d


def provenance(pk, outdir=None, ancestor_depth=None, descendant_depth=None, fmt=None):
    from aiida.tools.visualization.graph import Graph

    node = _node(pk)
    outdir = outdir or os.path.join(os.path.expanduser("~"), "aiida_work", "figures", str(node.pk))
    os.makedirs(outdir, exist_ok=True)
    graph = Graph(graph_attr={"rankdir": "TB"})
    graph.recurse_ancestors(node, depth=ancestor_depth or 5, annotate_links="both", include_process_outputs=True)
    graph.recurse_descendants(node, depth=descendant_depth or 3, annotate_links="both", include_process_inputs=True)
    base = os.path.join(outdir, f"provenance_{node.pk}")
    path = graph.graphviz.render(base, format=fmt or "png", cleanup=True)
    return {"pk": node.pk, "file": path, "nodes": len(graph.nodes), "edges": len(graph.edges)}
