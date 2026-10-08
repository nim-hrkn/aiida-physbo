# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Read-only queries over the AiiDA database (aiida imported lazily)."""
import datetime
import os
import time

import numpy as np

OPTIMIZE_LABEL = "PhysboOptimizeWorkChain"
CALCFUNCTION_LABELS = ("candidates_from_file", "candidates_from_grid", "search_box", "observe", "propose",
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

    from .. import __version__
    from ..cli.control import daemon_status
    from ..data import CandidatesData, ObservationsData, SearchBoxData

    profile = get_manager().get_profile()
    info = {"profile": profile.name, "storage_backend": profile.storage_backend,
            "broker_backend": profile.process_control_backend, "aiida_version": aiida_version,
            "physbo_version": physbo.__version__, "aiida_physbo_version": __version__}
    try:
        info["daemon"] = daemon_status()
    except Exception as exc:  # noqa: BLE001
        info["daemon"] = {"error": str(exc)}
    info["counts"] = {
        "candidates": orm.QueryBuilder().append(CandidatesData).count(),
        "search_boxes": orm.QueryBuilder().append(SearchBoxData).count(),
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
def _space_input(process):
    """the space node (link `space`, or `candidates` in 0.1.0 graphs) of a process, or None."""
    ins = input_nodes(process)
    return ins.get("space") or ins.get("candidates")


def space_info(pk, head=None):
    """shape / box, feature names and (discrete) the first rows of a CandidatesData or SearchBoxData."""
    from ..data import CandidatesData, SearchBoxData

    node = _node(pk)
    if isinstance(node, CandidatesData):
        X = node.X
        head_rows, _ = _rows(X, int(head or 5))
        info = {"pk": node.pk, "label": node.label, "space": "discrete", "num_candidates": node.num_candidates,
                "dim": node.dim, "columns": node.columns, "min": X.min(axis=0).tolist(), "max": X.max(axis=0).tolist(),
                "head": head_rows}
    elif isinstance(node, SearchBoxData):
        info = {"pk": node.pk, "label": node.label, "space": "range", "dim": node.dim, "columns": node.columns,
                "min": node.min_X.tolist(), "max": node.max_X.tolist()}
    else:
        raise ValueError(f"Node<{node.pk}> is a {node.__class__.__name__}, not CandidatesData or SearchBoxData")
    info["creator"] = _creator(node)
    if node.creator is not None:
        ins = input_nodes(node.creator)
        if "source" in ins:
            info["source_file"] = ins["source"].filename
        if "spec" in ins:
            info["spec"] = ins["spec"].get_dict()
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
                      "num_new": int(new.get_array("t").shape[0]) if new is not None else None,
                      "new_from": _creator(new) if new is not None else None})
        node = ins.get("observations")
    return list(reversed(chain))


def history(pk, minimize=False, max_rows=None):
    """observations in order, the best-so-far sequence (or Pareto front), and the observe chain."""
    from ..calcfunctions import best_of, best_sequence

    node = _node(pk)
    obs = _observations_of(node)
    maximize = not minimize
    actions, X, t = obs.actions, obs.X, obs.t
    space = _space_input(obs.creator) if obs.creator is not None else None
    if X is None and actions is not None and space is not None:          # 0.1.0 node
        X = space.X[actions]
    cap = int(max_rows or 500)
    t_rows, truncated = _rows(t, cap)
    d = {"pk": obs.pk, "label": obs.label, "space": obs.space, "num_observations": obs.num_observations,
         "num_objectives": obs.num_objectives, "dim": int(X.shape[1]) if X is not None else None, "maximize": maximize,
         "actions": _rows(actions, cap)[0] if actions is not None else None,
         "X": _rows(X, cap)[0] if X is not None else None, "t": t_rows, "truncated": truncated,
         "best": best_of(t, maximize, actions=actions, X=X), "best_sequence": best_sequence(t, maximize)[:cap],
         "chain": _chain(obs)}
    if space is not None:
        d["space_pk"] = space.pk
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
    names = set(prop.get_arraynames())
    d = {"pk": prop.pk, "process_pk": proc.pk if proc is not None else None,
         "actions": prop.get_array("actions").tolist() if "actions" in names else None, "X": prop.get_array("X").tolist()}
    if "summary" in outs:
        d["summary"] = outs["summary"].get_dict()
    if "posterior" in outs:
        post = outs["posterior"]
        fmean, fstd = post.get_array("fmean"), post.get_array("fstd")
        d["posterior"] = {"pk": post.pk, "shape": list(fmean.shape), "on_grid": "X" in post.get_arraynames(),
                          "fmean_argmax": int(np.argmax(fmean[:, 0])), "fmean_max": float(fmean[:, 0].max()),
                          "fstd_max": float(fstd.max())}
        if d["actions"] is not None:
            d["posterior"]["fmean_at_proposal"] = fmean[d["actions"]].tolist()
            d["posterior"]["fstd_at_proposal"] = fstd[d["actions"]].tolist()
    if proc is not None:
        ins = input_nodes(proc)
        d["inputs"] = {k: v.pk for k, v in ins.items()}
        if "observations" not in ins:
            d["note"] = "no observations were given: the proposal is random"
    return d


def _posterior_of(node):
    """(propose process, posterior ArrayData) meant by a pk: a propose process, its proposal or its posterior node."""
    from aiida import orm

    if isinstance(node, orm.ProcessNode):
        proc = node
    else:
        proc = node.creator
    if proc is None or proc.process_label != "propose":
        raise ValueError(f"Node<{node.pk}> is not a propose process or one of its outputs")
    outs = output_nodes(proc)
    if "posterior" not in outs:
        raise ValueError(f"propose<{proc.pk}> stored no posterior; call propose with posterior=true")
    return proc, outs["posterior"]


def _thin(n, max_points):
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, int(max_points)).astype(int))


def posterior(pk, max_points=None):
    """posterior mean / std and the acquisition on the candidates (discrete) or the grid (range) of a propose step,
    with the observations it saw and the proposed points: everything needed to draw the step."""
    node = _node(pk)
    proc, post = _posterior_of(node)
    names = set(post.get_arraynames())
    outs = output_nodes(proc)
    ins = input_nodes(proc)
    Xg = post.get_array("X") if "X" in names else None
    if Xg is None:                                     # 0.2.0 node: candidates only
        space = _space_input(proc)
        Xg = space.X
    idx = _thin(Xg.shape[0], max_points or 1001)
    d = {"pk": post.pk, "process_pk": proc.pk, "dim": int(Xg.shape[1]), "num_points": int(Xg.shape[0]),
         "thinned_to": int(idx.size), "X": Xg[idx].tolist(), "fmean": post.get_array("fmean")[idx].tolist(),
         "fstd": post.get_array("fstd")[idx].tolist(),
         "score": post.get_array("score")[idx].tolist() if "score" in names else None}
    if "summary" in outs:
        summ = outs["summary"].get_dict()
        d["summary"] = {k: summ.get(k) for k in ("space", "mode", "score", "maximize", "num_observed", "proposed_actions", "proposed_X")}
    prop = outs.get("proposal")
    if prop is not None:
        d["proposal"] = {"X": prop.get_array("X").tolist(),
                         "actions": prop.get_array("actions").tolist() if "actions" in prop.get_arraynames() else None}
    obs = ins.get("observations")
    if obs is not None:
        space = _space_input(proc)
        Xo = obs.X if obs.X is not None else (space.X[obs.actions] if obs.actions is not None else None)
        d["observations"] = {"pk": obs.pk, "X": Xo.tolist() if Xo is not None else None, "t": obs.t.tolist()}
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
            try:
                from .. import objectives

                obj = d["summary"].get("objective") or {}
                km = objectives.known_minimum(obj) if obj else None
                if km is not None and "best_value" in d["summary"].get("best", {}):
                    f_star = km[0] if not obj.get("maximize") else -km[0]
                    best = d["summary"]["best"]["best_value"]
                    d["known_minimum"] = {"f_star": km[0], "points": km[1],
                                          "regret": abs(best - f_star), "noise": obj.get("noise") or 0.0}
            except Exception:  # noqa: BLE001  the comparison is a convenience
                pass
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
