# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""The single table of CLI subcommands.

Read by `cli/main.py` (argparse + dispatch) and by `mcp/server.py` (allow-list and argv of each tool).
No aiida import.

Each entry: kind (READ / SUBMIT / CONTROL), impl ("module:function", imported lazily), help, and the
options: name -> (type, required, help). Names use underscores here and become `--with-dashes` on the
command line. SUBMIT means "creates nodes" (candidates / observe / propose run calcfunctions in this
process; submit-optimize hands a WorkChain to the daemon).
"""
READ, SUBMIT, CONTROL = "read", "submit", "control"

_PK = ("int", True, "pk of the node")
_LABEL = ("str", False, "label of the created node(s)")
_SCORE = ("str", False, "acquisition: TS (default) | EI | PI for one objective; TS | EHVI | HVPI for several")
_NRB = ("int", False, "random feature basis size (default 0 = exact Gaussian process; e.g. 500 for large candidate sets)")
_NSEARCH = ("int", False, "number of candidates to propose per step (default 1)")
_SEED = ("int", False, "numpy random seed")
_MINIMIZE = ("bool", False, "the objective values are to be minimized (default: maximized, as PHYSBO does)")

SUBCOMMANDS = {
    # ---- read
    "status": dict(kind=READ, impl="aiida_physbo.query.nodes:status",
                   help="profile, daemon, versions and the number of candidates / observations / optimize runs", options={}),
    "daemon-status": dict(kind=READ, impl="aiida_physbo.query.nodes:daemon_status", help="daemon workers", options={}),
    "test-functions": dict(kind=READ, impl="aiida_physbo.query.nodes:test_functions",
                           help="PHYSBO test functions usable as objective of submit-optimize", options={}),
    "process": dict(kind=READ, impl="aiida_physbo.query.nodes:process",
                    help="state, exit code, in/outputs, last reports (and children) of a process", options={"pk": _PK}),
    "list": dict(kind=READ, impl="aiida_physbo.query.nodes:list_processes",
                 help="recent physbo processes (propose / observe / candidates / optimize)",
                 options={"label_prefix": ("str", False, "only labels starting with this"),
                          "days": ("int", False, "created within the last N days (default 7)"),
                          "state": ("str", False, "created|waiting|running|finished|excepted|killed"),
                          "limit": ("int", False, "max rows (default 50)")}),
    "wait": dict(kind=READ, impl="aiida_physbo.query.nodes:wait",
                 help="wait (at most 45 s) for a process to terminate",
                 options={"pk": _PK, "wait_seconds": ("int", False, "seconds to wait, max 45 (default 30)")}),
    "candidates-info": dict(kind=READ, impl="aiida_physbo.query.nodes:candidates_info",
                            help="shape, columns, ranges and the first rows of a CandidatesData",
                            options={"pk": _PK, "head": ("int", False, "rows to show (default 5)")}),
    "history": dict(kind=READ, impl="aiida_physbo.query.nodes:history",
                    help="observations in order, best-so-far (or Pareto front) and the observe chain; pk of an "
                         "ObservationsData, an optimize WorkChain, or a propose/observe process",
                    options={"pk": _PK, "minimize": _MINIMIZE, "max_rows": ("int", False, "rows returned (default 500)")}),
    "proposal": dict(kind=READ, impl="aiida_physbo.query.nodes:proposal",
                     help="actions and X of a proposal (pk of the ArrayData or of the propose process), summary, posterior",
                     options={"pk": _PK}),
    "results": dict(kind=READ, impl="aiida_physbo.query.nodes:results",
                    help="summary of an optimize WorkChain (best, best sequence, observations pk) or of a propose/observe",
                    options={"pk": _PK}),
    "plot": dict(kind=READ, impl="aiida_physbo.plot:plot_cli",
                 help="PNG of best-so-far vs step (one objective) or the Pareto front (several); pk as in history",
                 options={"pk": _PK, "minimize": _MINIMIZE,
                          "outdir": ("str", False, "output directory (default ~/aiida_work/figures/<pk>)"),
                          "prefix": ("str", False, "file prefix (default: label)")}),
    "provenance": dict(kind=READ, impl="aiida_physbo.query.nodes:provenance",
                       help="draw the provenance graph with graphviz",
                       options={"pk": _PK, "outdir": ("str", False, "output directory"),
                                "ancestor_depth": ("int", False, "default 5"),
                                "descendant_depth": ("int", False, "default 3"),
                                "fmt": ("str", False, "png|pdf|svg (default png)")}),
    # ---- submit (create nodes)
    "candidates": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:candidates",
                       help="store the candidate set X: from a csv/npy/npz file, a grid spec, or the box of a test function",
                       options={"file": ("str", False, "csv / tsv / npy / npz file with one candidate per row"),
                                "format": ("str", False, "csv|tsv|npy|npz (default: from the extension)"),
                                "delimiter": ("str", False, "csv delimiter (default ,)"),
                                "skip_header": ("int", False, "header lines to skip (csv)"),
                                "columns": ("str", False, "comma separated column indices to use as X (csv; default all)"),
                                "grid": ("str", False, 'JSON {"min": [...], "max": [...], "num": int|[...]} for a regular grid'),
                                "test_function": ("str", False, "grid over the default box of this PHYSBO test function"),
                                "kwargs": ("str", False, "JSON kwargs of the test function (e.g. {\"dim\": 3})"),
                                "num": ("int", False, "grid points per dimension with --test-function (default 21)"),
                                "names": ("str", False, "comma separated feature names"),
                                "label": _LABEL}),
    "observe": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:observe",
                    help="record measured values of candidates (appends to an observations chain)",
                    options={"candidates_pk": ("int", True, "pk of the CandidatesData"),
                             "observations_pk": ("int", False, "pk of the previous ObservationsData (omit to start a chain)"),
                             "actions": ("str", False, "comma separated candidate indices, e.g. 3,17,42"),
                             "values": ("str", False, "comma separated values (one objective) or JSON [[...],[...]] (several)"),
                             "file": ("str", False, "csv with action in the first column and the objective values after it"),
                             "label": _LABEL}),
    "propose": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:propose",
                    help="PHYSBO proposes the next candidates from the observations so far (random when there are none)",
                    options={"candidates_pk": ("int", True, "pk of the CandidatesData"),
                             "observations_pk": ("int", False, "pk of the ObservationsData (omit for random proposals)"),
                             "score": _SCORE, "num_rand_basis": _NRB, "num_search_each_probe": _NSEARCH, "seed": _SEED,
                             "num_objectives": ("int", False, "number of objectives (default: from the observations)"),
                             "minimize": _MINIMIZE,
                             "random": ("bool", False, "random proposals instead of Bayesian ones"),
                             "posterior": ("bool", False, "also store the posterior mean / std on every candidate"),
                             "interval": ("int", False, "hyperparameter learning interval (default 0 = once)"),
                             "label": _LABEL}),
    "submit-optimize": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:submit_optimize",
                            help="submit a closed-loop WorkChain on a PHYSBO test function (daemon): random -> Bayesian steps",
                            options={"test_function": ("str", True, "objective: a PHYSBO test function name (see test-functions)"),
                                     "kwargs": ("str", False, "JSON kwargs of the test function"),
                                     "maximize": ("bool", False, "store and maximize -f instead of minimizing f"),
                                     "candidates_pk": ("int", False, "pk of the CandidatesData (default: a grid over the function box)"),
                                     "num": ("int", False, "grid points per dimension when no candidates_pk (default 21)"),
                                     "observations_pk": ("int", False, "observations to start from"),
                                     "num_random": ("int", False, "random evaluations first (default 10)"),
                                     "num_bayes": ("int", False, "Bayesian steps (default 20)"),
                                     "score": _SCORE, "num_rand_basis": _NRB, "num_search_each_probe": _NSEARCH, "seed": _SEED,
                                     "label": _LABEL}),
    # ---- control
    "daemon-start": dict(kind=CONTROL, impl="aiida_physbo.cli.control:daemon_start",
                         help="start the daemon", options={"workers": ("int", False, "number of workers (default 1)")}),
    "daemon-stop": dict(kind=CONTROL, impl="aiida_physbo.cli.control:daemon_stop", help="stop the daemon", options={}),
    "kill": dict(kind=CONTROL, impl="aiida_physbo.cli.control:kill", help="kill a running process", options={"pk": _PK}),
}

KINDS = (READ, SUBMIT, CONTROL)


def subcommands_of_kind(kind: str) -> list:
    return [name for name, s in SUBCOMMANDS.items() if s["kind"] == kind]


def option_flag(name: str) -> str:
    return "--" + name.replace("_", "-")
