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
_SPACE_PK = ("int", True, "pk of the search space: CandidatesData (discrete) or SearchBoxData (range)")
_TEST_FN = ("str", False, "PHYSBO test function whose default box defines the space (see test-functions)")
_KWARGS = ("str", False, "JSON kwargs of the test function (e.g. {\"dim\": 3})")
_NAMES = ("str", False, "comma separated feature names")
# range space only: how the acquisition function is maximized over the box
_RANGE_OPTS = {"optimizer": ("str", False, "range space: random (default, uniform samples) | odatse (ODAT-SE algorithm)"),
               "optimizer_nsamples": ("int", False, "range space, optimizer random: number of samples (default 1000)"),
               "odatse_algorithm": ("str", False, "range space, optimizer odatse: exchange (default) | pamc | minsearch | bayes"),
               "odatse_params": ("str", False, "range space, optimizer odatse: JSON merged over PHYSBO's default_alg_dict")}

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
    "space-info": dict(kind=READ, impl="aiida_physbo.query.nodes:space_info",
                       help="a search space: CandidatesData (shape, ranges, first rows) or SearchBoxData (box)",
                       options={"pk": _PK, "head": ("int", False, "candidate rows to show (default 5)")}),
    "history": dict(kind=READ, impl="aiida_physbo.query.nodes:history",
                    help="observations in order, best-so-far (or Pareto front) and the observe chain; pk of an "
                         "ObservationsData, an optimize WorkChain, or a propose/observe process",
                    options={"pk": _PK, "minimize": _MINIMIZE, "max_rows": ("int", False, "rows returned (default 500)")}),
    "proposal": dict(kind=READ, impl="aiida_physbo.query.nodes:proposal",
                     help="actions and X of a proposal (pk of the ArrayData or of the propose process), summary, posterior",
                     options={"pk": _PK}),
    "posterior": dict(kind=READ, impl="aiida_physbo.query.nodes:posterior",
                      help="posterior mean / std and acquisition of a propose step (run with --posterior) on the candidates or "
                           "the grid, with the observations it saw and the proposed points: the data of a figure",
                      options={"pk": ("int", True, "pk of the propose process, its proposal or its posterior node"),
                               "max_points": ("int", False, "thin the arrays to at most this many points (default 1001)")}),
    "results": dict(kind=READ, impl="aiida_physbo.query.nodes:results",
                    help="summary of an optimize WorkChain (best, best sequence, observations pk) or of a propose/observe",
                    options={"pk": _PK}),
    "plot": dict(kind=READ, impl="aiida_physbo.plot:plot_cli",
                 help="PNGs: best-so-far vs step, observed points (dim 2), Pareto front (several objectives); for a propose "
                      "step run with --posterior on a 1-D space also the posterior mean / band, acquisition and proposal",
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
                                "test_function": _TEST_FN, "kwargs": _KWARGS,
                                "num": ("int", False, "grid points per dimension with --test-function (default 21)"),
                                "names": _NAMES, "label": _LABEL}),
    "search-box": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:search_box",
                       help="store a continuous search space (box) for the PHYSBO range policies: proposals are coordinates",
                       options={"min": ("str", False, "comma separated lower bounds, e.g. -2,-2"),
                                "max": ("str", False, "comma separated upper bounds, e.g. 2,2"),
                                "test_function": _TEST_FN, "kwargs": _KWARGS, "names": _NAMES, "label": _LABEL}),
    "observe": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:observe",
                    help="record measured values (appends to an observations chain): actions for a discrete space, x for a range space",
                    options={"space_pk": _SPACE_PK,
                             "observations_pk": ("int", False, "pk of the previous ObservationsData (omit to start a chain)"),
                             "actions": ("str", False, "discrete: comma separated candidate indices, e.g. 3,17,42"),
                             "x": ("str", False, "range: coordinates as JSON rows [[x1,x2],...] or 'x1,x2;x1,x2'"),
                             "values": ("str", False, "comma separated values (one objective) or JSON [[...],[...]] (several)"),
                             "file": ("str", False, "csv: discrete -> action then values per row; range -> the d coordinates then values"),
                             "label": _LABEL}),
    "propose": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:propose",
                    help="PHYSBO proposes the next points from the observations so far (random when there are none)",
                    options={"space_pk": _SPACE_PK,
                             "observations_pk": ("int", False, "pk of the ObservationsData (omit for random proposals)"),
                             "score": _SCORE, "num_rand_basis": _NRB, "num_search_each_probe": _NSEARCH, "seed": _SEED,
                             "num_objectives": ("int", False, "number of objectives (default: from the observations)"),
                             "minimize": _MINIMIZE,
                             "random": ("bool", False, "random proposals instead of Bayesian ones"),
                             "posterior": ("bool", False, "also store posterior mean / std and acquisition on every candidate "
                                                          "(discrete) or on a grid over the box (range, dim <= 2); read with `posterior`, draw with `plot`"),
                             "posterior_num": ("int", False, "range + --posterior: grid points per dimension (default 101)"),
                             "interval": ("int", False, "hyperparameter learning interval (default 0 = once)"),
                             **_RANGE_OPTS, "label": _LABEL}),
    "submit-optimize": dict(kind=SUBMIT, impl="aiida_physbo.cli.steps:submit_optimize",
                            help="submit a closed-loop WorkChain on a PHYSBO test function (daemon): random -> Bayesian steps",
                            options={"test_function": ("str", True, "objective: a PHYSBO test function name (see test-functions)"),
                                     "kwargs": _KWARGS,
                                     "maximize": ("bool", False, "store and maximize -f instead of minimizing f"),
                                     "noise": ("float", False, "Gaussian observation noise (std) added to every evaluation (default 0)"),
                                     "noise_seed": ("int", False, "seed of the observation noise (default 0)"),
                                     "space_pk": ("int", False, "pk of the CandidatesData / SearchBoxData (default: built from the function box)"),
                                     "space": ("str", False, "without space_pk: discrete (default, a grid) | range (the box itself)"),
                                     "num": ("int", False, "grid points per dimension for a discrete space built here (default 21)"),
                                     "observations_pk": ("int", False, "observations to start from"),
                                     "num_random": ("int", False, "random evaluations first (default 10)"),
                                     "num_bayes": ("int", False, "Bayesian steps (default 20)"),
                                     "score": _SCORE, "num_rand_basis": _NRB, "num_search_each_probe": _NSEARCH, "seed": _SEED,
                                     **_RANGE_OPTS, "label": _LABEL}),
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
