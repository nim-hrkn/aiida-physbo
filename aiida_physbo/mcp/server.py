# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""`physbo-mcp`: MCP server "aiida-physbo" exposing the `physbo-aiida` CLI as tools.

Promises (docs/mcp.md):
- never imports aiida or physbo; it only runs the `physbo-aiida` binary with `--json`;
- the only binary it can run is `physbo-aiida` and the only subcommands are those of cli/spec.py;
- every call is bounded by TIMEOUT (< 60 s bridge); the WorkChain submission returns a pk;
- node-creating tools (candidates / observe / propose / submit-optimize) appear only with --allow-submit,
  daemon start/stop and kill only with --allow-control.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

from ..cli.spec import CONTROL, READ, SUBCOMMANDS, SUBMIT, option_flag

SERVER_NAME = "aiida-physbo"
ALLOWED_BINARY_NAMES = ("physbo-aiida",)
ALLOWED_SUBCOMMANDS = frozenset(SUBCOMMANDS)
TIMEOUT = 55

# tool name -> subcommand
TOOLS = {
    "physbo_status": "status", "physbo_daemon_status": "daemon-status", "physbo_test_functions": "test-functions",
    "physbo_process": "process", "physbo_list": "list", "physbo_wait": "wait",
    "physbo_space_info": "space-info", "physbo_history": "history", "physbo_proposal": "proposal",
    "physbo_results": "results", "physbo_plot": "plot", "physbo_provenance": "provenance",
    "physbo_candidates": "candidates", "physbo_search_box": "search-box", "physbo_observe": "observe", "physbo_propose": "propose",
    "physbo_submit_optimize": "submit-optimize",
    "physbo_daemon_start": "daemon-start", "physbo_daemon_stop": "daemon-stop", "physbo_kill": "kill",
}

_SETTINGS = {"profile": None}


def binary() -> str:
    """path of the physbo-aiida binary: PHYSBO_AIIDA_BIN, else next to this interpreter, else PATH."""
    env = os.environ.get("PHYSBO_AIIDA_BIN")
    if env:
        path = env
    else:
        path = os.path.join(os.path.dirname(sys.executable), "physbo-aiida")
        if not os.path.exists(path):
            path = shutil.which("physbo-aiida") or path
    if os.path.basename(path) not in ALLOWED_BINARY_NAMES:
        raise RuntimeError(f"refusing to run {path!r}: only {ALLOWED_BINARY_NAMES} are allowed")
    return path


def build_argv(subcommand: str, kwargs: dict) -> list:
    """argv of the CLI call; every non-None argument of the tool is passed as an option."""
    if subcommand not in ALLOWED_SUBCOMMANDS:
        raise ValueError(f"subcommand {subcommand!r} is not allowed")
    options = SUBCOMMANDS[subcommand]["options"]
    unknown = set(kwargs) - set(options)
    if unknown:
        raise ValueError(f"{subcommand}: unknown arguments {sorted(unknown)}")
    argv = [binary(), "--json", "--caller", "mcp"]
    if _SETTINGS["profile"]:
        argv += ["--profile", _SETTINGS["profile"]]
    argv.append(subcommand)
    for name, (typ, _required, _help) in options.items():
        value = kwargs.get(name)
        if value is None:
            continue
        if typ == "bool":
            if value:
                argv.append(option_flag(name))
        elif isinstance(value, (dict, list)):
            # MCP clients may turn a JSON string argument into an object before sending it
            argv.append(f"{option_flag(name)}={json.dumps(value)}")
        else:
            # --flag=value: a value such as "-4.0,-8.0" must not be read as an option by argparse
            argv.append(f"{option_flag(name)}={value}")
    return argv


def run(subcommand: str, **kwargs) -> dict:
    argv = build_argv(subcommand, kwargs)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{subcommand} did not finish within {TIMEOUT} s",
                "hint": "for large candidate sets use num_rand_basis (e.g. 500) or fewer candidates; "
                        "the CLI may still be running, see physbo_list"}
    stderr_tail = proc.stderr.strip().splitlines()[-5:]
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": f"{subcommand} returned no JSON (exit {proc.returncode})",
                "stdout": proc.stdout[-2000:], "stderr_tail": stderr_tail}
    if not out.get("ok") and stderr_tail:
        out["stderr_tail"] = stderr_tail
    return out


# ---------------------------------------------------------------- tools (READ)
def physbo_status() -> dict:
    """AiiDA profile, daemon state, aiida/physbo versions, and how many candidate sets, observation nodes and optimize runs exist."""
    return run("status")


def physbo_daemon_status() -> dict:
    """Whether the AiiDA daemon runs and with how many workers (needed only for physbo_submit_optimize)."""
    return run("daemon-status")


def physbo_test_functions() -> dict:
    """PHYSBO test functions (name, single/multi objective, constructor parameters, default box) usable as the objective of physbo_submit_optimize."""
    return run("test-functions")


def physbo_process(pk: int) -> dict:
    """State, exit code, inputs/outputs, last reports and children of a process node."""
    return run("process", pk=pk)


def physbo_list(label_prefix: str | None = None, days: int | None = None, state: str | None = None,
                limit: int | None = None) -> dict:
    """Recent physbo processes (propose, observe, candidates, evaluate, optimize WorkChains), newest first."""
    return run("list", label_prefix=label_prefix, days=days, state=state, limit=limit)


def physbo_wait(pk: int, wait_seconds: int | None = None) -> dict:
    """Wait at most 45 s for a process to terminate and report its state."""
    return run("wait", pk=pk, wait_seconds=wait_seconds)


def physbo_space_info(pk: int, head: int | None = None) -> dict:
    """A search space node: CandidatesData (discrete: shape, feature names, per-feature min/max, first rows) or SearchBoxData
    (range: the box min/max)."""
    return run("space-info", pk=pk, head=head)


def physbo_history(pk: int, minimize: bool = False, max_rows: int | None = None) -> dict:
    """All observations of a campaign in order (actions for a discrete space, coordinates X, and values t), the best-so-far
    sequence or the Pareto front, and the chain of observe steps. `pk` is an ObservationsData, an optimize WorkChain, or a
    propose/observe process. minimize=true evaluates "best" as the smallest value."""
    return run("history", pk=pk, minimize=minimize, max_rows=max_rows)


def physbo_proposal(pk: int) -> dict:
    """The proposed points (actions = candidate indices for a discrete space, and coordinates X) of a proposal (pk of the
    proposal ArrayData or of its propose process), the summary (mode, score, best so far, posterior mean/std at the proposed
    points) and, when stored, the full posterior node."""
    return run("proposal", pk=pk)


def physbo_results(pk: int) -> dict:
    """Summary of an optimize WorkChain (best value / Pareto front, best X, best-so-far sequence, observations pk, steps done)
    or of a propose / observe process."""
    return run("results", pk=pk)


def physbo_plot(pk: int, minimize: bool = False, outdir: str | None = None, prefix: str | None = None) -> dict:
    """Write a PNG: best-so-far vs observation (one objective) or observed points with the Pareto front (several). Returns the paths."""
    return run("plot", pk=pk, minimize=minimize, outdir=outdir, prefix=prefix)


def physbo_provenance(pk: int, outdir: str | None = None, ancestor_depth: int | None = None,
                      descendant_depth: int | None = None, fmt: str | None = None) -> dict:
    """Draw the provenance graph of a node with graphviz (png/pdf/svg); returns the file path."""
    return run("provenance", pk=pk, outdir=outdir, ancestor_depth=ancestor_depth, descendant_depth=descendant_depth, fmt=fmt)


# ---------------------------------------------------------------- tools (SUBMIT: create nodes)
def physbo_candidates(file: str | None = None, format: str | None = None, delimiter: str | None = None,
                      skip_header: int | None = None, columns: str | None = None, grid: str | dict | None = None,
                      test_function: str | None = None, kwargs: str | dict | None = None, num: int | None = None,
                      names: str | None = None, label: str | None = None) -> dict:
    """Store a DISCRETE search space (the candidate table, one row per candidate; proposals are row indices) as a
    CandidatesData node and return its pk. Give exactly one of: `file` (csv/tsv/npy/npz path on this machine; `columns` picks
    csv columns, `skip_header` skips lines), `grid` ({"min": [...], "max": [...], "num": int or [...]} regular grid), or
    `test_function` (grid over the default box of a PHYSBO test function, `num` points per dimension). `names` are comma
    separated feature names. For a continuous space use physbo_search_box instead."""
    return run("candidates", file=file, format=format, delimiter=delimiter, skip_header=skip_header, columns=columns,
               grid=grid, test_function=test_function, kwargs=kwargs, num=num, names=names, label=label)


def physbo_search_box(min: str | None = None, max: str | None = None, test_function: str | None = None,
                      kwargs: str | dict | None = None, names: str | None = None, label: str | None = None) -> dict:
    """Store a CONTINUOUS search space (a box; PHYSBO range policies propose coordinates inside it, not indices) as a
    SearchBoxData node and return its pk. Give `min` and `max` as comma separated bounds per dimension (e.g. "-2,-2" and
    "2,2"), or `test_function` to take the default box of a PHYSBO test function. `names` are comma separated feature names."""
    return run("search-box", min=min, max=max, test_function=test_function, kwargs=kwargs, names=names, label=label)


def physbo_observe(space_pk: int, observations_pk: int | None = None, actions: str | None = None,
                   x: str | list | None = None, values: str | list | None = None, file: str | None = None,
                   label: str | None = None) -> dict:
    """Record measured objective values. Discrete space (CandidatesData): `actions` = comma separated candidate indices
    (e.g. "3,17"). Range space (SearchBoxData): `x` = the measured coordinates as JSON rows [[x1,x2],...] or "x1,x2;x1,x2".
    `values` = the corresponding values ("0.12,0.07" for one objective, or JSON rows [[f1,f2],...] for several); or `file` =
    csv (discrete: action then values per row; range: the d coordinates then values). Pass the previous `observations_pk` to
    append to a campaign (omit it to start one). Returns the new ObservationsData pk, which the next physbo_propose takes."""
    return run("observe", space_pk=space_pk, observations_pk=observations_pk, actions=actions, x=x, values=values,
               file=file, label=label)


def physbo_propose(space_pk: int, observations_pk: int | None = None, score: str | None = None,
                   num_rand_basis: int | None = None, num_search_each_probe: int | None = None, seed: int | None = None,
                   num_objectives: int | None = None, minimize: bool = False, random: bool = False, posterior: bool = False,
                   interval: int | None = None, optimizer: str | None = None, optimizer_nsamples: int | None = None,
                   odatse_algorithm: str | None = None, odatse_params: str | dict | None = None,
                   label: str | None = None) -> dict:
    """PHYSBO proposes the next points to evaluate from the observations so far (random when observations_pk is omitted or
    random=true). The space is a CandidatesData (discrete: proposals are candidate indices + their rows X) or a SearchBoxData
    (range: proposals are coordinates X found by maximizing the acquisition with `optimizer` random (uniform samples,
    `optimizer_nsamples`) or odatse (`odatse_algorithm` exchange | pamc | minsearch | bayes, `odatse_params` JSON overrides)).
    score: TS (default) | EI | PI for one objective, TS | EHVI | HVPI for several; num_rand_basis 0 = exact Gaussian process
    (use ~500 for thousands of candidates); num_search_each_probe = how many points; minimize=true if the recorded values are
    to be minimized. Returns the proposal pk, actions / X, and a summary with the best so far and the posterior at the
    proposed points. Runs synchronously (seconds) and is recorded as a calcfunction."""
    return run("propose", space_pk=space_pk, observations_pk=observations_pk, score=score,
               num_rand_basis=num_rand_basis, num_search_each_probe=num_search_each_probe, seed=seed,
               num_objectives=num_objectives, minimize=minimize, random=random, posterior=posterior, interval=interval,
               optimizer=optimizer, optimizer_nsamples=optimizer_nsamples, odatse_algorithm=odatse_algorithm,
               odatse_params=odatse_params, label=label)


def physbo_submit_optimize(test_function: str, kwargs: str | dict | None = None, maximize: bool = False,
                           space_pk: int | None = None, space: str | None = None, num: int | None = None,
                           observations_pk: int | None = None, num_random: int | None = None, num_bayes: int | None = None,
                           score: str | None = None, num_rand_basis: int | None = None, num_search_each_probe: int | None = None,
                           seed: int | None = None, optimizer: str | None = None, optimizer_nsamples: int | None = None,
                           odatse_algorithm: str | None = None, odatse_params: str | dict | None = None,
                           label: str | None = None) -> dict:
    """Submit a closed-loop Bayesian optimization WorkChain to the AiiDA daemon on a PHYSBO test function (see
    physbo_test_functions): num_random random evaluations, then num_bayes Bayesian steps of propose -> evaluate -> observe.
    The space is `space_pk` (CandidatesData or SearchBoxData) or is built from the function's box: space="discrete" (default,
    a grid of `num` points per dimension) or space="range" (the box itself, with optimizer / odatse options as in
    physbo_propose). The function is minimized unless maximize=true. Returns the WorkChain pk; poll with physbo_wait /
    physbo_process, read with physbo_results."""
    return run("submit-optimize", test_function=test_function, kwargs=kwargs, maximize=maximize, space_pk=space_pk, space=space,
               num=num, observations_pk=observations_pk, num_random=num_random, num_bayes=num_bayes, score=score,
               num_rand_basis=num_rand_basis, num_search_each_probe=num_search_each_probe, seed=seed, optimizer=optimizer,
               optimizer_nsamples=optimizer_nsamples, odatse_algorithm=odatse_algorithm, odatse_params=odatse_params, label=label)


# ---------------------------------------------------------------- tools (CONTROL)
def physbo_daemon_start(workers: int | None = None) -> dict:
    """Start the AiiDA daemon (logged)."""
    return run("daemon-start", workers=workers)


def physbo_daemon_stop() -> dict:
    """Stop the AiiDA daemon (logged)."""
    return run("daemon-stop")


def physbo_kill(pk: int) -> dict:
    """Kill a running process (logged)."""
    return run("kill", pk=pk)


def tool_functions(allow_submit: bool = False, allow_control: bool = False) -> dict:
    """tool name -> function, filtered by the flags."""
    allowed_kinds = {READ}
    if allow_submit:
        allowed_kinds.add(SUBMIT)
    if allow_control:
        allowed_kinds.add(CONTROL)
    module = sys.modules[__name__]
    return {name: getattr(module, name) for name, sub in TOOLS.items() if SUBCOMMANDS[sub]["kind"] in allowed_kinds}


def build_server(allow_submit: bool = False, allow_control: bool = False):
    from mcp.server.mcpserver import MCPServer

    from .. import __version__

    flags = []
    if allow_submit:
        flags.append("submit")
    if allow_control:
        flags.append("control")
    server = MCPServer(
        name=SERVER_NAME, version=__version__,
        instructions="Bayesian optimization with PHYSBO, recorded in AiiDA. Two kinds of search space: a discrete candidate "
                     "table (physbo_candidates; proposals are row indices) or a continuous box (physbo_search_box; proposals "
                     "are coordinates). Interactive loop: store the space -> physbo_propose (next points) -> evaluate outside "
                     "-> physbo_observe (record values, returns a new observations pk) -> physbo_propose with that pk -> ... "
                     "physbo_history shows the campaign. "
                     "physbo_submit_optimize runs a closed loop on a PHYSBO test function in the daemon (returns a pk; poll "
                     "with physbo_wait / physbo_results). Read-only tools are always available"
                     + (f"; enabled write tools: {', '.join(flags)}" if flags else "; no write tools enabled") + ".")
    for name, fn in tool_functions(allow_submit, allow_control).items():
        server.add_tool(fn, name=name, description=fn.__doc__)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="physbo-mcp", description="MCP server for aiida-physbo")
    parser.add_argument("--allow-submit", action="store_true", help="expose the node-creating tools (candidates, observe, propose, submit-optimize)")
    parser.add_argument("--allow-control", action="store_true", help="expose daemon start/stop and kill")
    parser.add_argument("--profile", default=None, help="AiiDA profile passed to every CLI call")
    args = parser.parse_args(argv)
    _SETTINGS["profile"] = args.profile
    build_server(args.allow_submit, args.allow_control).run("stdio")


if __name__ == "__main__":
    main()
