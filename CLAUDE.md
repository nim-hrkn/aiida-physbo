# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

aiida-physbo records PHYSBO Bayesian optimization in AiiDA provenance and exposes it as a CLI
(`physbo-aiida`) and an MCP server (`physbo-mcp`, server name `aiida-physbo`). It mirrors the shape of
the sibling plugin aiida-akaikkr (`../../AKAIKKR/aiida-akaikkr`): `cli/spec.py` is the single table
of subcommands, argparse and the MCP tool list are both generated from it, and the MCP server never
imports aiida or physbo. Read `README.md`, `docs/design.md` and `docs/mcp.md` first.

## Environment and commands

The package is installed editable in the conda env `akaikkr` (`/home/kino/miniforge3/envs/akaikkr`),
which holds AiiDA 2.9 + the `akaikkr` profile + the daemon + mcp 2.x. PHYSBO is installed editable from
`../PHYSBO`. Always put that env's `bin` on PATH (verdi, physbo-aiida, python).

```bash
export PATH=/home/kino/miniforge3/envs/akaikkr/bin:$PATH
python -m pytest -q tests/test_spec_and_mcp.py                       # layer 1: no profile, ~1 s
python -m pytest -q -p aiida.tools.pytest_fixtures tests              # layer 2: temporary sqlite profile, ~20 s
python -m pytest -q -p aiida.tools.pytest_fixtures tests/test_workchain.py::test_optimize_workchain_sphere
physbo-aiida --json status                                            # against the real profile
pip install -e ".[mcp,plot,test]"                                     # after changing entry points in pyproject.toml
```

After changing entry points or anything the daemon runs (workflows, calcfunctions, data):
`verdi daemon restart` from a shell with that PATH and `SSH_AUTH_SOCK=/home/kino/.ssh/agent.sock`,
and only when `verdi process list` shows nothing running. After changing `mcp/server.py`:
reconnect the server in the MCP client (`/mcp` → aiida-physbo → reconnect).

## Architecture in one paragraph

`CandidatesData` (array `X`; discrete space), `SearchBoxData` (arrays `min_X`, `max_X`; continuous
range space) and `ObservationsData` (arrays `t` always (M, k), `X` (M, d), and `actions` for a discrete
space; attribute `space`) are typed ArrayData nodes (`data.py`). `calcfunctions.propose(space,
parameters, observations=None)` is stateless: it rebuilds a `physbo.search.{discrete,range}[_multi].Policy`
with `initial_data`, calls `bayes_search(max_num_probes=1, simulator=None)` (or `random_search` when
nothing is observed) and returns `proposal` (X, and actions for discrete), `summary`, optional
`posterior` (X, fmean, fstd, score on the candidates or on a grid over a 1-D/2-D box; read with the
`posterior` command, drawn by `plot` for 1-D). For a range space the acquisition is maximized by `physbo.search.optimize.random` or
`.odatse` (run in a scratch cwd). `observe` appends to the chain and rejects duplicate actions
(discrete only). `PhysboOptimizeWorkChain` (`workflows/optimize.py`, entry point
`physbo.optimize`) loops propose → `evaluate_test_function` → observe on a PHYSBO test function
(`objectives.py`; test functions are minimized, so it stores f and sets `maximize=False`). `query/nodes.py`
holds every read; `cli/steps.py` validates before creating nodes and logs to `~/.aiida-physbo/log/`.

## Rules carried over from the sibling projects

- Add a subcommand in `cli/spec.py` → implement → add the `physbo_<name>` tool in `mcp/server.py`
  with the **same parameter names**; `test_every_tool_argument_reaches_argv` fails otherwise.
- The MCP allow-list is `SUBCOMMANDS`; never add a way to run `verdi` or another binary from the MCP.
- `--json` stdout is one JSON object, also on failure. Library prints go to stderr.
- Value options are passed as `--flag=value` (a value such as `-4.0,-8.0` is otherwise read as a flag).
- `propose` parameter keys live only in `PROPOSE_DEFAULTS`; the CLI builds them through
  `steps._propose_parameters`.
- Bump `__version__` in `aiida_physbo/__init__.py` and `pyproject.toml` together, and add a row to the
  README "Versions" table.
- Known PHYSBO 3.2.1 limits (checked before running, keep the checks): ODAT-SE `mapper` is excluded; a
  range space cannot propose several points per step with `num_rand_basis > 0`.
