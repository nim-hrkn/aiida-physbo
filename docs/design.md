# Design

## Why AiiDA for Bayesian optimization

A PHYSBO campaign in a notebook is a `Policy` object in memory. When the objective is an experiment or
a long calculation, the loop spans days and several people/agents; the state must live outside any
process. Here the state is the **chain of `ObservationsData` nodes**: each `observe` creates a new node
that contains everything observed so far and links to the previous one. `propose` reads
(candidates, observations) and returns the next actions; it keeps nothing. Any proposal can be traced
back to exactly which observations and which parameters produced it.

## Nodes

| node | type | content |
|---|---|---|
| candidates | `CandidatesData(ArrayData)`, entry point `physbo.candidates` | `X` (N, d); attribute `columns` |
| observations | `ObservationsData(ArrayData)`, `physbo.observations` | `actions` (M,) int, `t` (M, k) float, observation order |
| proposal | `ArrayData` | `actions` (n,), `X` (n, d) |
| summary | `Dict` | mode, score, best so far, timings, physbo version |
| posterior | `ArrayData` (optional) | `fmean`, `fstd` (N, k) on every candidate |

`t` is always 2-D, like `physbo.Variable.t`. Values are stored **raw**; the sign convention is a
parameter of `propose` (`maximize`, default True as in PHYSBO). PHYSBO test functions are minimization
problems, so the WorkChain stores `f` and runs `propose` with `maximize=False`.

## calcfunctions

| function | inputs | output |
|---|---|---|
| `candidates_from_file(source, options)` | `SinglefileData`, `Dict` | `CandidatesData` |
| `candidates_from_grid(spec)` | `Dict {min, max, num}` | `CandidatesData` (physbo `make_grid`) |
| `observe(candidates, new, observations=None)` | `ArrayData {actions, t}`, previous chain node | new `ObservationsData` |
| `propose(candidates, parameters, observations=None)` | `Dict` (see `PROPOSE_DEFAULTS`) | `proposal`, `summary`, `[posterior]` |
| `evaluate_test_function(candidates, proposal, objective)` | `Dict {name, kwargs, maximize}` | `ArrayData {actions, t}` |
| `summarize(candidates, observations, settings)` | | `Dict` (best, best_X, best_sequence) |

`propose` rules:

- no observations (or `random: true`) → `random_search`; otherwise `bayes_search(max_num_probes=1,
  simulator=None)` which learns the hyperparameters every call (`interval=0`) and returns the actions.
- observed actions are excluded through `initial_data` (PHYSBO removes them from `policy.actions`).
- `num_objectives` defaults to the number of columns of the observations; the acquisition must match
  (`TS/EI/PI` for one, `TS/EHVI/HVPI` for several).
- every candidate observed → `ValueError("every candidate has been observed already")`
  (exit 420 in the WorkChain).
- `observe` rejects actions outside `[0, N)`, duplicates (an action is a candidate; measuring it twice
  is a different experiment design), and a change of the number of objectives.

## PhysboOptimizeWorkChain (`physbo.optimize`)

Inputs: `candidates`, `objective` Dict, `parameters` Dict (propose parameters; `num_objectives` and
`maximize` are set from the objective), `num_random` (default 10), `num_bayes` (default 20),
`observations` (optional start), `label`. One random `propose` of `num_random` points, then `num_bayes`
Bayesian steps; each step is `propose → evaluate_test_function → observe`, all called by the WorkChain
(`call_link_label` `propose_<i>` etc.). Outputs `observations` (the last chain node) and `summary`
(created by `summarize`). The seed, when given, is `seed + step`.

Exit codes: 400 a step raised; 410 bad objective (unknown name, kwargs, wrong dimension, unknown
parameter); 420 candidates exhausted; 421 `num_random` 0 without observations.

## CLI / MCP

The same shape as aiida-akaikkr: `cli/spec.py` is the single table, `cli/main.py` builds argparse from
it, `mcp/server.py` builds the tools from it and only runs `physbo-aiida --json`. Validation is done in
`cli/steps.py` before anything is stored or submitted (`submit-optimize` even runs a tiny random
`propose` to validate the parameters, because the WorkChain would fail minutes later in the daemon).

## Not done yet

- PHYSBO `range` (continuous) policies: proposals would be points, not indices; observations would
  need `X` instead of `actions`.
- An objective that is itself an AiiDA process (e.g. an AkaiKKR CalcJob through aiida-akaikkr). The
  interactive mode covers it today: `propose` → submit the CalcJob by hand → `observe` its result.
- Saving the trained predictor: `propose` is cheap enough to refit (seconds for hundreds of
  observations with `num_rand_basis` set).
