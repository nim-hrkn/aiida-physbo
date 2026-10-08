# Design

## Why AiiDA for Bayesian optimization

A PHYSBO campaign in a notebook is a `Policy` object in memory. When the objective is an experiment or
a long calculation, the loop spans days and several people/agents; the state must live outside any
process. Here the state is the **chain of `ObservationsData` nodes**: each `observe` creates a new node
that contains everything observed so far and links to the previous one. `propose` reads
(candidates, observations) and returns the next actions; it keeps nothing. Any proposal can be traced
back to exactly which observations and which parameters produced it.

## Two kinds of space

| | discrete (`CandidatesData`) | range (`SearchBoxData`) |
|---|---|---|
| PHYSBO policy | `discrete.Policy` / `discrete_multi.Policy` | `range.Policy` / `range_multi.Policy` |
| what a proposal is | row indices (`actions`) + the rows `X` | coordinates `X` inside the box |
| how it is found | argmax of the acquisition over the table | an optimizer over the box: `random` (uniform samples, `optimizer_nsamples`) or `odatse` (`odatse_algorithm` exchange / pamc / minsearch / bayes, `odatse_params` merged over PHYSBO's `default_alg_dict`) |
| observations | `actions`, `X`, `t`; duplicates rejected | `X`, `t`; repeats allowed (noise) |
| runs out of points | yes (exit 420) | never |

`propose` dispatches on the type of the `space` input; the range-only parameters are rejected on a
discrete space (exit 411 in the WorkChain) so that a flag never silently does nothing.

ODAT-SE writes `odatse_output/` into the working directory, so the range `bayes_search` runs inside a
scratch directory that is removed afterwards (`_in_tempdir`); the daemon's cwd is not touched.

## Nodes

| node | type | content |
|---|---|---|
| candidates | `CandidatesData(ArrayData)`, entry point `physbo.candidates` | `X` (N, d); attribute `columns` |
| search box | `SearchBoxData(ArrayData)`, `physbo.search_box` | `min_X`, `max_X` (d,); attribute `columns` |
| observations | `ObservationsData(ArrayData)`, `physbo.observations` | `t` (M, k) float, `X` (M, d), `actions` (M,) int (discrete only); attribute `space`. 0.1.0 nodes have `actions`, `t` only |
| proposal | `ArrayData` | `X` (n, d) and, discrete, `actions` (n,) |
| posterior | `ArrayData` (optional) | `X`, `fmean`, `fstd` (N, k), `score` (N,) on the candidates or a grid |
| summary | `Dict` | mode, score, best so far, timings, physbo version |

`t` is always 2-D, like `physbo.Variable.t` (PHYSBO calls the objective value `t`; it is the usual "y").
Values are stored **raw**; the sign convention is a
parameter of `propose` (`maximize`, default True as in PHYSBO). PHYSBO test functions are minimization
problems, so the WorkChain stores `f` and runs `propose` with `maximize=False`.

## calcfunctions

| function | inputs | output |
|---|---|---|
| `candidates_from_file(source, options)` | `SinglefileData`, `Dict` | `CandidatesData` |
| `candidates_from_grid(spec)` | `Dict {min, max, num}` | `CandidatesData` (physbo `make_grid`) |
| `search_box(spec)` | `Dict {min, max}` | `SearchBoxData` |
| `observe(space, new, observations=None)` | `ArrayData {actions | X, t}`, previous chain node | new `ObservationsData` |
| `propose(space, parameters, observations=None)` | `Dict` (see `PROPOSE_DEFAULTS`) | `proposal`, `summary`, `[posterior]` (discrete) |
| `evaluate_test_function(space, proposal, objective)` | `Dict {name, kwargs, maximize}` | `ArrayData {actions?, X, t}` |
| `summarize(space, observations, settings)` | | `Dict` (best, best_X, best_sequence) |

`propose` rules:

- no observations (or `random: true`) → `random_search`; otherwise `bayes_search(max_num_probes=1,
  simulator=None)` which learns the hyperparameters every call (`interval=0`) and returns the actions.
- observed actions are excluded through `initial_data` (PHYSBO removes them from `policy.actions`).
- `num_objectives` defaults to the number of columns of the observations; the acquisition must match
  (`TS/EI/PI` for one, `TS/EHVI/HVPI` for several).
- every candidate observed (discrete) → `ValueError("every candidate has been observed already")`
  (exit 420 in the WorkChain).
- the summary always carries `posterior_at_proposal` (mean / std at the proposed points). With
  `posterior: true` a `posterior` ArrayData is stored with `X`, `fmean`, `fstd` (stored sign) and the
  acquisition `score`, on every candidate (discrete) or on a grid of `posterior_num` points per dimension
  (range, dim ≤ 2). `physbo-aiida posterior --pk` returns it thinned; `plot --pk <propose pk>` draws a
  1-D step. The ODAT-SE seed follows the propose `seed` (otherwise every step would start from the same
  point and `minsearch` could repeat one proposal).
- `observe` rejects actions outside `[0, N)`, duplicates (an action is a candidate; measuring it twice
  is a different experiment design), and a change of the number of objectives.

## PhysboOptimizeWorkChain (`physbo.optimize`)

Inputs: `space` (CandidatesData or SearchBoxData), `objective` Dict, `parameters` Dict (propose parameters; `num_objectives` and
`maximize` are set from the objective), `num_random` (default 10), `num_bayes` (default 20),
`observations` (optional start), `label`. One random `propose` of `num_random` points, then `num_bayes`
Bayesian steps; each step is `propose → evaluate_test_function → observe`, all called by the WorkChain
(`call_link_label` `propose_<i>` etc.). Outputs `observations` (the last chain node) and `summary`
(created by `summarize`). The seed, when given, is `seed + step`.

Exit codes: 400 a step raised; 410 bad objective (unknown name, kwargs, wrong dimension); 411 bad
parameters (unknown key, range-only option on a discrete space, observations of the other kind of
space); 420 candidates exhausted; 421 `num_random` 0 without observations.

## CLI / MCP

The same shape as aiida-akaikkr: `cli/spec.py` is the single table, `cli/main.py` builds argparse from
it, `mcp/server.py` builds the tools from it and only runs `physbo-aiida --json`. Validation is done in
`cli/steps.py` before anything is stored or submitted (`submit-optimize` even runs a tiny random
`propose` to validate the parameters, because the WorkChain would fail minutes later in the daemon).

## Not done yet

- An objective that is itself an AiiDA process (e.g. an AkaiKKR CalcJob through aiida-akaikkr). The
  interactive mode covers it today: `propose` → submit the CalcJob by hand → `observe` its result.
- Saving the trained predictor: `propose` is cheap enough to refit (seconds for hundreds of
  observations with `num_rand_basis` set).
