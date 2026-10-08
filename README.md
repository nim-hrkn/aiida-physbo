# aiida-physbo

Bayesian optimization with [PHYSBO](https://github.com/issp-center-dev/PHYSBO) where every step is a node
in an [AiiDA](https://www.aiida.net) provenance graph, driven from the command line (`physbo-aiida`) or
from an MCP client such as Claude Code / Claude Desktop (`physbo-mcp`, server name `aiida-physbo`).

```
space (candidates X | box) ──► propose ──► proposal (actions, X | X) ──► [your experiment / calculation]
                                  ▲                                                │
                                  └──── observations ◄──── observe ◄── values (t) ──┘
```

- **Two kinds of search space.** A *discrete* space is a table of candidates (`CandidatesData`, array
  `X`; PHYSBO `discrete` policies; a proposal is a row index = action). A *range* space is a continuous
  box (`SearchBoxData`, arrays `min_X`, `max_X`; PHYSBO `range` policies; a proposal is a coordinate found
  by maximizing the acquisition with an optimizer: uniform random samples, or an ODAT-SE algorithm).
- **Interactive mode** (any objective, evaluated outside): store the space → `propose` → measure →
  `observe` → `propose` with the new observations pk → …  The whole state of a campaign is the chain of
  `ObservationsData` nodes (arrays `t` (M, k), `X` (M, d), and `actions` for a discrete space);
  `propose` is stateless and rebuilds the PHYSBO policy from them each time.
- **Closed loop** (PHYSBO test functions, for benchmarking and demonstration): `submit-optimize` hands a
  `PhysboOptimizeWorkChain` to the AiiDA daemon, on either kind of space.

Single objective (TS / EI / PI) and multi objective (TS / EHVI / HVPI, Pareto front) are supported on
both spaces.

## Install

Into the conda environment that holds the AiiDA profile and daemon (here `akaikkr`):

```bash
pip install -e /path/to/PHYSBO
pip install -e "/path/to/aiida-physbo[mcp,plot,test]"
verdi daemon restart          # the daemon must see the new entry points (physbo.optimize, physbo.candidates, ...)
```

## CLI

```bash
physbo-aiida --json status
physbo-aiida test-functions
# discrete space
physbo-aiida candidates --grid '{"min": [-5, -5], "max": [5, 5], "num": 21}' --label demo   # -> pk C
physbo-aiida candidates --file candidates.csv --columns 0,1,2 --names a,b,c
physbo-aiida propose --space-pk C --num-search-each-probe 5 --seed 1                          # random (nothing observed)
physbo-aiida observe --space-pk C --actions 12,57,201 --values 0.3,0.1,0.9                     # -> pk O1
physbo-aiida propose --space-pk C --observations-pk O1 --score EI --posterior                 # Bayesian
physbo-aiida observe --space-pk C --observations-pk O1 --actions 77 --values 1.2               # -> pk O2
# range space (continuous box)
physbo-aiida search-box --min -2,-2 --max 2,2 --names x,y --label demo-box                     # -> pk B
physbo-aiida propose --space-pk B --num-search-each-probe 5 --seed 1                          # random points
physbo-aiida observe --space-pk B --x '[[0.3,-1.2],[1.1,0.4]]' --values -1.5,-1.3              # -> pk R1
physbo-aiida propose --space-pk B --observations-pk R1 --score EI --optimizer odatse --odatse-algorithm minsearch
# read
physbo-aiida history --pk O2
physbo-aiida plot --pk O2
physbo-aiida submit-optimize --test-function Sphere --kwargs '{"dim": 2}' --num 21 --num-random 10 --num-bayes 20 --score EI
physbo-aiida submit-optimize --test-function Sphere --space range --num-random 10 --num-bayes 20 --score EI
physbo-aiida results --pk <workchain pk>
```

Option values that start with `-` (negative numbers) are accepted: the CLI rewrites `--values -4,-8` to
`--values=-4,-8` before argparse sees it.

Values are **maximized** by default, as in PHYSBO; pass `--minimize` to `propose` / `history` / `plot`
when the recorded values are to be minimized (or record `-f`). Multi-objective values are given as
JSON rows: `--values '[[0.1, 2.0], [0.3, 1.5]]'`.

With `--json` stdout is exactly one JSON object, also on failure (`{"ok": false, "error": ...}`).

## MCP

```bash
claude mcp add aiida-physbo -- /path/to/env/bin/physbo-mcp --allow-submit
```

Tools are `physbo_<subcommand>`: read tools always; `physbo_candidates`, `physbo_search_box`,
`physbo_observe`, `physbo_propose`, `physbo_submit_optimize` with `--allow-submit`; daemon start/stop and kill with
`--allow-control`. The server never imports aiida; it runs `physbo-aiida --json` as a subprocess with a
55 s timeout (see `docs/mcp.md`).

## Tests

```bash
python -m pytest tests/test_spec_and_mcp.py                              # no profile needed
python -m pytest -p aiida.tools.pytest_fixtures tests                    # temporary sqlite profile
```

## Layout

| file | role |
|---|---|
| `aiida_physbo/data.py` | `CandidatesData` (array `X`), `SearchBoxData` (arrays `min_X`, `max_X`), `ObservationsData` (arrays `t` (M, k), `X` (M, d), `actions` (discrete only)) |
| `aiida_physbo/calcfunctions.py` | `candidates_from_file`, `candidates_from_grid`, `search_box`, `observe`, `propose`, `evaluate_test_function`, `summarize`; `PROPOSE_DEFAULTS` |
| `aiida_physbo/objectives.py` | PHYSBO test functions as objectives (`{"name", "kwargs", "maximize"}`) |
| `aiida_physbo/workflows/optimize.py` | `PhysboOptimizeWorkChain` (entry point `physbo.optimize`) |
| `aiida_physbo/query/nodes.py` | read-only queries (status, history, proposal, results, provenance) |
| `aiida_physbo/cli/spec.py` | the single table of subcommands (also the MCP allow-list) |
| `aiida_physbo/mcp/server.py` | `physbo-mcp` |

Action log: `~/.aiida-physbo/log/action-<YYYY-MM>.jsonl` (every node-creating or control call, with `caller` cli/mcp).

## Versions

The version lives in `aiida_physbo/__init__.py` (`__version__`) and `pyproject.toml`; `physbo-aiida status`
reports it as `aiida_physbo_version`. The install is editable (`pip install -e`), so the installed version
is whatever the checked-out source says.

| version | date | what |
|---|---|---|
| 0.1.0 | 2026-10-08 | discrete space only: CandidatesData, ObservationsData (`actions`, `t`), propose / observe / optimize WorkChain, CLI, MCP |
| 0.2.0 | 2026-10-08 | range (continuous) space: SearchBoxData, PHYSBO `range` / `range_multi` policies with the random or ODAT-SE acquisition optimizer; ObservationsData gains `X` and the `space` attribute; CLI/MCP options `space_pk`, `search-box`, `space-info`, `--x`, optimizer options; negative option values accepted |

Known PHYSBO 3.2.1 limits surfaced here: the ODAT-SE `mapper` optimizer is excluded (ODAT-SE 4 writes a
header line PHYSBO does not skip), and a range space cannot propose several points per step with
`num_rand_basis > 0` (`Variable.add` shape error); both are rejected before anything runs.

License: Apache-2.0. PHYSBO itself is MPL-2.0.
