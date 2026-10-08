# Results of the examples

All runs were made on 2026-10-08 with aiida-physbo 0.2.0–0.4.1, PHYSBO 3.2.1, ODAT-SE 4.0.1, on one AiiDA
profile (sqlite). Every number below can be reproduced from the nodes it names (`physbo-aiida history`,
`results`, `posterior`) or by rerunning the scripts in `examples/`. Figures live in `examples/figures/`
(committed) and, for the plugin-drawn ones, in `~/aiida_work/figures/<pk>/` on the machine that ran them.

Convention: PHYSBO maximizes; every objective here is a minimization problem, so values were either
recorded as f with `minimize=true` (interactive) or the WorkChain was told `maximize: false`.
"regret" is |best − f*| in raw f.

## 1. Interactive loop from the MCP tools (2-D)

Objective f(x, y) = (x − 0.5)² + (y + 1)², minimum 0 at (0.5, −1); values computed by the agent and
recorded with `physbo_observe`. Transcripts: `examples/mcp_session_range.md`, `examples/mcp_session_discrete.md`.

| space | evaluations | best f | best point | notes |
|---|---|---|---|---|
| range, box [−2, 2]² (pk 23327) | 5 random + 5 EI | 0.0159 | (0.541, −0.881) | optimizers used per step: ODAT-SE minsearch, random 3000, ODAT-SE exchange (numsteps 200), random, random; one step (exchange) explored a corner (f = 7.7) |
| discrete, 21 × 21 grid (pk 23372) | 5 random + 5 EI/TS | 0.05 | action 279 = (0.6, −0.8) | grid optimum 0.01 at (0.6, −1); a duplicate `observe` of action 276 was rejected; one step proposed 2 points at once; the last step used TS with `num_rand_basis=200` |

Each `propose` took 0.3–0.6 s. The posterior at the proposed point is in every summary; in the discrete
run the full posterior over the 441 candidates was stored once (pk 23384).

## 2. One-dimensional runs with figures (`examples/one_dimensional.py`)

Forrester f(x) = (6x − 2)² sin(12x − 4) on [0, 1], minimum −6.0207 at x = 0.7572. 3 random points
(seed 7), then 6 Bayesian EI steps, each run with `--posterior` so that the posterior mean ± 2σ and the
acquisition could be drawn.

| run | best f | at x | figure |
|---|---|---|---|
| discrete, 101 candidates | −6.0167 | 0.76 (grid optimum) | `examples/figures/one_dimensional_discrete.png` |
| range, random optimizer (1000 samples) | −6.0172 | 0.7547 | `examples/figures/one_dimensional_range.png` |
| range, ODAT-SE minsearch | −5.9110 | 0.7713 | `examples/figures/one_dimensional_range_minsearch.png` |

What the figures show:

- With 3–6 observations PHYSBO's hyperparameter fit (type-II maximum likelihood) can collapse to a very
  short length scale: flat posterior mean, uniform band, constant EI, proposals at the edges. From 5–7
  points on the posterior tracks f and EI points at the minimum. This is PHYSBO's behaviour, not the
  plugin's.
- The EI panel explains each proposal: early steps probe the widest band, later steps exploit the valley
  near x ≈ 0.75.
- ODAT-SE minsearch starts from a point set by its seed. Before aiida-physbo 0.3.0 the same point was
  proposed six times in a row because the seed was fixed; the propose `seed` is now passed through.

The plugin draws the same picture for one step, without the true f: `physbo-aiida plot --pk <propose pk>`
(for example pk 23800, `~/aiida_work/figures/23800/propose_23800_posterior.png`).

## 3. Benchmark (`examples/benchmark.py --set quick`)

One `PhysboOptimizeWorkChain` per case in the daemon, seed 11, EI with the exact GP unless noted, random
acquisition optimizer on range spaces unless noted. Figure `examples/figures/benchmark_quick.png`, raw
table `examples/figures/benchmark_quick.json`. WorkChain pks 23815–24075.

| case | space | evals | best | f* | regret | reading |
|---|---|---|---|---|---|---|
| Branin | discrete 41×41 | 35 | 0.458 | 0.398 | 0.060 | one of three minima found; grid spacing limits the rest |
| Branin | range | 35 | 1.167 | 0.398 | 0.769 | right basin, not refined |
| SixHumpCamel | range | 35 | −0.916 | −1.032 | 0.116 | one of the two global basins |
| GoldsteinPrice | discrete 41×41 | 35 | 60.0 | 3.0 | 57.0 | values span 3 … 10⁶; the GP fits the plateau (see §4) |
| Levy (2-D) | range | 40 | 0.013 | 0 | 0.013 | solved despite many local minima |
| Hartmann3 | range | 40 | −3.854 | −3.863 | 0.008 | solved |
| Hartmann6 | range, random 5000 | 60 | −3.069 | −3.322 | 0.254 | typical for 60 evaluations in 6-D |
| Hartmann6 | range, ODAT-SE exchange (300 steps) | 60 | −2.748 | −3.322 | 0.575 | worse than 5000 uniform samples at the same budget |
| Forrester + noise σ = 0.5 | range | 15 | −1.714 | −6.021 | 4.307 | too few points once noise is on |
| GramacyLee | discrete 201 | 18 | −0.342 | −0.869 | 0.528 | high-frequency; 18 points cannot resolve the period |
| DTLZ2 (2 objectives, 3-D) | range, HVPI | 30 | Pareto front of 16 points | sphere | – | |

## 4. Goldstein–Price with the log transform (0.4.1, `--transform log`)

Same seed 11, 10 random + 25 EI. WorkChain pks 26527 (discrete log), 26546 (range log), 26569 (range raw).

| case | best (raw f) | f* | regret |
|---|---|---|---|
| discrete 41×41, raw | 60.0 | 3 | 57.0 |
| discrete 41×41, log | 7.67 at (0.0, −0.9) | 3 | 4.67 |
| range, raw | 20.2 | 3 | 17.2 |
| range, log | 31.1 | 3 | 28.1 |

On the grid, recording log f turns "stuck on the 10⁵ plateau" into "next to the minimum (0, −1)". On the
range space one seed does not separate the two; both end in the right valley but far from the bottom.

## 5. A noisy Branin from the MCP tools (0.4.0)

`physbo_submit_optimize(test_function="Branin", space="range", num_random=6, num_bayes=12, score="EI",
optimizer_nsamples=3000, noise=0.1, noise_seed=5, seed=21)` → WorkChain pk 26411, 18 evaluations in
about a minute: best 2.154 at (8.92, 2.83), in the basin of the third minimum (9.42, 2.48); regret 1.76
reported by `physbo_results` from `known_minimum`.

## 6. Lessons

- Scale matters more than multimodality: Levy and Hartmann3 (O(1) values) are solved in 40 evaluations;
  Goldstein–Price (six orders of magnitude) is not until the observations are log-transformed.
- In 6-D, 5000 uniform samples beat an ODAT-SE replica-exchange run of 300 steps at maximizing EI;
  increase `optimizer_nsamples` with the dimension.
- Few-point hyperparameter collapse hurts most with a small budget or noisy observations; use more
  random points first (≥ 5–10) or the random-feature BLM (`num_rand_basis`).
- One seed is not a conclusion (the range Goldstein–Price pair is the example).
- Operational: while other jobs saturate the machine, the daemon worker's BLAS threads compete and a
  propose step can go from 1 s to 20–70 s; set `OMP_NUM_THREADS` for the daemon when sharing a node.

## 7. PHYSBO 3.2.1 issues found on the way

| issue | plugin behaviour | fix (fork branches of PHYSBO) |
|---|---|---|
| ODAT-SE `mapper`: `default_alg_dict` passes numpy arrays; ODAT-SE 4's `MeshIterator` computes `[1] + num_list` element-wise, the mesh collapses onto a diagonal; `ColorMap.txt` has a header line the reader fed to `float()` | `mapper` excluded from the optimizer choices | `fix/odatse-mapper-header`: plain lists, use `result["x"]`, header-tolerant reader |
| range / range_multi with `num_search_each_probe > 1` and `num_rand_basis > 0`: virtual training points added without their basis Z → "The number of X and Z must be the same" | rejected before running | `fix/range-multi-probe-blm`: pass `predictor.get_basis(...)` to `Variable.add` |

Once a PHYSBO with both fixes is installed, the two pre-checks in `calcfunctions.py` and
`cli/steps.py` can be removed.
