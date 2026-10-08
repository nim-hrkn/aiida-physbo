# Examples

Two ways to drive the same Bayesian-optimization loop, each on both kinds of search space.

| file | what |
|---|---|
| `interactive_loop.sh discrete|range [n_random] [n_bayes]` | the interactive (ask–tell) loop with the `physbo-aiida` CLI: store the space, `propose`, evaluate f outside, `observe`, repeat; prints the best and writes the figures |
| `submit_optimize.sh discrete|range [test_function] [n_random] [n_bayes]` | the closed loop in the AiiDA daemon (`PhysboOptimizeWorkChain`) on a PHYSBO test function |
| `one_dimensional.py discrete|range [--steps N] [--score EI] [--optimizer odatse --odatse-algorithm minsearch]` | 1-D Forrester function, 3 random points then N Bayesian steps run with `--posterior`; draws `one_dimensional_<space>.png`: per step the true f, observations, posterior mean ± 2σ, acquisition and the proposed point (data from `physbo-aiida posterior`) |
| `benchmark.py [--set quick|full]` | submits one WorkChain per benchmark case (Branin, SixHumpCamel, GoldsteinPrice, Levy, Hartmann3/6 with random and ODAT-SE optimizers, noisy Forrester, GramacyLee, DTLZ2), waits, prints a regret table and draws best-so-far curves (`figures/benchmark_quick.png`, `.json`) |
| `figures/` | the figures of `one_dimensional.py` for both spaces (and the ODAT-SE minsearch variant) |
| `mcp_session_range.md` | transcript of the loop driven from Claude Code through the MCP tools, continuous box |
| `mcp_session_discrete.md` | the same on a 21 × 21 candidate grid |

The objective in the scripts and the transcripts is f(x, y) = (x − 0.5)² + (y + 1)², minimized (true
minimum 0 at (0.5, −1)); in practice the values come from an experiment or a calculation. Run the scripts
from the conda env that holds the AiiDA profile:

```bash
export PATH=/home/kino/miniforge3/envs/akaikkr/bin:$PATH
examples/interactive_loop.sh range 5 5
examples/interactive_loop.sh discrete 5 5
examples/submit_optimize.sh range Sphere 5 10        # needs the daemon
```

Everything the scripts create is labelled `example_*`; the MCP transcripts created nodes labelled
`mcp_range_demo` / `mcp_discrete_demo` (pks 23327–23408 in the `akaikkr` profile).

## One-dimensional figures

`one_dimensional.py` minimizes the Forrester function f(x) = (6x − 2)² sin(12x − 4) on [0, 1] (minimum −6.021
at x = 0.757). Discrete: 101 candidates; range: the box with a 101-point posterior grid. 3 random points
(seed 7), then 6 Bayesian EI steps.

| run | result | figure |
|---|---|---|
| discrete | f = −6.017 at x = 0.76 (grid optimum) after 9 evaluations | `figures/one_dimensional_discrete.png` |
| range, random optimizer (default, 1000 samples) | f = −6.017 at x = 0.7547 after 9 evaluations | `figures/one_dimensional_range.png` |
| range, ODAT-SE minsearch | f = −5.911 at x = 0.7713 | `figures/one_dimensional_range_minsearch.png` |

What the figures show
- Steps 2–4 of the discrete run: with 4–6 observations PHYSBO's hyperparameter fit collapses to a very
  short length scale (flat posterior mean, wide band, acquisition that is flat except at the data). It
  recovers from step 5. This is PHYSBO's ML-II fit with few points, not a property of the plugin; with
  ~7 points the posterior tracks f.
- The acquisition (EI) panel explains each proposal: early steps probe the edges where the band is widest,
  later steps exploit the minimum near x ≈ 0.75.
- ODAT-SE `minsearch` (Nelder–Mead) starts from a point set by its seed; aiida-physbo passes the propose
  seed through so that the start moves between steps (before 0.3.0 the same point was proposed every step).
  On a flat EI landscape it can still stall; the random optimizer is the robust default in 1-D / 2-D.
- The same figure for a single step, without the true f, is produced by the plugin itself:
  `physbo-aiida plot --pk <propose process pk> --minimize` (1-D spaces, propose run with `--posterior`).

## Benchmark (`benchmark.py --set quick`, 2026-10-08, aiida-physbo 0.4.0, PHYSBO 3.2.1, seed 11)

One `PhysboOptimizeWorkChain` per case in the daemon; EI with the exact GP (`num_rand_basis=0`) unless
noted; range spaces use the random acquisition optimizer (2000–5000 samples) unless noted. "evals" =
random + Bayesian evaluations. Figure: `figures/benchmark_quick.png`, raw table `figures/benchmark_quick.json`.

| case | space | evals | best | f* | regret | reading |
|---|---|---|---|---|---|---|
| Branin | discrete 41×41 | 35 | 0.458 | 0.398 | 0.060 | one of the three minima found; grid spacing limits the rest |
| Branin | range | 35 | 1.167 | 0.398 | 0.769 | found the right basin, not refined; EI kept exploring the other two basins |
| SixHumpCamel | range | 35 | −0.916 | −1.032 | 0.116 | one of the two global basins |
| GoldsteinPrice | discrete 41×41 | 35 | 60.0 | 3.0 | 57.0 | **hard for a plain GP**: values span 3 … 10⁶, the GP fits the 10⁵ plateau and the minimum region is a tiny flat spot; a log transform of the observations (`observe` the log) is the standard fix |
| Levy (2-D) | range | 40 | 0.013 | 0 | 0.013 | many local minima, still solved |
| Hartmann3 | range | 40 | −3.854 | −3.863 | 0.008 | solved |
| Hartmann6 | range, random 5000 | 60 | −3.069 | −3.322 | 0.254 | typical for 60 evaluations in 6-D; still improving at the end |
| Hartmann6 | range, ODAT-SE exchange (300 steps) | 60 | −2.748 | −3.322 | 0.575 | worse than 5000 uniform samples at the same budget; the chain spends its steps near one mode of EI |
| Forrester + noise σ=0.5 | range | 15 | −1.714 | −6.021 | 4.307 | 3 random + 12 steps were not enough once noise is on; the hyperparameter fit stayed in the collapsed regime (flat posterior) for most steps |
| GramacyLee | discrete 201 | 18 | −0.342 | −0.869 | 0.528 | high-frequency sin / (2x): 18 points cannot resolve the period; needs ≥ 30 |
| DTLZ2 (2 obj, 3-D) | range, HVPI | 30 | Pareto 16 | sphere | – | 16 non-dominated points after 30 evaluations |

### Goldstein–Price with `--transform log` (same seed 11, 10 random + 25 EI)

The benchmark row above used the raw f. Recording log f instead (`submit-optimize --transform log`,
aiida-physbo 0.4.1; the regret below is converted back to raw f):

| case | best (raw f) | f* | regret |
|---|---|---|---|
| discrete 41×41, raw | 60.0 | 3 | 57.0 |
| discrete 41×41, **log** | 7.67 at (0.0, −0.9) | 3 | 4.67 |
| range, raw | 20.2 | 3 | 17.2 |
| range, **log** | 31.1 | 3 | 28.1 |

On the grid the log transform makes the difference between "stuck on the 10⁵ plateau" and "next to the
minimum (0, −1)". On the range space one seed is not enough to tell (both runs end in the right valley
but far from the bottom); more seeds or more evaluations are needed before drawing a conclusion there.

Lessons that carry over to real problems
- Scale matters more than multimodality: Levy and Hartmann3 (bounded, O(1) values) are solved in 40
  evaluations, Goldstein–Price (6 orders of magnitude) is not. Transform wide-range objectives before
  recording them, or record −log f.
- In 6-D, 5000 uniform samples beat an ODAT-SE replica-exchange run of 300 steps at maximizing EI; the
  optimizer of the acquisition is a real knob for range spaces. Increase `optimizer_nsamples` with the
  dimension (it is cheap: the GP prediction is vectorized).
- Few-point hyperparameter collapse (also seen in the 1-D figures) hurts most when the budget is small
  or the observations are noisy. More random points first (≥ 5–10) or a BLM (`num_rand_basis`) are the
  two available remedies in PHYSBO.
- The regret column comes from `results` (`known_minimum`), so any run can be scored after the fact.
