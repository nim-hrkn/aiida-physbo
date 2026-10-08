# Examples

Two ways to drive the same Bayesian-optimization loop, each on both kinds of search space.

| file | what |
|---|---|
| `interactive_loop.sh discrete|range [n_random] [n_bayes]` | the interactive (ask–tell) loop with the `physbo-aiida` CLI: store the space, `propose`, evaluate f outside, `observe`, repeat; prints the best and writes the figures |
| `submit_optimize.sh discrete|range [test_function] [n_random] [n_bayes]` | the closed loop in the AiiDA daemon (`PhysboOptimizeWorkChain`) on a PHYSBO test function |
| `one_dimensional.py discrete|range [--steps N] [--score EI] [--optimizer odatse --odatse-algorithm minsearch]` | 1-D Forrester function, 3 random points then N Bayesian steps run with `--posterior`; draws `one_dimensional_<space>.png`: per step the true f, observations, posterior mean ± 2σ, acquisition and the proposed point (data from `physbo-aiida posterior`) |
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
