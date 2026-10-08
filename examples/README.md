# Examples

Two ways to drive the same Bayesian-optimization loop, each on both kinds of search space.

| file | what |
|---|---|
| `interactive_loop.sh discrete|range [n_random] [n_bayes]` | the interactive (ask–tell) loop with the `physbo-aiida` CLI: store the space, `propose`, evaluate f outside, `observe`, repeat; prints the best and writes the figures |
| `submit_optimize.sh discrete|range [test_function] [n_random] [n_bayes]` | the closed loop in the AiiDA daemon (`PhysboOptimizeWorkChain`) on a PHYSBO test function |
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
