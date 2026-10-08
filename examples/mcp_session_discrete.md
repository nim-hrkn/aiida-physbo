# MCP session: discrete space (candidate grid)

Recorded 2026-10-08 from Claude Code with the `aiida-physbo` MCP server (aiida-physbo 0.2.0, PHYSBO 3.2.1,
profile `akaikkr`). Same objective as the range session, f(x, y) = (x − 0.5)² + (y + 1)² minimized, on a
21 × 21 grid over [−2, 2]² (441 candidates, spacing 0.2). On the grid the minimum is f = 0.01 at (0.4, −1)
or (0.6, −1).

| call | arguments | result |
|---|---|---|
| `physbo_candidates` | grid={"min": [-2, -2], "max": [2, 2], "num": 21}, names="x,y", label="mcp_discrete_demo" | CandidatesData **pk 23372**, 441 × 2 |
| `physbo_space_info` | pk=23372, head=3 | min/max, columns, first rows, creator `candidates_from_grid` |
| `physbo_propose` | space_pk=23372, num_search_each_probe=5, seed=42 | mode random, actions [78, 439, 152, 281, 394] with their rows X |
| `physbo_observe` | space_pk=23372, actions="78,439,152,281,394", values="7.61,10.09,1.21,0.37,6.05" | ObservationsData **pk 23379** (5) |
| `physbo_propose` | observations_pk=23379, score="EI", minimize=true, posterior=true, seed=1 | action 276 = (0.6, −1.4); 1.08 ± 2.24; posterior node pk 23384 (441 × 1) |
| `physbo_observe` | observations_pk=23379, actions="276", values="0.17" | pk 23387 (6) |
| `physbo_proposal` | pk=23382 | the proposal node with the full posterior stats (argmax 440, max std 3.95) |
| `physbo_propose` | observations_pk=23387, score="EI", minimize=true, num_search_each_probe=2, seed=2 | actions [424, 189] = (2.0, −1.2), (−0.2, −2.0) (marginal score for the 2nd point) |
| `physbo_observe` | observations_pk=23387, actions="424,189", values="2.29,1.49" | pk 23394 (8) |
| `physbo_propose` | observations_pk=23394, score="EI", minimize=true, seed=3 | action 279 = (0.6, −0.8); 0.036 ± 0.42 |
| `physbo_observe` | observations_pk=23394, actions="276", values="0.17" | **rejected**: `actions [276] are already observed in ObservationsData<23394>` (no node created) |
| `physbo_observe` | observations_pk=23394, actions="279", values="0.05" | pk 23401 (9) |
| `physbo_propose` | observations_pk=23401, score="TS", num_rand_basis=200, minimize=true, seed=4 | action 280 = (0.6, −0.6); 0.105 ± 0.145 (random-feature BLM path) |
| `physbo_history` | pk=23401, minimize=true | best f=0.05 at action 279; best-so-far 7.61 → 1.21 → 0.37 → 0.17 → 0.05; `num_remaining_after` 436 → 431 |
| `physbo_plot` | pk=23401, minimize=true | `mcp_discrete_demo_points.png`, `mcp_discrete_demo_best.png` |
| `physbo_observe` | observations_pk=23401, actions="280", values="0.17" | **pk 23408** (10, final) |

Outcome: best f = 0.05 at action 279 = (0.6, −0.8), next to the grid optimum (0.6, −1.0), after 5 random +
5 Bayesian evaluations.

What the discrete space guarantees that the range space does not
- every proposal is a row of the candidate table (action and X both returned);
- an observed action is never proposed again (`initial_data` removes it from the policy);
- recording an action twice is refused.
