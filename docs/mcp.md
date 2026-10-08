# MCP server `aiida-physbo` (`physbo-mcp`)

## Promises

1. `mcp/server.py` imports neither `aiida` nor `physbo`; it runs the `physbo-aiida` binary with `--json`
   (`tests/test_spec_and_mcp.py::test_mcp_server_does_not_import_aiida_or_physbo`).
2. The only binary it runs is `physbo-aiida` (`PHYSBO_AIIDA_BIN` may point to it, nothing else), and the
   only subcommands are the keys of `cli/spec.py:SUBCOMMANDS`. No `verdi`: nothing from the MCP can
   delete nodes.
3. Every call is bounded by 55 s (the MCP bridge cuts at 60 s). `propose` runs synchronously; for a
   large candidate set pass `num_rand_basis` (e.g. 500) so PHYSBO uses the random-feature model.
   `submit-optimize` returns a pk immediately.
4. Tiers: read tools always; `--allow-submit` adds `physbo_candidates`, `physbo_search_box`, `physbo_observe`,
   `physbo_propose`, `physbo_submit_optimize`; `--allow-control` adds `physbo_daemon_start`,
   `physbo_daemon_stop`, `physbo_kill`.
5. Every tool argument reaches argv (`test_every_tool_argument_reaches_argv`), so no flag is silently dropped.
   Values are passed as `--flag=value`, so a negative number is never read as an option.

## Tools

| tool | subcommand | kind |
|---|---|---|
| physbo_status, physbo_daemon_status, physbo_test_functions | status, daemon-status, test-functions | read |
| physbo_process, physbo_list, physbo_wait | process, list, wait | read |
| physbo_space_info, physbo_history, physbo_proposal, physbo_posterior, physbo_results | space-info, history, proposal, posterior, results | read |
| physbo_plot, physbo_provenance | plot, provenance | read |
| physbo_candidates, physbo_search_box, physbo_observe, physbo_propose, physbo_submit_optimize | candidates, search-box, observe, propose, submit-optimize | submit |
| physbo_daemon_start, physbo_daemon_stop, physbo_kill | daemon-start, daemon-stop, kill | control |

## Wiring

```bash
claude mcp add aiida-physbo -- /home/kino/miniforge3/envs/akaikkr/bin/physbo-mcp --allow-submit
```

The AiiDA profile is the default profile of `~/.aiida` (or `--profile <name>` after `physbo-mcp`).
After editing `server.py`, reconnect the server in the client (`/mcp` → aiida-physbo → reconnect) so the
tool definitions are read again. After `pip install` of a new version: `verdi daemon restart` first
(with the env's `bin` on PATH), then restart the MCP client.

## A session

```
# discrete
physbo_candidates(grid={"min":[-5,-5],"max":[5,5],"num":21}, label="demo")      -> candidates pk C
physbo_propose(space_pk=C, num_search_each_probe=5, seed=1)                      -> 5 random actions (+ their X)
   ... evaluate them ...
physbo_observe(space_pk=C, actions="12,57,201,33,8", values="0.3,0.1,0.9,0.2,0.5") -> observations pk O1
physbo_propose(space_pk=C, observations_pk=O1, score="EI")                       -> next action
physbo_observe(space_pk=C, observations_pk=O1, actions="77", values="1.2")        -> O2
physbo_history(pk=O2)                                                             -> best so far, chain
# range
physbo_search_box(min="-2,-2", max="2,2", names="x,y")                            -> box pk B
physbo_propose(space_pk=B, num_search_each_probe=5, seed=1)                      -> 5 random coordinates
physbo_observe(space_pk=B, x=[[0.3,-1.2],[1.1,0.4]], values="-1.5,-1.3")          -> R1
physbo_propose(space_pk=B, observations_pk=R1, score="EI", optimizer="odatse", odatse_algorithm="minsearch")
```

## Troubleshooting

- `returned no JSON`: the CLI crashed before printing; `stderr_tail` has the reason (usually the profile
  or a missing package in the env of `physbo-aiida`).
- `did not finish within 55 s`: too many candidates for an exact GP; use `num_rand_basis`.
- `submit-optimize` returns `hint: the daemon is not running`: the WorkChain waits in Created until
  `physbo_daemon_start` (needs `--allow-control`) or `verdi daemon start` in the env.
- The WorkChain is `excepted` with `MissingEntryPointError`: the daemon was started before
  aiida-physbo was installed; `verdi daemon restart`.
