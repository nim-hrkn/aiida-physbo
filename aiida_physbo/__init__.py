# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""aiida-physbo: Bayesian optimization with PHYSBO, recorded in AiiDA.

Layers (the same shape as aiida-akaikkr):

- data.py            typed nodes CandidatesData (the search space) and ObservationsData (what was measured)
- calcfunctions.py   candidates_from_file / candidates_from_grid / observe / propose / evaluate_test_function
- workflows/         PhysboOptimizeWorkChain: a closed loop over a PHYSBO test function (runs in the daemon)
- query/             read-only queries (status, history, proposal, results, ...)
- cli/               `physbo-aiida [--json] <sub>`; the table of subcommands is cli/spec.py
- mcp/               `physbo-mcp`: MCP tools that only run the CLI with --json (no aiida import)
"""
__version__ = "0.4.2"
