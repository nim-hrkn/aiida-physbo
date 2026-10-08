Layer 1 (`test_spec_and_mcp.py`): no AiiDA profile needed; structure of the CLI spec and the MCP server.

Layer 2 (`test_calcfunctions.py`, `test_workchain.py`): run on a temporary sqlite profile created by
aiida-core's pytest fixtures; nothing touches the real profile:

    python -m pytest -p aiida.tools.pytest_fixtures tests

`pip install -e .` must have been done (the entry points are needed for the typed nodes and the WorkChain).
