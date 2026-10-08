"""Layer-1 tests: no aiida profile needed. Structure of the CLI spec and the MCP server."""
import inspect
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_PY = os.path.join(HERE, "..", "aiida_physbo", "mcp", "server.py")


def test_mcp_server_does_not_import_aiida_or_physbo():
    code = ("import sys; import aiida_physbo.mcp.server as s; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('aiida', 'physbo', 'numpy')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", out


def test_no_verdi_in_server():
    assert "verdi" not in open(SERVER_PY).read()


def test_allowlist_is_the_spec():
    from aiida_physbo.cli.spec import SUBCOMMANDS
    from aiida_physbo.mcp import server

    assert server.ALLOWED_SUBCOMMANDS == frozenset(SUBCOMMANDS)
    assert set(server.TOOLS.values()) == set(SUBCOMMANDS), "every subcommand has exactly one tool"
    assert len(server.TOOLS) == len(SUBCOMMANDS)
    assert server.ALLOWED_BINARY_NAMES == ("physbo-aiida",)
    assert all(name.startswith("physbo_") for name in server.TOOLS)


def test_more_than_ten_tools_and_flags_filter():
    from aiida_physbo.mcp import server

    assert len(server.TOOLS) > 10
    read_only = server.tool_functions()
    assert "physbo_propose" not in read_only and "physbo_kill" not in read_only and "physbo_status" in read_only
    assert "physbo_history" in read_only
    assert "physbo_propose" in server.tool_functions(allow_submit=True)
    assert "physbo_submit_optimize" in server.tool_functions(allow_submit=True)
    assert "physbo_kill" in server.tool_functions(allow_control=True)
    assert "physbo_kill" not in server.tool_functions(allow_submit=True)


def test_every_tool_argument_reaches_argv(monkeypatch):
    """the failure to guard against: a tool argument that is silently dropped from argv."""
    from aiida_physbo.cli.spec import SUBCOMMANDS, option_flag
    from aiida_physbo.mcp import server

    monkeypatch.setenv("PHYSBO_AIIDA_BIN", "/x/physbo-aiida")
    for tool, sub in server.TOOLS.items():
        fn = getattr(server, tool)
        params = [p for p in inspect.signature(fn).parameters if p != "self"]
        options = SUBCOMMANDS[sub]["options"]
        assert set(params) == set(options), f"{tool}: parameters {params} != spec options {list(options)}"
        sample = {}
        for name, (typ, _req, _help) in options.items():
            sample[name] = {"int": 7, "float": 1.5, "str": "x", "bool": True}[typ]
        argv = server.build_argv(sub, sample)
        assert argv[0] == "/x/physbo-aiida" and argv[1] == "--json" and sub in argv
        for name, (typ, _req, _help) in options.items():
            if typ == "bool":
                assert option_flag(name) in argv, f"{tool}: {name} not passed"
            else:
                assert f"{option_flag(name)}={sample[name]}" in argv, f"{tool}: {name} not passed"


def test_dict_arguments_are_passed_as_json(monkeypatch):
    from aiida_physbo.mcp import server

    monkeypatch.setenv("PHYSBO_AIIDA_BIN", "/x/physbo-aiida")
    argv = server.build_argv("candidates", {"grid": {"min": [0], "max": [1], "num": 5}})
    assert argv[-1] == "--grid=" + json.dumps({"min": [0], "max": [1], "num": 5})
    argv = server.build_argv("observe", {"space_pk": 3, "actions": "1,2", "values": [[0.1, 0.2], [0.3, 0.4]]})
    assert "--values=[[0.1, 0.2], [0.3, 0.4]]" in argv


def test_negative_values_are_not_options():
    """`observe --values -4.0,-8.0` was read by argparse as a missing argument (2026-10-08, first real run)."""
    from aiida_physbo.cli.main import build_parser, join_negative_values
    from aiida_physbo.mcp import server

    argv = ["--json", "observe", "--space-pk", "3", "--actions", "1,2", "--values", "-4.0,-8.0", "--label", "-x"]
    joined = join_negative_values(argv)
    assert "--values=-4.0,-8.0" in joined and "--label=-x" in joined
    args = build_parser().parse_args(joined)
    assert args.values == "-4.0,-8.0" and args.label == "-x" and args.space_pk == 3
    # a real flag after a value option is left alone
    assert join_negative_values(["propose", "--space-pk", "3", "--minimize"]) == ["propose", "--space-pk", "3", "--minimize"]
    # the MCP side never produces the two-token form
    argv = server.build_argv("observe", {"space_pk": 3, "actions": "1", "values": "-4.0"})
    assert "--values=-4.0" in argv


def test_unknown_subcommand_or_argument_rejected(monkeypatch):
    from aiida_physbo.mcp import server

    monkeypatch.setenv("PHYSBO_AIIDA_BIN", "/x/physbo-aiida")
    with pytest.raises(ValueError):
        server.build_argv("node-delete", {})
    with pytest.raises(ValueError):
        server.build_argv("status", {"pk": 1})


def test_binary_name_guard(monkeypatch):
    from aiida_physbo.mcp import server

    monkeypatch.setenv("PHYSBO_AIIDA_BIN", "/usr/bin/verdi")
    with pytest.raises(RuntimeError):
        server.binary()


def test_cli_parser_builds_and_json_failure_is_json():
    from aiida_physbo.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["--json", "propose", "--space-pk", "5", "--minimize", "--score", "EI"])
    assert args.command == "propose" and args.space_pk == 5 and args.minimize and args.score == "EI"
    proc = subprocess.run([sys.executable, "-m", "aiida_physbo.cli.main", "--json", "--profile", "no-such-profile", "status"],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ok"] is False and "error" in out


def test_propose_parameters_match_defaults_table():
    """the CLI builds the propose parameters only from keys that propose knows."""
    from aiida_physbo.calcfunctions import PROPOSE_DEFAULTS
    from aiida_physbo.cli import steps

    p = steps._propose_parameters(score="EI", num_rand_basis=100, num_search_each_probe=2, seed=1, minimize=True,
                                  random=True, posterior=True, interval=3)
    assert p == {"score": "EI", "num_rand_basis": 100, "num_search_each_probe": 2, "seed": 1, "interval": 3,
                 "maximize": False, "random": True, "posterior": True}
    assert set(p) <= set(PROPOSE_DEFAULTS)
    assert steps._propose_parameters() == {}


def test_mcp_server_builds_with_mcp_2():
    """the tool definitions are accepted by the mcp 2.x server (import of the package is enough)."""
    mcp = pytest.importorskip("mcp")
    from aiida_physbo.mcp import server

    srv = server.build_server(allow_submit=True, allow_control=True)
    import asyncio

    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert names == set(server.TOOLS), names - set(server.TOOLS)


def test_values_and_coordinates_row_parsing():
    """`--values -0.9,-4.6` is M single-objective rows; `--x 0.3,-1.2` is one point (2026-10-08, range smoke test)."""
    from aiida_physbo.cli.steps import _parse_values, _rows

    assert _parse_values("-0.9,-4.6,1.0").shape == (3, 1)
    assert _parse_values("0.1,2.0;0.3,1.5").tolist() == [[0.1, 2.0], [0.3, 1.5]]
    assert _parse_values("[[0.1, 2.0], [0.3, 1.5]]").shape == (2, 2)
    assert _parse_values([[1.0, 2.0]]).shape == (1, 2)
    assert _rows("0.3,-1.2", "--x").tolist() == [[0.3, -1.2]]
    assert _rows("0.3,-1.2;1.1,0.4", "--x").shape == (2, 2)
    assert _rows([[0.3, -1.2], [1.1, 0.4]], "--x").shape == (2, 2)
    assert _rows("[[0.3,-1.2]]", "--x").shape == (1, 2)
