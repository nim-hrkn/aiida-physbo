#!/usr/bin/env python3
"""One-dimensional Bayesian optimization, discrete or range, drawn step by step.

    python examples/one_dimensional.py discrete [--steps 6] [--score EI]
    python examples/one_dimensional.py range    [--steps 6] [--score EI] [--optimizer odatse --odatse-algorithm minsearch]

Objective: the Forrester function f(x) = (6x - 2)^2 sin(12x - 4) on [0, 1], minimized (minimum -6.02 at x = 0.757).
The loop is driven through the `physbo-aiida --json` CLI only (no aiida import here), so every step is a node in
the AiiDA profile; each Bayesian step is run with --posterior, and `physbo-aiida posterior` returns the posterior
mean / std / acquisition on the 101 candidates (discrete) or on a 101-point grid over the box (range).
The figure `one_dimensional_<space>.png` shows, per step, the true f, the observations, the posterior band, the
acquisition and the proposed point. Needs physbo-aiida on PATH and matplotlib.
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def f(x):
    x = np.asarray(x, dtype=float)
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def cli(*args):
    proc = subprocess.run(["physbo-aiida", "--json", *args], capture_output=True, text=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if not out.get("ok"):
        sys.exit(f"physbo-aiida {' '.join(args)} failed: {out.get('error')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", choices=["discrete", "range"])
    ap.add_argument("--steps", type=int, default=6, help="Bayesian steps after the 3 random points")
    ap.add_argument("--score", default="EI")
    ap.add_argument("--optimizer", default=None, help="range: random | odatse")
    ap.add_argument("--odatse-algorithm", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="figure path (default one_dimensional_<space>.png)")
    args = ap.parse_args()
    label = f"example_1d_{args.space}"

    if args.space == "discrete":
        sp = cli("candidates", "--grid", json.dumps({"min": [0.0], "max": [1.0], "num": 101}), "--names", "x", "--label", label)
    else:
        sp = cli("search-box", "--min", "0", "--max", "1", "--names", "x", "--label", label)
    spk = sp["pk"]
    print(f"{args.space} space pk {spk}")

    def observe(prop, obs_pk):
        X = np.asarray(prop["X"])
        values = ",".join(str(v) for v in f(X[:, 0]))
        cmd = ["observe", "--space-pk", str(spk), "--values", values, "--label", label]
        if obs_pk:
            cmd += ["--observations-pk", str(obs_pk)]
        if args.space == "discrete":
            cmd += ["--actions", ",".join(map(str, prop["actions"]))]
        else:
            cmd += ["--x", json.dumps(X.tolist())]
        return cli(*cmd)["pk"]

    # 3 random points, then the Bayesian steps (each with the posterior stored)
    prop = cli("propose", "--space-pk", str(spk), "--num-search-each-probe", "3", "--seed", str(args.seed), "--label", label)
    obs_pk = observe(prop, None)
    steps = []
    for i in range(args.steps):
        cmd = ["propose", "--space-pk", str(spk), "--observations-pk", str(obs_pk), "--score", args.score, "--minimize",
               "--posterior", "--seed", str(args.seed + i + 1), "--label", label]
        if args.optimizer:
            cmd += ["--optimizer", args.optimizer]
        if args.odatse_algorithm:
            cmd += ["--odatse-algorithm", args.odatse_algorithm]
        prop = cli(*cmd)
        steps.append(cli("posterior", "--pk", str(prop["process_pk"])))
        x_new = prop["X"][0][0]
        print(f"step {i + 1}: proposed x={x_new:.4f} f={f(x_new):.4f}  best so far {prop['summary']['best_so_far']['best_value']:.4f}")
        obs_pk = observe(prop, obs_pk)
    hist = cli("history", "--pk", str(obs_pk), "--minimize")
    print(f"best f={hist['best']['best_value']:.4f} at x={hist['best']['best_X'][0]:.4f}  (true -6.0207 at 0.7572); observations pk {obs_pk}")

    # ---- figure: one column per step, posterior on top, acquisition below
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xx = np.linspace(0, 1, 401)
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, squeeze=False)
    for i, st in enumerate(steps):
        X = np.asarray(st["X"])[:, 0]
        order = np.argsort(X)
        X, mu, sd = X[order], np.asarray(st["fmean"])[order, 0], np.asarray(st["fstd"])[order, 0]
        xo = np.asarray(st["observations"]["X"])[:, 0]
        to = np.asarray(st["observations"]["t"])[:, 0]
        xp = st["proposal"]["X"][0][0]
        ax = axes[0, i]
        ax.plot(xx, f(xx), "k:", lw=1, label="true f")
        ax.fill_between(X, mu - 2 * sd, mu + 2 * sd, color="tab:blue", alpha=0.15)
        ax.plot(X, mu, color="tab:blue", lw=1.5, label="posterior mean ± 2σ")
        ax.plot(xo, to, "ko", ms=4, label="observed")
        ax.axvline(xp, color="tab:red", ls="--", lw=1.2, label="proposed")
        ax.set_title(f"step {i + 1}: {len(xo)} obs", fontsize=10)
        ax.set_ylim(-10, 18)
        if i == 0:
            ax.set_ylabel("f(x)")
            ax.legend(fontsize=7, loc="upper left")
        ax2 = axes[1, i]
        if st.get("score") is not None:
            sc = np.asarray(st["score"])[order]
            ax2.plot(X, sc, color="tab:green", lw=1.2)
            ax2.fill_between(X, sc.min(), sc, color="tab:green", alpha=0.15)
        ax2.axvline(xp, color="tab:red", ls="--", lw=1.2)
        ax2.set_xlabel("x")
        if i == 0:
            ax2.set_ylabel(f"acquisition ({args.score})")
    fig.suptitle(f"PHYSBO {args.space} policy on the Forrester function (aiida-physbo, observations pk {obs_pk})", fontsize=11)
    fig.tight_layout()
    out = args.out or f"one_dimensional_{args.space}.png"
    fig.savefig(out, dpi=120)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
