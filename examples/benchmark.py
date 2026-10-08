#!/usr/bin/env python3
"""Run a set of benchmark functions through PhysboOptimizeWorkChain in the daemon and tabulate the results.

    python examples/benchmark.py [--set quick|full] [--out examples/figures/benchmark.png] [--label bench]

Each case is one WorkChain (submitted through `physbo-aiida --json`, no aiida import here). When all have
finished, prints a table (best found, known minimum, regret, evaluations) and draws the best-so-far curves.
Needs physbo-aiida on PATH, the daemon running, and matplotlib.
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

QUICK = [
    # name, kwargs, space, extra options, n_random, n_bayes
    ("Branin", {}, "discrete", {"num": 41, "score": "EI"}, 10, 25),
    ("Branin", {}, "range", {"score": "EI", "optimizer_nsamples": 2000}, 10, 25),
    ("SixHumpCamel", {}, "range", {"score": "EI", "optimizer_nsamples": 2000}, 10, 25),
    ("GoldsteinPrice", {}, "discrete", {"num": 41, "score": "EI"}, 10, 25),
    ("Levy", {"dim": 2}, "range", {"score": "EI", "optimizer_nsamples": 2000}, 10, 30),
    ("Hartmann3", {}, "range", {"score": "EI", "optimizer_nsamples": 3000}, 10, 30),
    ("Hartmann6", {}, "range", {"score": "EI", "optimizer_nsamples": 5000}, 15, 45),
    ("Hartmann6", {}, "range", {"score": "EI", "optimizer": "odatse", "odatse_algorithm": "exchange",
                               "odatse_params": json.dumps({"exchange": {"numsteps": 300}})}, 15, 45),
    ("Forrester", {}, "range", {"score": "EI", "optimizer_nsamples": 1000, "noise": 0.5}, 3, 12),
    ("GramacyLee", {}, "discrete", {"num": 201, "score": "EI"}, 3, 15),
    ("DTLZ2", {"nobj": 2, "dim": 3}, "range", {"score": "HVPI", "optimizer_nsamples": 2000}, 10, 20),
]
FULL = QUICK + [
    ("Levy", {"dim": 4}, "range", {"score": "EI", "optimizer_nsamples": 4000}, 15, 45),
    ("Rastrigin", {"dim": 2}, "discrete", {"num": 51, "score": "TS", "num_rand_basis": 500}, 10, 40),
    ("Ackley", {"dim": 2}, "range", {"score": "EI", "optimizer_nsamples": 3000}, 10, 40),
    ("DTLZ2", {"nobj": 3, "dim": 4}, "range", {"score": "HVPI", "optimizer_nsamples": 2000}, 12, 24),
]


def cli(*args):
    proc = subprocess.run(["physbo-aiida", "--json", *args], capture_output=True, text=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if not out.get("ok"):
        sys.exit(f"physbo-aiida {' '.join(args)} failed: {out.get('error')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["quick", "full"], default="quick")
    ap.add_argument("--out", default="benchmark.png")
    ap.add_argument("--label", default="bench")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    cases = QUICK if args.set == "quick" else FULL

    submitted = []
    for i, (name, kwargs, space, opts, n_random, n_bayes) in enumerate(cases):
        tag = f"{args.label}_{name}_{space}_{i}"
        cmd = ["submit-optimize", "--test-function", name, "--kwargs", json.dumps(kwargs), "--space", space,
               "--num-random", str(n_random), "--num-bayes", str(n_bayes), "--seed", str(args.seed), "--label", tag]
        for k, v in opts.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        s = cli(*cmd)
        submitted.append((tag, name, kwargs, space, opts, s["pk"], s.get("known_minimum")))
        print(f"submitted {tag}: pk {s['pk']}")

    pending = {pk for *_, pk, _ in submitted}
    t0 = time.time()
    while pending and time.time() - t0 < 3600:
        for pk in sorted(pending):
            w = cli("wait", "--pk", str(pk), "--wait-seconds", "20")
            if w["terminated"]:
                pending.discard(pk)
                print(f"  pk {pk}: {w['state']} (exit {w['exit_status']})  [{time.time() - t0:.0f} s]")
    if pending:
        print(f"still running: {sorted(pending)}; results below are partial")

    rows, curves = [], []
    for tag, name, kwargs, space, opts, pk, km in submitted:
        r = cli("results", "--pk", str(pk))
        summ = r.get("summary") or {}
        best = summ.get("best") or {}
        h = cli("history", "--pk", str(pk), "--minimize") if r.get("observations_pk") else None
        row = {"case": tag, "function": name, "kwargs": kwargs, "space": space, "score": opts.get("score"),
               "optimizer": opts.get("optimizer", "random" if space == "range" else "-"), "pk": pk,
               "state": r.get("state"), "exit": r.get("exit_status"), "evaluations": best.get("num_observations"),
               "best": best.get("best_value"), "best_X": best.get("best_X"), "f_star": (km or [None])[0],
               "regret": (r.get("known_minimum") or {}).get("regret"), "pareto_size": best.get("pareto_size"),
               "noise": opts.get("noise", 0)}
        rows.append(row)
        if h and h.get("best_sequence"):
            curves.append((tag, np.asarray(h["best_sequence"]), row["f_star"]))

    print("\n| case | space | score / optimizer | evals | best | f* | regret |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        best = "-" if r["best"] is None else f"{r['best']:.4f}"
        fs = "-" if r["f_star"] is None else f"{r['f_star']:.4f}"
        rg = "-" if r["regret"] is None else f"{r['regret']:.4f}"
        extra = f" (Pareto {r['pareto_size']})" if r["pareto_size"] else ""
        print(f"| {r['case']} | {r['space']} | {r['score']} / {r['optimizer']} | {r['evaluations']} | {best}{extra} | {fs} | {rg} |")
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(rows, f, indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(curves)
    cols = 4
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (tag, seq, f_star) in zip(axes.flat, curves):
        steps = np.arange(1, len(seq) + 1)
        if f_star is not None:
            ax.semilogy(steps, np.maximum(seq - f_star, 1e-6), "o-", ms=3, lw=1.2)
            ax.set_ylabel("best f − f*")
        else:
            ax.plot(steps, seq, "o-", ms=3, lw=1.2)
            ax.set_ylabel("best f")
        ax.set_title(tag.replace(args.label + "_", ""), fontsize=9)
        ax.set_xlabel("evaluation")
        ax.grid(alpha=0.3)
    fig.suptitle(f"aiida-physbo benchmark ({args.set}): best-so-far regret", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"figure: {args.out}; table: {args.out.rsplit('.', 1)[0] + '.json'}")


if __name__ == "__main__":
    main()
