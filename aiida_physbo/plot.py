# Copyright (c) 2026 Hiori Kino.
# Distributed under the terms of the Apache License, Version 2.0.
"""Figures of a campaign: best-so-far vs step, observed values, Pareto front (2 objectives). matplotlib is optional."""
import os

import numpy as np


def plot_cli(pk, outdir=None, prefix=None, minimize=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .calcfunctions import _nondominated_mask, best_sequence
    from .query.nodes import _node, _observations_of

    node = _node(pk)
    try:
        step = _posterior_step(node)          # a propose step run with posterior=true
    except ValueError:
        step = None
    if step is not None and step["dim"] == 1:
        return _plot_posterior_1d(step, outdir, prefix, minimize)
    obs = _observations_of(node)
    outdir = outdir or os.path.join(os.path.expanduser("~"), "aiida_work", "figures", str(obs.pk))
    os.makedirs(outdir, exist_ok=True)
    prefix = prefix or (obs.label or f"physbo_{obs.pk}")
    t = obs.t
    X = obs.X
    steps = np.arange(1, t.shape[0] + 1)
    files = []
    if X is not None and X.shape[1] == 2 and t.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sc = ax.scatter(X[:, 0], X[:, 1], c=t[:, 0], cmap="viridis", s=30)
        for i, (x, y) in enumerate(X):
            ax.annotate(str(i + 1), (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
        fig.colorbar(sc, ax=ax, label="t")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title(f"observed points {obs.pk} (numbers: observation order)")
        path = os.path.join(outdir, f"{prefix}_points.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        files.append(path)
    if t.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, t[:, 0], "o", ms=4, alpha=0.6, label="observed")
        ax.plot(steps, best_sequence(t, not minimize), "-", lw=2, label="best so far")
        ax.set_xlabel("observation")
        ax.set_ylabel("t")
        ax.legend()
        ax.set_title(f"observations {obs.pk}")
        path = os.path.join(outdir, f"{prefix}_best.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        files.append(path)
    else:
        sign = -1.0 if minimize else 1.0
        mask = _nondominated_mask(sign * t)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(t[:, 0], t[:, 1], s=12, c=steps, cmap="viridis", label="observed (colour: step)")
        front = t[mask]
        order = np.argsort(front[:, 0])
        ax.plot(front[order, 0], front[order, 1], "r-o", ms=5, lw=1.5, label="Pareto front")
        ax.set_xlabel("t[0]")
        ax.set_ylabel("t[1]")
        ax.legend()
        ax.set_title(f"observations {obs.pk} ({t.shape[1]} objectives, first two shown)")
        path = os.path.join(outdir, f"{prefix}_pareto.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        files.append(path)
    return {"pk": obs.pk, "space": obs.space, "files": files, "num_observations": int(t.shape[0]), "num_objectives": int(t.shape[1])}


def _posterior_step(node):
    from .query.nodes import posterior

    return posterior(node.pk, max_points=100000)


def _plot_posterior_1d(step, outdir=None, prefix=None, minimize=False):
    """one propose step on a 1-D space: posterior mean and 2-sigma band, observations, proposal, acquisition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pk = step["process_pk"]
    outdir = outdir or os.path.join(os.path.expanduser("~"), "aiida_work", "figures", str(pk))
    os.makedirs(outdir, exist_ok=True)
    prefix = prefix or f"propose_{pk}"
    X = np.asarray(step["X"])[:, 0]
    order = np.argsort(X)
    X = X[order]
    fmean = np.asarray(step["fmean"])[order]
    fstd = np.asarray(step["fstd"])[order]
    k = fmean.shape[1]
    nrows = k + (1 if step.get("score") is not None else 0)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 2.6 * nrows + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    obs = step.get("observations")
    prop = step.get("proposal", {})
    for j in range(k):
        ax = axes[j]
        ax.fill_between(X, fmean[:, j] - 2 * fstd[:, j], fmean[:, j] + 2 * fstd[:, j], color="tab:blue", alpha=0.15,
                        label="posterior mean ± 2σ")
        ax.plot(X, fmean[:, j], color="tab:blue", lw=1.5)
        if obs and obs.get("X") is not None:
            xo = np.asarray(obs["X"])[:, 0]
            to = np.asarray(obs["t"])[:, j]
            ax.plot(xo, to, "ko", ms=5, label=f"observed ({len(xo)})")
            for i, (x, t) in enumerate(zip(xo, to)):
                ax.annotate(str(i + 1), (x, t), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for xp in np.asarray(prop.get("X", []))[:, 0] if prop.get("X") else []:
            ax.axvline(xp, color="tab:red", ls="--", lw=1.2, label="proposed")
        ax.set_ylabel("t" if k == 1 else f"t[{j}]")
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8)
    summ = step.get("summary") or {}
    axes[0].set_title(f"propose {pk}: {summ.get('score')} after {summ.get('num_observed')} observations"
                      f" ({'minimize' if minimize or summ.get('maximize') is False else 'maximize'})")
    if step.get("score") is not None:
        ax = axes[-1]
        sc = np.asarray(step["score"])[order]
        ax.plot(X, sc, color="tab:green", lw=1.5)
        ax.fill_between(X, np.min(sc), sc, color="tab:green", alpha=0.15)
        for xp in np.asarray(prop.get("X", []))[:, 0] if prop.get("X") else []:
            ax.axvline(xp, color="tab:red", ls="--", lw=1.2)
        ax.set_ylabel(f"acquisition ({summ.get('score')})")
    axes[-1].set_xlabel("x")
    path = os.path.join(outdir, f"{prefix}_posterior.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"pk": pk, "space": summ.get("space"), "files": [path], "num_observations": summ.get("num_observed"),
            "proposed_X": prop.get("X")}
