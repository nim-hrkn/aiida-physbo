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
