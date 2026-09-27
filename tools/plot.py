#!/usr/bin/env python3
"""Plots from results/published into docs/figures (run tools/analyze.py first)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 8,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

COLOR = {"gpu-frontier": "C0", "gpu-nearfar": "C1", "gpu-topo": "C3", "gpu-edge": "C2",
         "delta-omp": "C4", "dial": "C5", "dijkstra": "C7"}
times = FuncFormatter(lambda v, _: f"{v:g}x")


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def log2_axes(ax, x=True, y=True):
    if x:
        ax.set_xscale("log", base=2)
    if y:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(times)


def speedup_vs_parallelism(d, out):
    pg = d["per_graph"]
    a = pg[(pg.experiment == "A") & ~pg.label.str.contains("natural|k=16")]
    unit = a[a.label.str.contains(" unit")].sort_values("Pi")
    w100 = a[a.label.str.contains(r"u\[1,100\]")].sort_values("Pi")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(unit.Pi, 2 ** unit.log2_Sinf, "o-", color="C7", label="layered, unit weights")
    err = [2 ** w100.log2_Sinf - 2 ** w100.log2_Sinf_lo, 2 ** w100.log2_Sinf_hi - 2 ** w100.log2_Sinf]
    ax.errorbar(w100.Pi, 2 ** w100.log2_Sinf, yerr=err, fmt="s-", color="C0", capsize=2,
                label="layered, weights 1-100")
    ax.plot(w100.Pi, 2 ** w100.log2_S1, "s--", color="C0", mfc="none", label="same, one-shot (incl. upload)")
    ax.plot(w100.Pi, 2 ** w100.log2_Sinf_model, ":", color="k", label="model, no wasted work")

    ho = pg[pg.experiment == "HO"]
    marks = {"grid": "D", "geometric": "^", "uniform": "v", "rmat": "P"}
    for _, r in ho.iterrows():
        name = r.topology + (" (row-major)" if "natural" in r.label else "")
        mk = "d" if "natural" in r.label else marks.get(r.topology, "o")
        ax.plot(r.Pi, 2 ** r.log2_Sinf, mk, color="C3", mfc="none", ms=7, label=name)

    ax.axhline(1, color="k", lw=0.8)
    log2_axes(ax)
    ax.set_xlabel("useful parallelism Π (useful arcs / shortest-path hop depth)")
    ax.set_ylabel("speedup, GPU / CPU (see docs/results.md)")
    ax.set_title("GPU vs CPU speedup, n = 2^20")
    handles, labels = ax.get_legend_handles_labels()
    first = labels.index("layered, weights 1-100")
    order = [first] + [i for i in range(len(labels)) if i != first]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right",
              title="red: held-out graphs", title_fontsize=8)
    fig.savefig(out / "speedup_vs_parallelism.png")
    plt.close(fig)


def sync_cost(d, out):
    a = d["sync_cost"]
    a = a[(a.relabel == 1) & (a.layer_links == 4)]
    alpha = d["constants"].get("alpha_frontier_us")
    fig, (l, r) = plt.subplots(1, 2, figsize=(10, 4))
    for solver in ["gpu-frontier", "gpu-nearfar"]:
        for w, style in [("unit", "-"), ("u[1,100]", "--")]:
            s = a[(a.solver == solver) & (a.wlabel == w)]
            g = s.groupby("layer_width").median(numeric_only=True).sort_values("bfs_frontier_wmean")
            lab = f"{solver}, {'unit' if w == 'unit' else '1-100'}"
            l.plot(g.bfs_frontier_wmean, g.solve_ms, "o" + style, color=COLOR[solver],
                   mfc="none" if w != "unit" else None, label=lab)
            r.plot(g.work_per_sync, g.us_per_sync, "o", color=COLOR[solver],
                   mfc="none" if w != "unit" else None, label=lab)
    if alpha:
        r.axhline(alpha, color="k", lw=0.8, ls=":", label=f"empty round, calibration ({alpha:.1f} µs)")
    for ax in (l, r):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
    l.set_xlabel("BFS frontier width")
    l.set_ylabel("solve time (ms)")
    l.set_title("(a) same frontier width, different time")
    r.set_xlabel("edges touched per synchronization")
    r.set_ylabel("time per synchronization (µs)")
    r.set_title("(b) per synchronization")
    fig.savefig(out / "sync_cost.png")
    plt.close(fig)


def skew(d, out):
    b = d["skew"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for solver in ["gpu-topo", "gpu-frontier"]:
        hub = b[(b.topology == "hub") & (b.solver == solver)]
        g = hub.groupby("deg_max").agg(ratio=("ratio", gmean), rho=("rho", "median")).reset_index()
        rm = b[(b.topology == "rmat") & (b.solver == solver)]
        rg = rm.groupby("relabel").agg(ratio=("ratio", gmean), rho=("rho", "median"),
                                       deg_max=("deg_max", "first"))
        for ax, x in zip(axes, ["deg_max", "rho"]):
            ax.plot(g[x], g.ratio, "o-", color=COLOR[solver], label=f"{solver} (planted hub)")
            ax.plot(rg[x], rg.ratio, "P", color=COLOR[solver], mfc="none", ms=8, label=f"{solver} (RMAT)")
    for ax in axes:
        ax.axhline(1, color="k", lw=0.8)
        log2_axes(ax)
    axes[1].axvline(1, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("largest degree")
    axes[0].set_ylabel("solve time relative to gpu-edge")
    axes[0].set_title("(a) by largest degree")
    axes[1].set_xlabel("span index ρ")
    axes[1].set_title("(b) by span index")
    axes[0].legend()
    fig.savefig(out / "skew.png")
    plt.close(fig)


def weights(d, out):
    c = d["weights"]
    order = ["unit", "logD1", "logD2", "logD4", "logD6"]
    ticks = ["unit", "1-10", "1-10²", "1-10⁴", "1-10⁶"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, topo, title in [(axes[0], "uniform", "(a) uniform, n = 2^20"),
                            (axes[1], "grid", "(b) 1024 x 1024 grid")]:
        g = c[c.topology == topo].groupby("wlabel").agg(
            best=("time_ratio_best_delta", gmean), heur=("time_ratio_heuristic", gmean),
            work=("work_ratio", gmean)).reindex(order)
        x = np.arange(len(order))
        ax.plot(x, g.best, "o-", color="C0", label="time ratio, best Δ")
        ax.plot(x, g.heur, "o--", color="C0", mfc="none", label="time ratio, default Δ")
        ax.plot(x, g.work, "s-", color="C7", label="edges-touched ratio")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(times)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("log-uniform weight range")
        ax.set_title(title)
    axes[0].set_ylabel("gpu-frontier / gpu-nearfar")
    axes[0].legend()
    fig.savefig(out / "weights.png")
    plt.close(fig)


def scaling(d, out):
    pg = d["per_graph"]
    e = d["scaling"]
    fig, (l, r) = plt.subplots(1, 2, figsize=(10, 4))
    for topo, color in [("uniform", "C0"), ("grid", "C3")]:
        s = pg[(pg.experiment == "E") & (pg.topology == topo)].sort_values("n")
        err = [2 ** s.log2_Sinf - 2 ** s.log2_Sinf_lo, 2 ** s.log2_Sinf_hi - 2 ** s.log2_Sinf]
        l.errorbar(s.n, 2 ** s.log2_Sinf, yerr=err, fmt="o-", color=color, capsize=2, label=topo)
    l.axhline(1, color="k", lw=0.8)
    log2_axes(l)
    l.set_xlabel("vertices")
    l.set_ylabel("resident speedup, GPU / CPU")
    l.set_title("(a) speedup vs size")
    l.legend()

    u = e[(e.topology == "uniform") & (e.run_cache == "warm")]
    for solver in ["gpu-frontier", "delta-omp", "dial"]:
        g = u[u.solver == solver].groupby("working_set_bytes").ns_per_touched_edge.median()
        r.plot(g.index / 2 ** 20, g.values, "o-", color=COLOR[solver], label=solver)
    for mb, name in [(36, "CPU L3 (36 MB)"), (72, "GPU L2 (72 MB)")]:
        r.axvline(mb, color="k", lw=0.8, ls=":")
        r.text(mb * 1.05, 0.5, name, rotation=90, va="center", fontsize=8,
               transform=r.get_xaxis_transform())
    r.set_xscale("log", base=2)
    r.set_yscale("log")
    r.set_xlabel("working set (MB)")
    r.set_ylabel("ns per touched edge")
    r.set_title("(b) cost per edge, uniform graphs")
    r.legend(loc="lower right")
    fig.savefig(out / "scaling.png")
    plt.close(fig)


def time_breakdown(d, out):
    t = d["time_breakdown"]
    cpu = t[t.device == "cpu"].groupby("graph_id").wall_ms.min()
    keep = t.solver.isin(["gpu-frontier", "gpu-nearfar"]) | (t.label.str.startswith("hub") & (t.solver == "gpu-topo"))
    g = t[(t.device == "gpu") & keep].sort_values(["label", "solver"]).reset_index(drop=True)
    names = [f"{lab.replace(' u[1,100]', '')}, {s}" for lab, s in zip(g.label, g.solver)]
    other = (g.wall_ms - g.setup_transfer_ms - g.solve_ms).clip(lower=0)
    parts = [("setup + transfer", g.setup_transfer_ms, "C7"),
             ("gaps between kernels", g.sync_gap_ms.clip(lower=0), "C1"),
             ("kernels", g.kernel_ms, "C0"),
             ("other host work", other, "C8")]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(g) + 1.2))
    left = np.zeros(len(g))
    for name, ms, color in parts:
        share = (ms / g.wall_ms).values
        ax.barh(names, share, left=left, color=color, label=name)
        left += share
    for i, r in g.iterrows():
        ax.text(1.02, i, f"{r.wall_ms:.1f} ms (CPU {cpu[r.graph_id]:.1f})", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of one-shot GPU time")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.savefig(out / "time_breakdown.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "results" / "published", type=Path)
    ap.add_argument("--out", default=ROOT / "docs" / "figures", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = {f.stem: pd.read_csv(f) for f in args.data.glob("*.csv")}
    const = d.pop("constants")
    d["constants"] = dict(zip(const.iloc[:, 0], const.iloc[:, 1]))
    for plot in [speedup_vs_parallelism, sync_cost, skew, weights, scaling, time_breakdown]:
        plot(d, args.out)
        print(f"wrote {plot.__name__}.png")


if __name__ == "__main__":
    main()
