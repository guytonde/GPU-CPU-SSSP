#!/usr/bin/env python3
"""Check the raw results and write the tables in results/published.

    tools/analyze.py [--raw results/raw] [--out results/published]
"""
import argparse
import glob
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
GPU = ["gpu-frontier", "gpu-nearfar"]
CPU = ["dijkstra", "dial", "delta", "delta-omp"]
FULL = ["A", "C", "E", "HO"]  # experiments with every main CPU and GPU solver
MODELS = {"syncs + edges": ["host_syncs", "edges_touched"],
          "syncs + vertices + edges": ["host_syncs", "edges_touched", "vertices_expanded"]}


# loading

def read_all(raw, name):
    frames = [pd.read_csv(f, low_memory=False)
              for f in sorted(glob.glob(str(raw / "**" / f"{name}.csv"), recursive=True))
              if Path(f).stat().st_size > 0]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def weight_label(r):
    if r.wdist == "unit":
        return "unit"
    if r.wdist == "logunif":
        return f"logD{int(float(r.decades))}"
    return f"u[{int(r.wmin)},{int(r.wmax)}]"


def graph_label(r):
    w = weight_label(r)
    natural = "" if int(r.relabel) == 1 else " natural"
    if r.topology == "layered":
        k = "" if int(r.layer_links) == 4 else f" k={int(r.layer_links)}"
        return f"layered W={int(r.layer_width)}{k}{natural} {w}"
    if r.topology == "hub":
        return f"hub h={r.hub_fraction:.3g} {w}"
    if r.topology == "grid":
        return f"grid {int(r.rows)}x{int(r.cols)}{natural} {w}"
    return f"{r.topology} n={int(r.n)}{natural} {w}"


def load(raw):
    runs, graphs, sources = read_all(raw, "runs"), read_all(raw, "graphs"), read_all(raw, "sources")
    samples, batch, calib = read_all(raw, "samples"), read_all(raw, "batch"), read_all(raw, "calibration")
    if samples.empty:
        sys.exit(f"no samples under {raw}")
    graphs = graphs.merge(runs[["run_id", "experiment"]], on="run_id", how="left")
    graphs["label"] = graphs.apply(graph_label, axis=1)
    graphs["wlabel"] = graphs.apply(weight_label, axis=1)
    info = runs[["run_id", "experiment", "delta_scale", "cache_state", "mode"]]
    samples = samples.merge(info.rename(columns={"cache_state": "run_cache", "mode": "run_mode"}),
                            on="run_id", how="left")
    samples["delta_scale"] = samples.delta_scale.fillna(0.0)
    samples["run_cache"] = samples.run_cache.fillna("warm")
    gpu = samples.device == "gpu"
    # Per query cost with the graph on the GPU, and the one time cost of putting it there.
    samples["q_ms"] = np.where(gpu, samples.solve_ms + samples.d2h_ms, samples.wall_ms)
    samples["U_ms"] = np.where(gpu, samples.wall_ms - samples.q_ms, 0.0)
    # Host work inside run() that no phase timer covers.
    phases = samples[["prep_ms", "alloc_ms", "h2d_ms", "solve_ms", "d2h_ms", "free_ms"]].sum(axis=1)
    samples["untimed_ms"] = samples.wall_ms - phases
    return runs, graphs, sources, samples, batch, calib


# validation

def validate(runs, graphs, sources, samples):
    """Returns rows of (check, status, detail). FAIL stops publication, WARN does not."""
    rows = []

    def check(name, ok, detail, hard=True):
        rows.append((name, "PASS" if ok else ("FAIL" if hard else "WARN"), detail))

    timed = samples[samples.warmup == 0]
    orphans = set(samples.run_id) - set(runs.run_id)
    check("every sample belongs to a run", not orphans, f"{len(orphans)} unknown runs")
    known = set(zip(sources.graph_id, sources.source))
    missing = sum(k not in known for k in zip(samples.graph_id, samples.source))
    check("every sample has source statistics", missing == 0, f"{missing} missing")
    wrong = samples[samples.mismatches != 0]
    check("every distance array matches the reference", wrong.empty,
          f"{len(wrong)} of {len(samples)} wrong")

    g = graphs.drop_duplicates("graph_id")
    shape = ["topology", "n", "m", "layer_width", "layer_links", "rows", "cols",
             "hub_fraction", "relabel", "seed"]
    groups = g.groupby([g[c].astype(str) for c in shape])
    varied = groups.wlabel.nunique() > 1
    broken = (groups.topo_hash.nunique()[varied] != 1).sum()
    check("changing weights never changed the topology", broken == 0,
          f"{varied.sum()} groups checked, {broken} broken")

    s = sources.merge(g[["graph_id", "topology", "n", "rows", "cols", "layer_width"]], on="graph_id")
    lay = s[s.topology == "layered"]
    layers = np.ceil(lay.n / lay.layer_width)
    ok = (lay.bfs_levels >= layers) & (lay.bfs_levels <= layers + np.log(lay.layer_width) / np.log(4) + 6)
    check("layered graphs have the depth their construction implies", ok.all(), f"{(~ok).sum()} off")
    grid = s[s.topology == "grid"]
    ok = grid.bfs_levels <= grid.rows + grid.cols - 1
    check("grid depth is at most rows + cols", ok.all(), f"{(~ok).sum()} off")
    geo = s[(s.topology == "geometric") & (s.n >= 1 << 20)]
    if len(geo):
        check("geometric graphs are deep (no spanning tree)", (geo.bfs_levels > 100).all(),
              f"smallest depth {geo.bfs_levels.min()}")

    phases = timed[["prep_ms", "alloc_ms", "h2d_ms", "solve_ms", "d2h_ms", "free_ms"]].sum(axis=1)
    low = timed[(phases < 0.9 * timed.wall_ms) & (timed.wall_ms > 1)]
    check("timed phases cover at least 90% of wall time", low.empty,
          f"{len(low)} runs below, from {', '.join(sorted(set(low.solver)))}", hard=False)
    gpu = timed[timed.device == "gpu"]
    shared = (gpu.gpu_foreign_procs > 0).sum()
    check("no other process was on the GPU", shared == 0, f"{shared} of {len(gpu)} GPU runs")
    clock = gpu.gpu_clock_mhz_measured.dropna()
    slow = (clock < 0.9 * clock.median()).sum()
    check("GPU at full clock before each run", slow == 0,
          f"median {clock.median():.0f} MHz, {slow} runs slower", hard=False)
    dirty = (runs.git_dirty.astype(str) == "1").sum()
    check("results come from committed code", dirty == 0,
          f"{dirty} of {len(runs)} runs, code hash {', '.join(sorted(set(runs.git_diff_hash.astype(str))))}",
          hard=False)
    dups = samples.duplicated(["run_id", "source", "solver", "session", "query_index", "rep", "warmup"]).sum()
    check("no duplicate samples", dups == 0, f"{dups} duplicates")
    return rows


# statistics

def boot_ci(values, stat, n=2000):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return stat(v) if len(v) else np.nan, np.nan, np.nan
    draws = np.random.default_rng(1).integers(0, len(v), size=(n, len(v)))
    boots = np.apply_along_axis(stat, 1, v[draws])
    return stat(v), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def kstar(kappa):
    return math.ceil(1 / kappa) if kappa > 0 else np.inf


# calibration

def constants(calib):
    k = {}
    rounds = calib[calib.bench == "round_overhead"].groupby("param").value.median()
    k["alpha_frontier_us"] = rounds.get("frontier_round")
    k["alpha_nearfar_us_per_sync"] = rounds.get("nearfar_inner") / 2
    k["alpha_sync_us"] = rounds.get("sync_only")
    for label in ["uniform_deg8", "grid_deg4"]:
        sweep = calib[calib.bench == f"frontier_round_{label}"]
        kernel = sweep[sweep.unit == "kernel_us"].set_index("param_value").value.sort_index()
        top = kernel.index >= kernel.index.max() / 16
        k[f"theta_{label}_edges_per_ns"] = 1e-3 / np.polyfit(kernel.index[top], kernel.values[top], 1)[0]
        verts = sweep[sweep.unit == "edges"].set_index("param_value").value.sort_index()
        vtop = verts.index >= verts.index.max() / 16
        per_vertex = np.polyfit(verts.index[vtop], kernel.reindex(verts.values).values[vtop], 1)[0]
        k[f"cost_per_vertex_{label}_ns"] = per_vertex * 1e3
    for name, v in calib[calib.bench == "hub_latency"].set_index("param").value.items():
        k[f"lambda_{name}_ns"] = v
    copy = calib[calib.bench == "copy"]
    for (name, size), ms in copy.groupby(["param", "param_value"]).value.median().items():
        if size in (2 ** 20, 2 ** 24, copy.param_value.max()):
            k[f"{name}_{round(size / 2 ** 20)}MB_GBps"] = size / (ms * 1e-3) / 1e9
    omp = calib[calib.bench == "omp_region"].groupby("param_value").value.median()
    k["omp_region_24_threads_us"] = omp.get(24)
    return k


# per source and per graph metrics

def per_source(med, sources, graphs, k):
    src = sources.drop_duplicates(["graph_id", "source"]).set_index(["graph_id", "source"])
    info = graphs.drop_duplicates(["experiment", "graph_id"]).set_index(["experiment", "graph_id"])
    rows = []
    for (exp, gid, source), grp in med[med.experiment.isin(FULL)].groupby(["experiment", "graph_id", "source"]):
        cpu, gpu = grp[grp.solver.isin(CPU)], grp[grp.solver.isin(GPU)]
        if cpu.empty or gpu.empty:
            continue
        st, g = src.loc[(gid, source)], info.loc[(exp, gid)]
        c = cpu.loc[cpu.q_ms.idxmin()]
        r = gpu.loc[gpu.q_ms.idxmin()]        # best GPU with the graph resident
        o = gpu.loc[gpu.wall_ms.idxmin()]     # best GPU including the upload
        depth, useful = max(1, st.sp_depth_max), max(1, st.m_reached)
        # The calibrated model with no wasted work.
        q_model = depth * k["alpha_frontier_us"] * 1e-3 + useful / k["theta_uniform_deg8_edges_per_ns"] * 1e-6
        rows.append({
            "experiment": exp, "graph_id": gid, "source": source, "label": g.label,
            "topology": g.topology, "n": g.n, "m": g.m, "m_reached": useful, "sp_depth": depth,
            "bfs_frontier_wmean": st.bfs_frontier_wmean, "Pi": useful / depth,
            "best_cpu": c.solver, "q_cpu_ms": c.q_ms, "best_gpu": r.solver, "q_gpu_ms": r.q_ms,
            "U_ms": r.U_ms, "S1": c.wall_ms / o.wall_ms, "Sinf": c.q_ms / r.q_ms,
            "kappa": (c.q_ms - r.q_ms) / r.U_ms if r.U_ms > 0 else np.nan,
            "upload_over_cpu_query": r.U_ms / c.q_ms,
            "iota": r.edges_touched / useful, "delta": r.host_syncs / depth,
            "c_cpu_ns": c.q_ms * 1e6 / useful, "host_syncs": r.host_syncs,
            "edges_touched": r.edges_touched, "vertices_expanded": r.vertices_expanded,
            "Sinf_model": c.q_ms / q_model,
        })
    return pd.DataFrame(rows)


def per_solver(med_all, sources, graphs):
    """Every solver on every graph, as the median over sources of the per source medians."""
    cols = ["q_ms", "wall_ms", "solve_ms", "untimed_ms", "host_syncs", "edges_touched",
            "vertices_expanded", "iota"]
    d = med_all[(med_all.delta_scale == 0) & (med_all.experiment != "F")]
    useful = sources.drop_duplicates(["graph_id", "source"])[["graph_id", "source", "m_reached"]]
    d = d.merge(useful, on=["graph_id", "source"])
    d["iota"] = d.edges_touched / d.m_reached.clip(lower=1)
    grp = d.groupby(["experiment", "graph_id", "solver", "run_cache"])
    out = grp[cols].median().join(grp.size().rename("sources")).reset_index()
    labels = graphs.drop_duplicates(["experiment", "graph_id"])[["experiment", "graph_id", "label"]]
    out = labels.merge(out, on=["experiment", "graph_id"])
    return out[["experiment", "label", "solver", "run_cache", "sources", *cols, "graph_id"]]


def per_graph(ps):
    rows = []
    for (exp, gid), grp in ps.groupby(["experiment", "graph_id"]):
        r = grp.iloc[0][["experiment", "graph_id", "label", "topology", "n", "m"]].to_dict()
        r["sources"] = len(grp)
        for col in ["S1", "Sinf"]:
            r[f"log2_{col}"], r[f"log2_{col}_lo"], r[f"log2_{col}_hi"] = boot_ci(np.log2(grp[col]), np.mean)
        r["kappa"], r["kappa_lo"], r["kappa_hi"] = boot_ci(grp.kappa, np.median)
        r["Kstar"] = kstar(r["kappa"])
        r["log2_Sinf_model"] = float(np.median(np.log2(grp.Sinf_model)))
        for col in ["Pi", "sp_depth", "bfs_frontier_wmean", "q_cpu_ms", "q_gpu_ms", "U_ms",
                    "upload_over_cpu_query", "iota", "delta", "c_cpu_ns"]:
            r[col] = grp[col].median()
        r["best_cpu"], r["best_gpu"] = grp.best_cpu.mode()[0], grp.best_gpu.mode()[0]
        rows.append(r)
    return pd.DataFrame(rows)


# model

def fit(rows, cols):
    """solve_ms as a nonnegative sum of counters, minimizing relative error."""
    coef, _ = nnls(rows[cols].values / rows.solve_ms.values[:, None], np.ones(len(rows)))
    return coef


def model_table(med, k):
    rows, fits = [], {}
    for solver in GPU:
        train = med[(med.experiment == "A") & (med.solver == solver)]
        for name, cols in MODELS.items():
            coef = fits[(solver, name)] = fit(train, cols)
            row = {"solver": solver, "model": name, "alpha_us": coef[0] * 1e3,
                   "ns_per_edge": coef[1] * 1e6,
                   "ns_per_vertex": coef[2] * 1e6 if len(coef) > 2 else np.nan}
            for exp in ["A", "B", "C", "E", "HO"]:
                test = med[(med.experiment == exp) & (med.solver == solver)]
                if len(test):
                    err = np.abs(test[cols].values @ coef - test.solve_ms) / test.solve_ms
                    row[f"error_{exp}"] = err.median()
            rows.append(row)
    rows.append({"solver": "gpu-frontier", "model": "calibration",
                 "alpha_us": k["alpha_frontier_us"], "ns_per_edge": 1 / k["theta_uniform_deg8_edges_per_ns"],
                 "ns_per_vertex": k["cost_per_vertex_uniform_deg8_ns"]})
    return pd.DataFrame(rows), fits


def device_choice(ps, k, fits):
    """How often simple rules pick the faster device on graphs not used to fit them.
    Regret is the time lost against always choosing correctly."""
    train, test = ps[ps.experiment == "A"], ps[ps.experiment.isin(["HO", "C", "E"])]
    best = np.minimum(test.q_cpu_ms, test.q_gpu_ms)
    truth = test.q_gpu_ms < test.q_cpu_ms

    def score(name, pick_gpu):
        chosen = np.where(pick_gpu, test.q_gpu_ms, test.q_cpu_ms)
        return {"rule": name, "accuracy": float(np.mean(pick_gpu == truth)),
                "regret": float((chosen - best).sum() / best.sum()), "cases": len(test)}

    slope = np.polyfit(np.log(train.bfs_frontier_wmean), np.log(train.Sinf), 1)
    q_gpu = test.sp_depth * k["alpha_frontier_us"] * 1e-3 + test.m_reached / k["theta_uniform_deg8_edges_per_ns"] * 1e-6
    q_cpu = test.m_reached * train.c_cpu_ns.median() * 1e-6
    cols = MODELS["syncs + vertices + edges"]
    q_run = np.array([r[cols].astype(float).values @ fits[(r.best_gpu, "syncs + vertices + edges")]
                      for _, r in test.iterrows()])
    return pd.DataFrame([
        score("always CPU", np.zeros(len(test), bool)),
        score("always GPU", np.ones(len(test), bool)),
        score("BFS frontier width, fitted", np.polyval(slope, np.log(test.bfs_frontier_wmean)) > 0),
        score("graph properties and calibrated constants", q_gpu < q_cpu),
        score("model on the run's own counters, true CPU time", q_run < test.q_cpu_ms),
    ])


def amortization(samples, batch, pg, graphs):
    """Break even query counts measured in D, predicted from one shot runs, and
    against the CPU batch."""
    d = samples[(samples.experiment == "D") & (samples.run_mode == "resident")]
    labels = graphs[graphs.experiment == "D"].drop_duplicates("graph_id").set_index("graph_id").label
    rows = []
    for gid, g in d.groupby("graph_id"):
        curves = {s: grp.groupby("query_index").wall_ms.median().cumsum() for s, grp in g.groupby("solver")}
        best_cpu = min((s for s in curves if s in CPU), key=lambda s: curves[s].iloc[-1])
        q_cpu = g[g.solver == best_cpu].wall_ms.median()
        b = batch[batch.graph_id == gid]
        q_batch = (b.wall_ms / b.batch_k).groupby(b.solver).median().min()
        pred = pg[(pg.graph_id == gid) & pg.experiment.isin(["A", "HO"])]
        for s in [s for s in curves if s in GPU]:
            wins = np.nonzero(curves[s].values <= curves[best_cpu].values)[0]
            q_gpu = g[(g.solver == s) & (g.query_index > 0)].wall_ms.median()
            upload = g[(g.solver == s) & (g.query_index == 0)].wall_ms.median() - q_gpu
            rows.append({"graph_id": gid, "label": labels[gid], "gpu_solver": s, "best_cpu": best_cpu,
                         "q_cpu_ms": q_cpu, "q_gpu_ms": q_gpu, "U_ms": upload,
                         "Kstar_measured": wins[0] + 1 if len(wins) else np.inf,
                         "Kstar_predicted": pred.Kstar.iloc[0] if len(pred) else np.nan,
                         "q_batch_ms": q_batch,
                         "Kstar_vs_batch": kstar((q_batch - q_gpu) / max(upload, 1e-9))})
    return pd.DataFrame(rows)


# data behind the plots

def plot_tables(out, med_all, ps, graphs, samples, k):
    def info(exp):
        return graphs[graphs.experiment == exp].drop_duplicates("graph_id").set_index("graph_id")

    med = med_all[(med_all.delta_scale == 0) & (med_all.run_cache == "warm")]

    a = med[(med.experiment == "A") & med.solver.isin(GPU)]
    a = a.merge(info("A")[["layer_width", "wlabel", "relabel", "layer_links"]], left_on="graph_id", right_index=True)
    a = a.merge(ps[ps.experiment == "A"][["graph_id", "source", "bfs_frontier_wmean"]], on=["graph_id", "source"])
    a["work_per_sync"] = a.edges_touched / a.host_syncs
    a["us_per_sync"] = a.solve_ms * 1e3 / a.host_syncs
    a.to_csv(out / "sync_cost.csv", index=False)

    # Time relative to gpu-edge. rho compares the hub walk with the rest of a round.
    edges_in_flight = k["lambda_gpu-topo_ns"] * k["theta_uniform_deg8_edges_per_ns"]
    g_b, rows = info("B"), []
    for (gid, source), grp in med[med.experiment == "B"].groupby(["graph_id", "source"]):
        t, g = grp.set_index("solver"), g_b.loc[gid]
        work_per_round = t.loc["gpu-edge", "edges_touched"] / max(1, t.loc["gpu-edge", "sync_rounds"])
        base = {"graph_id": gid, "source": source, "topology": g.topology, "relabel": g.relabel,
                "hub_fraction": g.hub_fraction, "deg_max": g.deg_max}
        for solver, lanes in [("gpu-topo", 1), ("gpu-frontier", 32)]:
            rows.append({**base, "solver": solver, "ratio": t.loc[solver, "solve_ms"] / t.loc["gpu-edge", "solve_ms"],
                         "rho": (g.deg_max / lanes) / (work_per_round / edges_in_flight)})
        rows.append({**base, "solver": "delta-omp/delta", "ratio": t.loc["delta-omp", "solve_ms"] / t.loc["delta", "solve_ms"],
                     "rho": np.nan})
    pd.DataFrame(rows).to_csv(out / "skew.csv", index=False)

    c = med_all[med_all.experiment == "C"].merge(info("C")[["topology", "wlabel"]], left_on="graph_id", right_index=True)
    rows = []
    for (gid, source), grp in c.groupby(["graph_id", "source"]):
        t = grp[grp.delta_scale == 0].set_index("solver")
        nearfar = grp[grp.solver == "gpu-nearfar"]
        best = nearfar.loc[nearfar.solve_ms.idxmin()]
        f, nf = t.loc["gpu-frontier"], t.loc["gpu-nearfar"]
        rows.append({"graph_id": gid, "source": source, "topology": grp.topology.iloc[0],
                     "wlabel": grp.wlabel.iloc[0], "time_ratio_heuristic": f.solve_ms / nf.solve_ms,
                     "time_ratio_best_delta": f.solve_ms / best.solve_ms, "best_delta_scale": best.delta_scale,
                     "work_ratio": f.edges_touched / nf.edges_touched, "sync_ratio": nf.host_syncs / f.host_syncs})
    pd.DataFrame(rows).to_csv(out / "weights.csv", index=False)

    e = med_all[(med_all.experiment == "E") & (med_all.delta_scale == 0)]
    e = e.merge(info("E")[["topology", "n", "m"]], left_on="graph_id", right_index=True)
    e["working_set_bytes"] = 4 * (e.n + 1) + 8 * e.m + 4 * e.n
    e["ns_per_touched_edge"] = e.solve_ms * 1e6 / e.edges_touched.clip(lower=1)
    e.to_csv(out / "scaling.csv", index=False)

    t = samples[(samples.experiment == "T") & (samples.instrumented == 1)].copy()
    t["setup_transfer_ms"] = t.prep_ms + t.alloc_ms + t.h2d_ms + t.d2h_ms + t.free_ms
    t["sync_gap_ms"] = t.solve_ms - t.kernel_ms
    t_sources = t[["graph_id", "solver", "source"]].drop_duplicates()
    t = t.groupby(["graph_id", "solver", "device"])[["wall_ms", "setup_transfer_ms", "kernel_ms",
                                                     "sync_gap_ms", "solve_ms"]].median().reset_index()
    # The same graph, solver and sources run without kernel events elsewhere.
    plain = med[~med.experiment.isin(["T", "F"])].merge(t_sources, on=["graph_id", "solver", "source"])
    plain = plain.groupby(["graph_id", "solver"]).solve_ms.median().rename("solve_ms_uninstrumented")
    t = t.merge(plain.reset_index(), on=["graph_id", "solver"], how="left")
    t.merge(info("T")[["label"]], left_on="graph_id", right_index=True).to_csv(out / "time_breakdown.csv", index=False)


# hardware counters

def bottleneck(r):
    """Fixed thresholds, so the labels can be checked against the counters."""
    if r.sm_tail >= 2:
        return "load imbalance"
    if r.sm_active_pct < 50 or r.occupancy_pct < 25:
        return "too little parallelism"
    if max(r.dram_pct, r.l2_pct) >= 60:
        return "bandwidth"
    if r.issue_active_pct >= 60:
        return "instruction issue"
    return "memory latency"


def read_perf(path):
    vals = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and not line.startswith("#"):
            vals[parts[2]] = pd.to_numeric(parts[0], errors="coerce")
    return vals


def profiles(raw):
    f = raw / "F"
    gpu, cpu, place, timeline = [], [], [], {}
    for path in sorted(f.glob("ncu_*.csv")):
        text = path.read_text()
        df = pd.read_csv(StringIO(text[text.index('"ID"'):]))
        df["v"] = pd.to_numeric(df["Metric Value"].astype(str).str.replace(",", ""), errors="coerce")
        w = df.pivot_table(index="ID", columns="Metric Name", values="v")
        t = w["gpu__time_duration.sum"]
        avg = lambda col: float(np.average(w[col], weights=t))
        r = pd.Series({
            "kernel": path.stem[4:], "launches": len(w), "kernel_us": t.median() / 1e3,
            "sm_active_pct": avg("sm__cycles_active.avg.pct_of_peak_sustained_elapsed"),
            "occupancy_pct": avg("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "sm_tail": float(np.average(w["sm__cycles_active.max"] / w["sm__cycles_active.avg"], weights=t)),
            "dram_pct": avg("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "l2_pct": avg("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
            "issue_active_pct": avg("smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "lanes_active": avg("smsp__thread_inst_executed_per_inst_executed.ratio"),
            "sectors_per_request": w["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"].sum()
                                   / w["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"].sum(),
        })
        r["bottleneck"] = bottleneck(r)
        gpu.append(r)

    kern, api = f / "nsys_grid_frontier_cuda_gpu_kern_sum.csv", f / "nsys_grid_frontier_cuda_api_sum.csv"
    if kern.exists() and api.exists():
        a = pd.read_csv(api)
        copies = a[a.Name.str.startswith("cudaMemcpy")]
        timeline = {"kernel_ms": pd.read_csv(kern)["Total Time (ns)"].sum() / 1e6,
                    "copy_ms": copies["Total Time (ns)"].sum() / 1e6, "copies": int(copies["Num Calls"].sum())}

    # CPU counters: 11 repetitions minus 1 leaves 10 solves.
    for r11 in sorted(f.glob("perf_*_r11.csv")):
        a, b = read_perf(r11), read_perf(Path(str(r11).replace("_r11", "_r1")))
        d = {key: a[key] - b.get(key, np.nan) for key in a}
        name = r11.stem[5:-4]
        solver = next(s for s in ["delta-omp", "dijkstra", "dial"] if name.startswith(s))
        ins = [d["cpu_core/instructions/"], d["cpu_atom/instructions/"]]
        wall = d["duration_time"] / 1e6
        cpu.append({"solver": solver, "graph": name[len(solver) + 1:], "ms_per_solve": wall / 10,
                    "ipc_p_core": ins[0] / d["cpu_core/cycles/"], "ipc_e_core": ins[1] / d["cpu_atom/cycles/"],
                    "e_core_share": ins[1] / sum(ins),
                    "llc_misses_per_1k_instructions": np.nansum([d.get("cpu_core/LLC-load-misses/"),
                                                                 d.get("cpu_atom/LLC-load-misses/")]) / (sum(ins) / 1e3),
                    "thread_utilization": d["task-clock"] / (wall * (24 if solver == "delta-omp" else 1))})

    # Serial CPU solvers pinned to one P core versus left to the scheduler.
    for run in sorted((f / "placement").glob("*/")):
        info = pd.read_csv(run / "runs.csv").iloc[0]
        s = pd.read_csv(run / "samples.csv")
        for solver, grp in s[s.warmup == 0].groupby("solver"):
            place.append({"graph": Path(info.graph_path).stem, "solver": solver,
                          "placement": info.affinity_label, "wall_ms": grp.wall_ms.median()})
    place = pd.DataFrame(place)
    if len(place):
        place = place.pivot_table(index=["graph", "solver"], columns="placement", values="wall_ms").reset_index()
        place["unpinned_over_pinned"] = place.unpinned / place.pcore
    return pd.DataFrame(gpu), pd.DataFrame(cpu), timeline, place


# output

def md_table(df, digits=2):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "never" if v == np.inf else ("" if np.isnan(v) else f"{v:.{digits}f}")
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "published")
    ap.add_argument("--allow-failures", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out

    runs, graphs, sources, samples, batch, calib = load(args.raw)
    checks = pd.DataFrame(validate(runs, graphs, sources, samples), columns=["check", "status", "detail"])
    (out / "validation.md").write_text("# Validation\n\n" + md_table(checks) + "\n")
    print(md_table(checks))
    if (checks.status == "FAIL").any() and not args.allow_failures:
        sys.exit("validation failed")

    k = constants(calib)
    good = samples[(samples.warmup == 0) & (samples.mismatches == 0)]
    timed = good[good.instrumented == 0]
    keys = ["experiment", "graph_id", "source", "solver", "device", "delta_scale", "run_cache"]
    counters = ["wall_ms", "q_ms", "U_ms", "solve_ms", "d2h_ms", "host_syncs", "sync_rounds",
                "edges_touched", "vertices_expanded", "untimed_ms"]
    med_all = timed[timed.run_mode == "oneshot"].groupby(keys)[counters].median().reset_index()
    med = med_all[(med_all.delta_scale == 0) & (med_all.run_cache == "warm")]

    ps = per_source(med, sources, graphs, k)
    solvers = per_solver(med_all, sources, graphs)
    pg = per_graph(ps)
    model, fits = model_table(med, k)
    choice = device_choice(ps, k, fits)
    amort = amortization(timed, batch, pg, graphs)
    prof_gpu, prof_cpu, timeline, place = profiles(args.raw)
    plot_tables(out, med_all, ps, graphs, good, k)
    tables = {"per_source": ps, "per_graph": pg, "per_solver": solvers, "model": model, "device_choice": choice,
              "amortization": amort, "profile_gpu": prof_gpu, "profile_cpu": prof_cpu, "placement": place}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    pd.Series(k, name="value").to_csv(out / "constants.csv", index_label="constant")

    study = timed[timed.experiment != "F"]
    env = runs.iloc[0]
    parts = [
        "# Summary\n",
        f"{env.cpu_model}, {env.gpu_name}, driver {env.nvidia_driver}. {len(study)} timed runs over "
        f"{study.graph_id.nunique()} graphs, plus {len(timed) - len(study)} profiling and placement runs.\n",
        "## Calibration constants\n", md_table(pd.DataFrame(k.items(), columns=["constant", "value"]), 3),
        "\n## Model (fitted on A, median error elsewhere)\n", md_table(model, 3),
        "\n## Picking the faster device\n", md_table(choice, 3),
        "\n## Break even query counts\n", md_table(amort.drop(columns="graph_id")),
        "\n## Hardware counters\n", md_table(prof_gpu),
        f"\nNsight Systems, grid, gpu-frontier, 4 solves: {timeline.get('kernel_ms', 0):.1f} ms in kernels, "
        f"{timeline.get('copy_ms', 0):.1f} ms in {timeline.get('copies', 0)} blocking copies.\n",
        md_table(prof_cpu),
        "\n## Serial CPU solvers pinned to one P core\n", md_table(place),
        "\n## Per graph\n",
        md_table(pg.assign(S1=2 ** pg.log2_S1, Sinf=2 ** pg.log2_Sinf)
                 [["experiment", "label", "Pi", "best_cpu", "best_gpu", "S1", "Sinf", "Kstar", "iota", "delta"]]
                 .sort_values(["experiment", "label"])),
    ]
    (out / "summary.md").write_text("\n".join(parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
