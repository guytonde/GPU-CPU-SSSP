#!/usr/bin/env python3
"""Runs experiments C0 to T. Each step generates a graph, runs the benchmark on
it and deletes it again.

  tools/experiments.py all            # everything, in order
  tools/experiments.py C0 A           # selected experiments
  tools/experiments.py --quick smoke  # tiny end-to-end check of the pipeline
  tools/experiments.py --list         # print the plan without running it

Finished steps are listed in results/raw/<exp>/done.txt and skipped next time.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"

CPU = ["dijkstra", "dial", "delta", "delta-omp"]
GPU = ["gpu-frontier", "gpu-nearfar"]
W100 = {"wdist": "uniform", "wmin": 1, "wmax": 100}


@dataclass
class Step:
    exp: str
    name: str
    gen: dict
    benches: list = field(default_factory=list)  # lists of bench arguments

    def key(self, bench):
        blob = json.dumps([self.gen, bench], sort_keys=True)
        return f"{self.name}:{hashlib.sha1(blob.encode()).hexdigest()[:10]}"


def only(*solvers):
    return ["--only", ",".join(solvers)]


def plan(quick):
    """Every step of the study. Quick mode shrinks every size."""
    n = 1 << (14 if quick else 20)
    side = 1 << (7 if quick else 10)
    reps = ["--reps", "2" if quick else "5"]
    srcs = ["--sources", "3" if quick else "8"]
    steps = []

    # A: layer width, with unit and [1,100] weights on the same edges.
    widths = [1 << k for k in range(4, (13 if quick else 19), 2)]
    for w in widths:
        for wname, wd in (("unit", {"wdist": "unit"}), ("w100", W100)):
            solvers = CPU + GPU + (["gpu-edge"] if w >= 4096 else [])
            steps.append(Step("A", f"layered_W{w}_{wname}",
                              {"topo": "layered", "n": n, "layer-width": w, **wd},
                              [only(*solvers) + srcs + reps]))
    for w in widths[1::2][:3]:
        steps.append(Step("A", f"layered_W{w}_w100_natural",
                          {"topo": "layered", "n": n, "layer-width": w, "no-relabel": True, **W100},
                          [only(*CPU, *GPU) + srcs + reps]))
        steps.append(Step("A", f"layered_W{w}_w100_links16",
                          {"topo": "layered", "n": n, "layer-width": w, "layer-links": 16, **W100},
                          [only(*CPU, *GPU) + srcs + reps]))

    # B: one hub holding a growing share of the edges, and two RMAT graphs.
    skew = ["dijkstra", "delta", "delta-omp", "bellman-ford", "gpu-topo", "gpu-edge", "gpu-frontier"]
    for k in (None, 12, 10, 8, 6, 4, 2):
        h = 0.0 if k is None else 2.0 ** -k
        steps.append(Step("B", f"hub_h{0 if k is None else k}",
                          {"topo": "hub", "n": n, "m": 4 * n, "hub-fraction": h, **W100},
                          [only(*skew) + srcs + reps]))
    for relabel in (True, False):
        gen = {"topo": "rmat", "n": n, "m": 4 * n, **W100}
        if not relabel:
            gen["no-relabel"] = True
        steps.append(Step("B", f"rmat_{'relabeled' if relabel else 'natural'}", gen,
                          [only(*skew) + srcs + reps]))

    # C: weight distributions on fixed edges, and a bucket width sweep.
    weights = [("unit", {"wdist": "unit"}),
               ("logD1", {"wdist": "logunif", "decades": 1}),
               ("logD2", {"wdist": "logunif", "decades": 2}),
               ("logD4", {"wdist": "logunif", "decades": 4}),
               ("logD6", {"wdist": "logunif", "decades": 6}),
               ("u100", W100),
               ("u1e5", {"wdist": "uniform", "wmin": 1, "wmax": 100000})]
    substrates = [("uniform", {"topo": "uniform", "n": n, "m": 4 * n}),
                  ("grid", {"topo": "grid", "rows": side, "cols": side})]
    for sname, sub in substrates:
        for wname, wd in weights:
            benches = [only(*CPU, *GPU) + srcs + reps]
            if wname != "unit":
                for scale in ("0.0625", "0.25", "4", "16"):
                    benches.append(only("delta", "delta-omp", "gpu-nearfar") + srcs +
                                   ["--reps", "2" if quick else "3", "--delta-scale", scale])
            steps.append(Step("C", f"{sname}_{wname}", {**sub, **wd}, benches))

    # D: resident sessions and the CPU batch control.
    k = "16" if quick else "128"
    d_graphs = [(f"layered_W{w}_w100", {"topo": "layered", "n": n, "layer-width": w, **W100})
                for w in ([1 << 6, 1 << 8, 1 << 10] if quick else [1 << 8, 1 << 12, 1 << 16])]
    d_graphs += [("uniform_w100", {"topo": "uniform", "n": n, "m": 4 * n, **W100}),
                 ("grid_w100", {"topo": "grid", "rows": side, "cols": side, **W100})]
    for name, gen in d_graphs:
        steps.append(Step("D", name, gen, [
            only("dial", "delta-omp", *GPU) + ["--mode", "resident", "--queries", k,
                                                "--reps", "2" if quick else "3"],
            only("dijkstra", "dial") + ["--mode", "batch", "--queries", k,
                                        "--reps", "2" if quick else "3"]]))

    # E: size, and three sizes with flushed caches.
    lo, hi = (10, 16) if quick else (16, 24)
    for kk in range(lo, hi + 1):
        size = 1 << kk
        big = kk >= 23
        e_reps = ["--reps", "3" if big else ("2" if quick else "5")]
        e_srcs = ["--sources", "3" if quick else "6"]
        main = only("dijkstra", "dial", "delta-omp", *GPU) + e_srcs + e_reps
        u_benches = [main]
        if kk in ((12, 14, 16) if quick else (18, 20, 22)):
            u_benches.append(main + ["--cache", "flushed"])
        steps.append(Step("E", f"uniform_n2e{kk}",
                          {"topo": "uniform", "n": size, "m": 4 * size, **W100}, u_benches))
        steps.append(Step("E", f"grid_n2e{kk}",
                          {"topo": "grid", "rows": 1 << ((kk + 1) // 2),
                           "cols": 1 << (kk // 2), **W100}, [main]))

    # HO: held out graphs, never used for fitting.
    all8 = ["dijkstra", "dial", "delta", "delta-omp", "gpu-topo", "gpu-edge", *GPU]
    held = [("grid_natural", {"topo": "grid", "rows": side, "cols": side, "no-relabel": True}),
            ("grid_relabeled", {"topo": "grid", "rows": side, "cols": side}),
            ("geometric", {"topo": "geometric", "n": n, "m": 4 * n}),
            ("uniform", {"topo": "uniform", "n": n, "m": 4 * n}),
            ("rmat", {"topo": "rmat", "n": n, "m": 4 * n})]
    for name, gen in held:
        steps.append(Step("HO", name, {**gen, **W100}, [only(*all8) + srcs + reps]))

    # T: runs with kernel events, used only for the time breakdown.
    ireps = ["--reps", "2" if quick else "3", "--sources", "2" if quick else "4", "--instrument"]
    budget = [("grid_unit", {"topo": "grid", "rows": side, "cols": side, "wdist": "unit"}),
              (f"layered_W{widths[-2]}_w100", {"topo": "layered", "n": n,
                                               "layer-width": widths[-2], **W100}),
              ("uniform_logD6", {"topo": "uniform", "n": n, "m": 4 * n, "wdist": "logunif",
                                 "decades": 6}),
              ("hub_h6", {"topo": "hub", "n": n, "m": 4 * n, "hub-fraction": 2.0 ** -6, **W100}),
              ("uniform_big", {"topo": "uniform", "n": 4 * n, "m": 16 * n, **W100})]
    for name, gen in budget:
        steps.append(Step("T", name, gen,
                          [only("dial", "delta-omp", "gpu-topo", "gpu-edge", *GPU) + ireps]))

    # smoke: one small graph through every mode.
    steps.append(Step("smoke", "layered_smoke",
                      {"topo": "layered", "n": 1 << 12, "layer-width": 64, **W100},
                      [["--sources", "2", "--reps", "1"],
                       only("dial", "gpu-frontier") + ["--mode", "resident", "--queries", "4",
                                                       "--reps", "1"],
                       only("dial") + ["--mode", "batch", "--queries", "8", "--reps", "1"]]))
    return steps


def gen_args(gen):
    out = []
    for k, v in gen.items():
        if v is True:
            out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def foreign_gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line for line in out.splitlines() if line.strip()]


def wait_for_idle_gpu(max_wait_min, log):
    waited = 0
    while True:
        busy = foreign_gpu_processes()
        if not busy:
            return
        if waited >= max_wait_min * 60:
            sys.exit(f"GPU still busy after {max_wait_min} min: {busy}")
        if waited % 600 == 0:
            log(f"  GPU busy ({busy}); waiting")
        time.sleep(60)
        waited += 60


def estimate_bytes(gen):
    n = gen.get("n") or gen.get("rows", 1) * gen.get("cols", 1)
    m = gen.get("m", 4 * n)
    return 4 * n + 16 * m + 1024


def cpu_config(raw):
    path = raw / "C0" / "cpu_config.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"label": "all", "taskset": "", "threads": 0, "env": {}}


def run(cmd, env, logf):
    logf.write("$ " + " ".join(cmd) + "\n")
    logf.flush()
    t = time.time()
    proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return time.time() - t


def choose_cpu_config(raw, graphs, log, quick):
    """Measures three CPU thread settings once and keeps one for every experiment."""
    path = raw / "C0" / "cpu_config.json"
    if path.exists():
        return
    size = 1 << (16 if quick else 22)
    g = graphs / "c0_uniform.bin"
    p_cpus = Path("/sys/devices/cpu_core/cpus")
    one_per_p = []
    if p_cpus.exists():
        seen = set()
        for cpu in expand_list(p_cpus.read_text().strip()):
            sib = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
            if sib not in seen:
                seen.add(sib)
                one_per_p.append(cpu)
    logical = os.cpu_count()
    configs = [{"label": "all_physical", "taskset": "", "threads": 0, "env": {}},
               {"label": "all_logical", "taskset": "", "threads": logical, "env": {}}]
    if one_per_p:
        configs.append({"label": f"p{len(one_per_p)}_pinned",
                        "taskset": ",".join(map(str, one_per_p)), "threads": len(one_per_p),
                        "env": {"OMP_PROC_BIND": "true"}})
    out = raw / "C0" / "cpu_configs"
    out.mkdir(parents=True, exist_ok=True)

    def median_for(label):
        for run in out.iterdir():
            rows = list(csv.DictReader(open(run / "runs.csv"))) if (run / "runs.csv").exists() else []
            if rows and rows[0].get("affinity_label") == label and rows[0].get("finished_utc"):
                t = sorted(float(r["solve_ms"]) for r in csv.DictReader(open(run / "samples.csv"))
                           if r["solver"] == "delta-omp" and r["warmup"] == "0")
                return t[len(t) // 2]
        return None

    results = []
    with open(raw / "C0" / "log.txt", "a") as logf:
        for c in configs:
            med = median_for(c["label"])
            if med is None:
                if not g.exists():
                    subprocess.run([str(BIN / "gen_graph"), "--topo", "uniform", "--n", str(size),
                                    "--m", str(4 * size), *gen_args(W100), "--out", str(g)],
                                   check=True, capture_output=True)
                cmd = (["taskset", "-c", c["taskset"]] if c["taskset"] else []) + [
                    str(BIN / "bench"), str(g), "--experiment", "C0", "--out", str(out),
                    "--only", "dial,delta-omp", "--sources", "4", "--reps", "3" if quick else "5",
                    "--affinity-label", c["label"], "--quiet"]
                if c["threads"]:
                    cmd += ["--threads", str(c["threads"])]
                run(cmd, {**os.environ, **c["env"]}, logf)
                med = median_for(c["label"])
            c["threads_effective"] = c["threads"] or len(set(
                Path(f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list").read_text()
                for i in range(logical)))
            results.append((med, c))
            log(f"  cpu config {c['label']}: delta-omp median {med:.2f} ms")
    g.unlink(missing_ok=True)
    # The fewest threads within 5% of the fastest.
    fastest = min(m for m, _ in results)
    tied = [(m, c) for m, c in results if m <= 1.05 * fastest]
    best = min(tied, key=lambda x: x[1]["threads_effective"])[1]
    best["rule"] = "fewest threads among configurations within 5% of the fastest delta-omp median"
    best["medians_ms"] = {c["label"]: round(m, 3) for m, c in results}
    path.write_text(json.dumps(best, indent=2) + "\n")
    log(f"  chose cpu config {best['label']}")


def expand_list(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", default=["all"])
    ap.add_argument("--quick", action="store_true", help="tiny sizes; writes to results/raw-quick")
    ap.add_argument("--keep-graphs", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-wait", type=int, default=240, help="minutes to wait for a busy GPU")
    args = ap.parse_args()

    raw = ROOT / ("results/raw-quick" if args.quick else "results/raw")
    graphs = ROOT / "graphs" / ("quick" if args.quick else "study")
    order = ["C0", "A", "B", "C", "D", "E", "HO", "T"]
    wanted = order if args.experiments == ["all"] else args.experiments
    steps = [s for s in plan(args.quick) if s.exp in wanted]

    if args.list:
        for s in steps:
            for b in s.benches:
                print(f"{s.exp:5} {s.name:32} gen {' '.join(gen_args(s.gen))} | bench {' '.join(b)}")
        return

    graphs.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def log(msg):
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp} +{(time.time() - start) / 60:5.1f}m] {msg}", flush=True)

    if "C0" in wanted:
        (raw / "C0").mkdir(parents=True, exist_ok=True)
        done = raw / "C0" / "done.txt"
        if not done.exists() or "calibrate" not in done.read_text():
            wait_for_idle_gpu(args.max_wait, log)
            log("C0 calibrate")
            with open(raw / "C0" / "log.txt", "a") as logf:
                run([str(BIN / "calibrate"), "--out", str(raw / "C0")], os.environ.copy(), logf)
            with open(done, "a") as f:
                f.write("calibrate\n")
        log("C0 cpu configuration")
        choose_cpu_config(raw, graphs, log, args.quick)

    cfg = cpu_config(raw)
    env = {**os.environ, **cfg.get("env", {})}
    prefix = ["taskset", "-c", cfg["taskset"]] if cfg.get("taskset") else []
    threads = ["--threads", str(cfg["threads"])] if cfg.get("threads") else []

    for step in steps:
        exp_dir = raw / step.exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        done_file = exp_dir / "done.txt"
        done = set(done_file.read_text().split()) if done_file.exists() else set()
        todo = [b for b in step.benches if step.key(b) not in done]
        if not todo:
            continue

        g = graphs / f"{step.exp}_{step.name}.bin"
        need = estimate_bytes(step.gen)
        free = shutil.disk_usage(graphs).free
        if free < need + (2 << 30):
            sys.exit(f"only {free >> 20} MB free; {step.name} needs about {need >> 20} MB")
        with open(exp_dir / "log.txt", "a") as logf:
            if not g.exists():
                t = run([str(BIN / "gen_graph"), *gen_args(step.gen), "--out", str(g)],
                        os.environ.copy(), logf)
                log(f"{step.exp} {step.name}: generated in {t:.1f}s")
            for b in todo:
                wait_for_idle_gpu(args.max_wait, log)
                cmd = prefix + [str(BIN / "bench"), str(g), "--experiment", step.exp,
                                "--out", str(exp_dir), "--affinity-label", cfg["label"],
                                "--quiet", *threads, *b]
                t = run(cmd, env, logf)
                with open(done_file, "a") as f:
                    f.write(step.key(b) + "\n")
                log(f"{step.exp} {step.name}: {' '.join(b)} ({t:.1f}s)")
        if not args.keep_graphs:
            g.unlink(missing_ok=True)
    log("done")


if __name__ == "__main__":
    main()
