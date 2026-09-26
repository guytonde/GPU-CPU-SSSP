#!/usr/bin/env python3
"""Pivot a sweep csv into one row per graph, one column per solver.

Rows are per (graph, source, solver); sources collapse to their median here.
"""
import csv
import statistics
import sys
from collections import defaultdict


def main(path, metric="solve_ms"):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("empty csv")
        return

    solvers = sorted({r["solver"] for r in rows},
                     key=lambda s: (s.startswith("gpu"), s))
    values = defaultdict(list)
    wrong = set()
    for r in rows:
        values[(r["tag"], r["solver"])].append(float(r[metric]))
        if r["correct"] != "1":
            wrong.add((r["tag"], r["solver"]))

    tags = list(dict.fromkeys(r["tag"] for r in rows))
    width = max(len(t) for t in tags) + 2
    print(f"{metric}, median over sources")
    print(f"{'graph':<{width}}" + "".join(f"{s:>14}" for s in solvers))
    print("-" * (width + 14 * len(solvers)))

    for tag in tags:
        cells = []
        for s in solvers:
            got = values.get((tag, s))
            if not got:
                cells.append("-")
            elif (tag, s) in wrong:
                cells.append("WRONG")
            else:
                cells.append(f"{statistics.median(got):.2f}")
        print(f"{tag:<{width}}" + "".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep.csv",
         sys.argv[2] if len(sys.argv) > 2 else "solve_ms")
