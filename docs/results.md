# Results

Everything here was measured on one machine with an RTX 4090 (128 SMs, 72 MB L2,
PCIe 4.0 x16) and an i9 13900KF (8 P cores and 16 E cores, 36 MB L3). The main
study is 27,246 timed runs over 62 graphs, and experiment F adds 803 profiling
and placement runs. Every distance array matched an independent reference, no
GPU run overlapped with another process on the GPU, and every run carries the
same code hash. The checks are in `results/published/validation.md` and every
number below comes from the files next to it.

A speedup compares the faster of `gpu-frontier` and `gpu-nearfar` with the
fastest CPU solver that ran on the same graph and source vertex, out of
`dijkstra`, `dial`, `delta` and `delta-omp`. The solvers that sweep the whole
graph every round, `gpu-topo`, `gpu-edge` and `bellman-ford`, ran only in some
experiments and have their own section below. The resident speedup assumes the
graph is already on the GPU, and the one shot speedup includes uploading it.
Both are geometric means over the sources, eight per graph or six in experiment
E.

## Calibration (C0)

Before the experiments, the microbenchmarks in `bin/calibrate` measure the costs
the model relies on. An empty round of the kind `gpu-frontier` runs, a memset,
two kernel launches and one blocking copy back to the host, takes 14.4 µs. A
frontier round costs about 1.8 ns per vertex on a degree 8 graph and 2.0 ns per
vertex on a degree 4 grid. The cost per vertex is the same at both degrees, so
`gpu-frontier` pays per vertex rather than per edge, which fits the fact that it
gives each vertex a warp of 32 threads. One thread walking a large hub takes 133
ns per edge. Copying 1 GB to the GPU runs at 16.8 GB/s from pageable memory and
20.3 GB/s from pinned memory. An empty OpenMP parallel region with 24 threads
costs 2.1 µs.

C0 also chose the CPU thread count. On a 2^22 vertex graph `delta-omp` took 97.7
ms with 24 threads, one per physical core, 97.6 ms with all 32 logical CPUs and
169.0 ms with 8 threads pinned to the P cores. The rule was to take the fewest
threads within 5% of the fastest, so every experiment used 24. The measurements
and the rule are in `results/raw/C0/cpu_config.json`.

## A. How much parallel work each round has

These are layered graphs with 2^20 vertices and an average degree of 8. Each
layer connects only to the next one, so a search from the first layer moves one
layer per round. The layer width W sets how wide the frontier is, and the number
of layers sets how many rounds there are. W goes from 16 (65,536 layers with a
tiny frontier) to 262,144 (four very wide layers). Each width was run with unit
weights and with weights from 1 to 100 on exactly the same edges. The plots use
Π, the number of useful edges divided by the depth of the shortest path tree, as
the measure of parallelism.

![Speedup against useful parallelism](figures/speedup_vs_parallelism.png)

```
layer width                              16     64    256   1024   4096  16384  65536 262144
resident speedup, unit                 0.09   0.34   1.38   5.38   15.5   5.07   6.80   6.40
resident speedup, 1 to 100             0.05   0.15   0.56   1.99   2.66   5.27   4.38   4.12
one shot speedup, 1 to 100             0.05   0.14   0.54   1.77   1.81   2.49   1.80   1.74
gpu-frontier edges per useful edge     1386    460    113   31.8   8.57   3.71   3.25   2.97
  (weights 1 to 100)
```

With unit weights the GPU loses on deep graphs and wins from W = 256 on, so the
crossover lies between Π = 512 and 2,048. With weights from 1 to 100 it moves to
between Π = 1,828 and 7,187. The reason is the last row. `gpu-frontier`
processes vertices in no particular order, so with varying weights it keeps
finding shorter paths to vertices it has already expanded and expands them
again. On the deepest graph it touched 1,386 edges for every one it needed. The
calibrated model assumes no wasted work. For unit weights it puts the crossover
in the same interval as the measurement, and for weighted graphs it puts it
between Π = 459 and 1,828, too early. `gpu-nearfar`, which processes vertices
roughly in distance order, was faster than `gpu-frontier` only at W = 65,536 and
262,144 with weights, and by at most 3%.

![Time per synchronization against work per synchronization](figures/sync_cost.png)

The left plot shows that the BFS frontier width cannot explain the weight
effect, because unit and weighted runs have the same frontier but different
times. The right plot measures the same runs per synchronization, and there runs
with similar work per synchronization take similar time whatever their weights,
for example 30.0 µs at 125,000 edges for unit weights at W = 16,384 and 31.4 µs
at 163,000 edges for weighted W = 16. Up to about 32,000 edges per
synchronization `gpu-frontier` costs a flat 14.4 to 14.6 µs per round, the same
as the empty round measured in C0. Beyond that its cost grows with the work, to
about 100 µs at 500,000 to 700,000 edges.

Two controls at W = 1024 with weights from 1 to 100 changed one more thing.
Numbering the vertices layer by layer instead of randomly made the best CPU
solver, `dial`, 2.8 times faster (98.1 ms down to 35.4 ms) while the GPU stayed
at 48.7 ms, which turned a 1.99x GPU win into a 0.72x loss. Raising the degree
from 8 to 32 made `dial` 2.3 times slower (222.2 ms) while the GPU time rose
only to 51.5 ms, so the speedup rose from 1.99x to 4.40x.

`delta-omp` runs a phase serially when it holds fewer than 4,096 vertices. For W
up to 1024 it ran every phase serially and was 4 to 5% slower than serial
`delta`.

## B. One very long adjacency list

This experiment asks how each GPU solver copes with degree skew. The graphs have
2^20 vertices and 2^22 undirected edges, and a growing share of the edges is
moved onto a single hub, so the largest degree goes from 26 to about a million.
Each solver is measured against `gpu-edge`, which gives every edge its own
thread. Two RMAT graphs, one with random vertex numbering and one with the
generator's own, serve as a held out check.

![Degree skew](figures/skew.png)

```
largest degree                 26     1k     4k    16k    66k   262k     1M
gpu-topo vs gpu-edge         1.43   2.16   5.21   17.1   62.3    153    293
gpu-frontier vs gpu-edge     1.31   1.34   1.42   1.69   2.20   4.72   9.20
```

`gpu-topo` gives each vertex one thread, so a single thread ends up walking the
whole hub. `gpu-frontier` gives each vertex a warp of 32 threads, and its ratio
passes 2 only at a degree of about 65,000, where that of `gpu-topo` passes 2 at
about 1,000. The right plot uses ρ, which compares the time one thread or warp
needs to walk the hub with the time the rest of a round takes, using the
constants from C0. At the largest degree the two curves are 32 times apart, and
at equal ρ they are within 3.5 times of each other. The RMAT points land within
25% of the hub curves at the same ρ. `gpu-edge` itself took 2.5 to 3.2 ms until
the hub held a sixteenth of all edges, and 5.4 ms when it held a quarter. On the
CPU, `delta-omp` stayed about 10 times faster than serial `delta` up to a degree
of 262,000 and was still 6.6 times faster at a million.

## C. Weight distributions

This experiment keeps the edges fixed and changes only the weights. It uses a
uniform random graph with 2^20 vertices, which is shallow, and a 1024 by 1024
grid, which is deep. The weights are unit, log uniform over 1 to 10^D for D of
1, 2, 4 and 6, and uniform over 1 to 100 and over 1 to 100,000. `gpu-nearfar`
and the delta stepping solvers were also run with their bucket width Δ set to
1/16, 1/4, 4 and 16 times the default.

![Weight distribution](figures/weights.png)

On the uniform graph `gpu-frontier` touched at most 7.6 edges per useful edge.
`gpu-nearfar` was at most 1.5 times faster than it, with log uniform weights
over two decades, and with four or six decades it was slower even with Δ tuned,
because there it touched about as many edges as `gpu-frontier` while
synchronizing 5.6 to 6.4 times as often. The resident speedup on the uniform
graph ranged from 4.9x to 14.3x across the weights.

On the grid the wasted work of `gpu-frontier` grows from none with unit weights
to 244 edges per useful edge with six decades. `gpu-nearfar` lost by 2.9x with
unit weights. With the other six weightings it won by 1.1x to 1.9x when Δ was
tuned, but with the default Δ it lost on four of them. The best Δ was 16 times
the default for four weightings and 1/4 and 1/16 of it for the other two, and
the default cost up to 2.9x. Several of these best values sit at the edge of the
range tried, so the true best may lie further out. The trade shows on the grid
with weights from 1 to 100, where `gpu-nearfar` touched 43 times fewer edges but
synchronized 12.7 times as often, and came out 1.47 times faster with the best Δ
and 1.96 times slower with the default. The resident speedup on the grid ranged
from 0.48x to 1.73x. On the uniform graph the best Δ was 4 or 16 times the
default for five of the six weightings, and the default cost at most 1.27x.

Weights from 1 to 100 and from 1 to 100,000 gave times within 10% of each other
for every solver except `dial`, which was 1.4 times slower on the uniform graph
and 3.8 times slower on the grid with the wider range, since its ring of buckets
grows with the largest weight.

## D. Keeping the graph on the GPU

For every graph in A, C, E and HO I computed the break even point, the number of
queries after which uploading the graph once and keeping it on the GPU beats the
CPU. For five graphs I also measured sessions of 128 queries, and compared them
with a CPU batch that answers the same queries on all cores at once.

The break even point was the first query for 37 of the 59 graphs and never for
the other 22. Uploading a graph cost a median of 16% of one CPU query and at
most 52%, and in every case where the GPU was faster per query, its saving on
the first query already covered the upload. When the GPU wins, the one shot
speedup keeps a median of 42% of the resident speedup, and the upload never
turns a win into a loss. The measured sessions matched the prediction for all 10
pairs of graph and GPU solver. Against the CPU batch, the layered graph with W =
4096 goes from winning on the first query to never winning, while the layered
graph with W = 65,536 and the uniform graph still win from the first query.

## E. Graph size

Uniform graphs with degree 8 and grids with degree 4 were run from 2^16 to 2^24
vertices with weights from 1 to 100. Three sizes of the uniform graph were
repeated with the caches flushed before every run.

![Scaling](figures/scaling.png)

On uniform graphs the resident speedup is 3.4x to 3.8x up to 2^19 vertices,
rises to 7.8x at 2^22 and falls back to 5.3x at 2^24. The right plot shows why.
Per edge touched, `gpu-frontier` costs about 0.14 ns from 2^18 to 2^20 vertices,
where the working set reaches 75 MB, about the 72 MB of the L2, and 0.60 ns at
2^24, more than four times as much. `gpu-nearfar`, the faster GPU solver from
2^20 on, goes from 0.30 ns at 2^21 to 0.69 ns at 2^24, while `delta-omp` goes
from 1.10 ns to 1.93 ns. The CPU slows gradually, with no step at its 36 MB L3.

Grids lose up to 2^21 vertices and win from 2^22 (1.16x) to 2^24 (2.32x). Their
Π grows roughly with the square root of the size, from 625 to 10,887, and they
cross over between Π = 3,139 and 4,821, inside the interval where the weighted
layered graphs of A cross.

Flushing the caches before every run changed no time by more than 3% at 2^20
vertices and 1% at 2^22. At 2^18 it made `delta-omp` 13% slower and `dial` 5%
slower and changed the GPU solvers by about 1%. Copy bandwidth to the GPU from
pageable memory rose from 13.0 GB/s at 1 MB to 16.8 GB/s at 16 MB and was the
same at 1 GB.

## Held out graphs (HO)

These graphs were never used to fit anything. A 1024 by 1024 grid runs at 0.72x
with random numbering and 0.36x with row major numbering, where the CPU is about
twice as fast (47.6 ms against 24.2 ms) and the GPU is not (70.5 ms and 69.3
ms). A geometric graph with the same number of vertices runs at 0.74x, a uniform
random graph at 4.84x and an RMAT graph at 3.50x. Compared with the weighted
layered curve at the same Π, the randomly numbered grid lands within 2% of it,
and the row major grid and the geometric graph at about half its height. The
uniform and RMAT graphs have more parallelism than any layered graph, and run at
1.11 and 0.80 times the layered graph with the most (W = 65,536, 4.38x).

## The solvers that sweep the whole graph

`gpu-topo`, `gpu-edge` and `bellman-ford` go over the whole graph in every
round, so a query costs them roughly the depth times the number of edges. They
ran in B and HO and in the time breakdown, and `gpu-edge` also ran on the
layered graphs of A with W of 4096 and up. `bellman-ford` ran only in B, where
it took 88 to 260 ms against 12 to 21 ms for `delta-omp`.

On those graphs `gpu-edge` was often the fastest GPU solver. It beat the faster
of `gpu-frontier` and `gpu-nearfar` on all nine graphs in B, by 1.24x with no
hub up to 9.5x with the largest hub, on six of the eight layered graphs in A, by
up to 1.32x and on two of them by only 1 to 2%, and on four of the five held out
graphs, by up to 1.93x on RMAT. It lost by up to 2.4x on the weighted layered
graphs with W = 4096 and 16,384 and by 1.25x on the randomly numbered grid. In A
and HO it never changed which device was faster, but in B it turned the largest
hub from a 0.37x loss for the GPU into a 3.6x win. So where it was measured, the
speedups in this document understate the GPU by as much as 1.93x. It was not run
in C, D or E. `gpu-topo` was slower than the best of the other three GPU solvers
on every graph in B and HO.

## Where the GPU time goes (T)

These runs place an event around every kernel so that GPU time can be split into
setup and transfer, gaps between kernels, and the kernels themselves. Compared
with the same graphs, solvers and sources timed without events, the events made
the GPU solvers up to 21% slower, on the unit weight grid, and changed the CPU
solvers by at most 5%, so these runs are used for nothing else.

![Time breakdown](figures/time_breakdown.png)

On the unit weight grid, `gpu-frontier` spends 51% of its time in the gaps
between kernels, 35% in kernels and 14% on setup, and `gpu-nearfar` spends 73%
in gaps. On the large uniform graph, setup and transfer take from 39% of the
time of `gpu-frontier` to 86% of the time of `gpu-edge`, and every GPU solver
still finishes before `delta-omp`, in 43 to 109 ms against 110 ms. With the big
hub, `gpu-topo` spends 96% of its time in kernels, where a single thread walks
the hub. The slice labelled other host work, 6 to 11% for `gpu-nearfar` on the
uniform, layered and hub graphs, is the time it spends scanning all edge weights
on every query to pick its default Δ, outside the timed phases. In E that scan
took 1.2 ms at 2^20 vertices and 20.1 ms at 2^24 on the uniform graphs.

## Hardware counters (F)

Nsight Compute was run on representative launches of eight kernels, with one
Nsight Systems timeline and perf on the CPU solvers.

```
kernel                         SM active  occupancy  SM tail  L2 use  sectors/request  bottleneck
grid, unit, gpu-frontier             65%        16%      1.1      7%              1.1  too little parallelism
hub, gpu-topo                         3%        60%     40.1      1%             12.1  load imbalance
hub, gpu-frontier                    59%        47%     48.9     12%              1.3  load imbalance
layered W=256, gpu-frontier          90%        73%      1.1     18%              1.3  memory latency
uniform 2^22, gpu-frontier          100%        83%      1.1      9%              1.3  memory latency
uniform 2^22, gpu-nearfar           100%        78%      1.9     10%              1.3  memory latency
uniform 2^22, gpu-edge              100%        88%      1.0     67%              3.3  bandwidth
uniform 2^22, gpu-topo              100%        87%      1.0     58%             13.7  memory latency
```

SM tail is the busiest SM's active time divided by the average, and the
bottleneck labels come from fixed thresholds in `tools/analyze.py`. The grid
kernel keeps only 16% of the possible warps resident. Over four grid solves
Nsight Systems recorded 35 ms of kernel time and 6,009 blocking copies taking 97
ms. On the hub the busiest SM is active 40 to 49 times as long as the average.
`gpu-edge` on the uniform graph uses 67% of the L2 bandwidth, and `gpu-frontier`
on the uniform and layered graphs keeps most warps resident with little
bandwidth use, which the thresholds label memory latency.

The counters also show why `gpu-edge` beats `gpu-topo` on uniform graphs
although both are balanced. `gpu-topo` needs 13.7 memory sectors per load
request against 3.3, which fits each of its threads reading a different
adjacency list. The active lanes counter reads about 25 of 32 for `gpu-frontier`
on both the degree 4 grid and the degree 8 uniform graph.

On the CPU, `delta-omp` kept on average about one of its 24 threads busy on the
grid, so it ran almost serially, and about 22 of 24 on the large uniform graph.
There its instructions per cycle were 0.23 on P cores and 0.16 on E cores, with
9.4 last level cache misses per thousand instructions against 2.1 on the grid,
which points to memory latency. perf counted 37 to 60% of the serial solvers'
instructions on E cores, but pinning them to one P core changed their times by
about 1% at most.

## The cost model

I fitted GPU solve time as α times the number of synchronizations, plus a cost
per vertex expanded, plus a cost per edge touched, on experiment A only, and
then measured the median error on the other experiments. The A column is the
error on the data the model was fitted to.

```
solver         model                        α (µs)  ns/vertex  ns/edge     A     C     E    HO     B
gpu-frontier   syncs and edges                16.0               0.064   26%   43%   42%   41%   45%
gpu-frontier   syncs, vertices and edges      13.4       0.84    0.015    3%    5%   13%    6%   37%
gpu-nearfar    syncs and edges                 5.6               0.120    7%    8%    8%    8%
gpu-nearfar    syncs, vertices and edges       5.7       0.78    0.026    6%    7%    7%    7%
```

The model in the comment on the counters in `include/sssp/solver.hpp` has only
synchronizations and edges, and for `gpu-frontier` it is off by 26 to 45%. The
three term model adds the cost per vertex. I added that term after seeing the
two term fit, so only its errors outside A say anything about it. It gets
`gpu-frontier` within 5 to 13% on C, E and HO, and its α of 13.4 µs is close to
the 14.4 µs empty round measured in C0. It fails under heavy skew, with 37% on
B, where one thread or warp walking the hub dominates, and it is weakest on E,
whose largest graphs no longer fit in the L2. For `gpu-nearfar` both forms are
within 6 to 8%. `gpu-nearfar` was not run in B.

I also checked how well simple rules pick the faster device, on 260 graph and
source pairs from the 33 graphs in C, E and HO, none of which were used for
fitting. Extra time is the time lost against always choosing correctly, as a
share of the best possible total.

```
rule                                               accuracy  extra time
always CPU                                              37%         95%
always GPU                                              63%         25%
BFS frontier width, fitted on A                         86%          8%
graph properties and calibrated constants               88%          6%
model on the run's own counters, true CPU time          99%        0.1%
```

Graph properties and the calibrated constants pick the right device 88% of the
time, about as often as the frontier width alone. With a run's actual counters
and the true CPU time the model is right 99% of the time. What the graph does
not tell in advance is how much work an algorithm will waste on given weights,
which A and C show changing on identical edges, and how fast the best CPU solver
will be, since its time per useful edge varies 25 times across these graphs,
from 1.4 to 34 ns.

## Limitations

These results come from one CPU and one GPU. The GPU solvers synchronize with
the host every round, so the synchronization cost measured here belongs to that
design and not to GPUs in general. `gpu-nearfar` recomputes its default Δ on
every query by scanning the weights on the host outside the timed phases, which
the analysis counts as part of the one time cost, and I left the code alone so
it matches the data. The headline speedups leave out `gpu-edge`, which was
faster on many shallow graphs. The machine is shared. The experiment script
waits whenever another process is on the GPU, which did not happen during this
run, but CPU load from other users cannot be ruled out, and the load average is
recorded with every run. With eight sources per graph, graphs where the result
depends a lot on the source have wide confidence intervals, for example 3.1x to
4.0x for the held out RMAT graph against 4.8x to 4.9x for the uniform graph.
