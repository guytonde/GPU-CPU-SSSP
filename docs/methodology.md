# Methodology

## What each run records

Every time a solver runs, the benchmark records how long each phase took and
how much work the solver did. The phases are these.

```
prep_ms    host work to prepare inputs (only gpu-edge has any, an array of edge sources)
alloc_ms   device allocation, zero when a resident solver reuses its buffers
h2d_ms     copying the graph to the GPU, from pageable memory
solve_ms   from initializing distances to the last relaxation, including every synchronization
d2h_ms     allocating the host result and copying the distances back
free_ms    releasing device memory
wall_ms    everything the caller sees
```

For the GPU the cost of one more query, once the graph is on the device, is
solve plus download, and everything else counts as a one time cost. For the CPU
the cost of a query is its whole wall time. The resident speedup is the ratio
of per query costs and the one shot speedup is the ratio of wall times.

The work counters are the number of blocking synchronizations with the host,
the number of vertices expanded and the number of adjacency entries read, all
defined in `include/sssp/solver.hpp`. The tests check that on a unit weight
graph `gpu-frontier` expands exactly the reachable vertices and reads exactly
their edges, in exactly as many rounds as there are BFS levels, with one
synchronization per round. The GPU counters can be switched off with
`--no-counters`.

## The model

The GPU time for one query is modelled as α times the number of
synchronizations, plus a cost per vertex expanded, plus a cost per edge read.
α is also measured on its own by `bin/calibrate`, so the fitted value can be
checked against it. The measure of parallelism in the plots is Π, the number of
edges leaving reachable vertices divided by the hop depth of the shortest path
tree. That depth is the number of rounds a solver needs when each round extends
paths by one edge, which is how the frontier solvers work. Solvers that update
in place during a sweep, like `bellman-ford`, can need fewer when paths follow
the vertex numbering.

## Graphs and sources

`bin/gen_graph` makes layered graphs for experiment A, planted hub graphs for
experiment B, and uniform, RMAT, grid and geometric graphs for the rest. No
graph gets a spanning tree, and the tests check that a request for zero edges
really gives an empty graph. The topology, the weights and the vertex numbering
come from three separate random seeds, so changing the weights leaves the edges
bit for bit identical. The tests check this for every topology, and the
validator checks it again on the study's graphs with a hash of the topology.
Vertices are numbered randomly unless a run asks for the generator's natural
order. The generator does its own sampling on top of `mt19937_64` instead of
using the standard library distributions, whose output differs between
implementations, and every graph file stores the parameters that made it.

Sources for layered graphs come from the first layer. Everywhere else they are
random vertices of nonzero degree in the largest connected component. Every
solver on a graph uses the same sources, so all comparisons are paired.

## Keeping the measurements clean

Each solver runs once untimed on each graph before timing starts, except in the
CPU batch runs of D. Then it gets five timed repetitions per source, or three
for graphs of 2^23 vertices and up, for the bucket width sweep in C and for the
instrumented runs. The order of the solvers is shuffled on every repetition.
CUDA kernels are loaded when the program starts rather than on first use. The
GPU sits idle while the CPU solvers run, so before every GPU run the benchmark
spins the GPU until a kernel with a known cycle count shows it is near full
clock, and it records the clock it measured. The CPU gets no such warm up. Every GPU run records whether another
process was using the GPU, and the experiment script waits until the GPU is
free before each benchmark.

The CPU thread count was chosen once by measurement in C0, as described in
`docs/results.md`. Caches are left warm, and experiment E repeats three sizes
with the caches flushed to check what that changes. The time breakdown runs put
an event around every kernel. Compared with the same graphs and sources run
without events, that made the GPU solvers up to 22% slower and changed the CPU
solvers by at most 4%, so those runs are used only for the breakdown.

## Correctness

There are two references and neither shares code with the solvers. One is a
textbook Bellman Ford with no early exit and the other is a Dijkstra that
orders by distance and then hop count. The tests check the second against the
first on 200 random graphs. When n times m is at most 2x10^8 both references
run in the benchmark and must agree. Every solver run is hashed and compared
with the reference, and any mismatch fails the run. `make test` runs 5,740
checks.

## Statistics

A solver's time on a graph and source is the median of its repetitions. The
GPU against CPU ratio is computed per source and then averaged over sources with
a geometric mean, and its 95% confidence interval comes from resampling the
sources 2,000 times. The break even query count comes from κ, the per query
saving divided by the upload cost, which avoids dividing by a saving close to
zero. The cost model is fitted on experiment A by nonnegative least squares on
relative error, since the times in A range from about 1 ms to almost 4 s, and it is judged
by its median absolute percentage error on the other experiments. No samples
are thrown away as outliers.

## Validation

`tools/analyze.py` writes `results/published/validation.md` and stops there if
a hard check fails. The hard checks are that every sample belongs to a known run
and source, that every distance array matched the reference, that changing the
weights never changed a topology, that layered graphs are as deep as their
layer count implies, that no grid is deeper than its rows plus columns, that
large geometric graphs are more than 100 levels deep, that no GPU run shared the
GPU, and that no sample appears twice. It warns when the timed phases cover
less than 90% of a run's wall time, when a GPU run started below 90% of the
median clock, and when the code was not committed.

Each run records the commit its binaries were built from and a hash of any
uncommitted changes to `src`, `include` and the Makefile. The study was run from
commit e7d807f with uncommitted changes whose hash is c9083752a348, and every run
has the same hash.

## Mistakes in the earlier version

The first version of this project, whose README is still on the main branch,
ran on a T1000 and an i5 14600, and none of its numbers are used here. Several
of its mistakes shaped the rules above. Its generator added a random spanning
tree to every graph for connectivity, which on a grid adds long range
shortcuts. This was noticed because Bellman Ford converged in 7 rounds on a 316
by 316 grid whose diameter is 630, and on a 1000 by 1000 grid delta stepping
needed 1,999 phases without the shortcuts and 11 with them. The fix removed the
tree from grids but kept it on geometric graphs. The benchmark always included
vertex 0 as a source, which is a corner of a grid and, since the generator did
not renumber vertices, a hub of an RMAT graph. With several sources,
every CSV row recorded the round count of the last one. What had been reported
as transfer time for `gpu-edge` was mostly a single threaded loop on the host,
36.3 of 40.9 ms in one run. On a shared two socket Xeon with 48 logical CPUs,
running 48 OpenMP threads made an empty parallel region take 6.5 ms, against
11.7 µs with 24. And the weights came from the same random stream as the edges,
so changing the weight distribution changed the graph.
