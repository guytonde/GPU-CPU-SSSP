# GPU-CPU-SSSP

Single-source shortest paths implemented from scratch in C++ and CUDA, to find
out where a GPU actually beats a CPU on graph work and where it does not.

The interesting question is not "is the GPU faster". It is which combination of
graph shape, weight distribution, and algorithm makes the GPU worth the PCIe
round trip at all. SSSP is a bad fit for a GPU in the ways that matter most:
the work is memory bound, irregular, and inherently sequential in its ordering.
Every implementation here is a different attempt to get around one of those.

Nine solvers, one interface, and a bench that runs all of them on the same
graph and checks every distance array against an independent O(V*E) oracle.
[Results from the full sweep](#results-the-full-sweep) are at the bottom; the
short version is that on this hardware the GPU wins the kernel, loses the wall
clock on one-shot queries, and wins it back once the graph stays resident.

## What is implemented

| solver | device | idea |
| --- | --- | --- |
| `dijkstra` | cpu | binary heap with lazy deletion. The reference. |
| `dial` | cpu | bucket queue. No log factor, but O(n * maxw) bucket scanning. |
| `bellman-ford` | cpu | relaxation rounds with early exit. Serial twin of `gpu-topo`. |
| `delta` | cpu | delta-stepping. Trades some redundant relaxation for parallelism. |
| `delta-omp` | cpu | delta-stepping across cores, lock-free CAS on the distance array. |
| `gpu-topo` | gpu | thread per vertex, every vertex every round. The naive port. |
| `gpu-edge` | gpu | thread per edge. Same work, even distribution across threads. |
| `gpu-frontier` | gpu | worklist. Warp per vertex, warp-aggregated queue pushes. |
| `gpu-nearfar` | gpu | near-far pile. Delta-stepping's bucket rule on the GPU. |

Each one is a single translation unit. Adding a tenth means writing the file and
adding one line to `src/core/registry.cpp`.

## Quick start

```bash
git clone <this repo> && cd GPU-CPU-SSSP
make                                    # builds bin/bench, bin/gen_graph, bin/test_sssp
make test                               # ~5500 assertions against an independent oracle
make smoke                              # generates a small graph and runs all solvers
```

`make test` is the correctness check and `make smoke` the end-to-end one. If the
first prints `0 failures` and the second prints a table with `yes` down the `ok`
column, the build is good.

To go straight to a real measurement:

```bash
./bin/gen_graph --n 1000000 --m 4000000 --topo rmat --wmax 100 --out graphs/r.bin
./bin/bench graphs/r.bin --reps 3
```

## Layout

```
include/sssp/     graph, solver interface, registry, timer, reference oracle
src/core/         CSR construction, graph IO, statistics, the registry, the oracle
src/cpu/          the five CPU solvers, plus thread-count policy
src/gpu/          cuda_common.cuh and the four .cu solvers
src/apps/         bench and gen_graph
tests/            test_sssp.cpp
tools/            sweep.sh, summarize.py
```

`Graph` is CSR and nothing else. Solvers that need something different, like the
edge-source array for `gpu-edge`, derive it themselves and are charged for it in
the `prep` column, separately from the PCIe transfer.

## Building

Requirements: a C++17 compiler with OpenMP, GNU make, and optionally a CUDA
toolkit. Nothing else; there are no third-party dependencies.

| command | what it does |
| --- | --- |
| `make` | cpu and gpu solvers |
| `make cpu` | cpu only (`NO_CUDA=1`), same binary minus the four gpu solvers |
| `make test` | build and run the test suite |
| `make smoke` | build, generate a small graph, run every solver once, run the tests |
| `make sweep` | build, then run `tools/sweep.sh` |
| `make info` | which nvcc, host compiler and archs got picked |
| `make clean` | remove `build/` and `bin/` |
| `make distclean` | also remove `graphs/` and `results/` |

`make` probes for a usable `nvcc` rather than trusting `$PATH`, because a
toolkit older than the system glibc dies inside `<stdlib.h>` long before it sees
any of this code. It compiles a trivial `.cu` with each candidate and takes the
first that survives. If none do, the build falls back to CPU-only and the GPU
solvers drop out of the registry; the bench then skips them with a note instead
of failing.

`ARCHS` defaults to `70 75 80 86`, which covers most cards you are likely to
have and costs a little compile time. Narrow it to the one you are benchmarking
on for faster builds, or widen it for a card the default misses:

```bash
make ARCHS="75"       # T1000, RTX 20xx
make ARCHS="80"       # A100
make ARCHS="89"       # Ada consumer parts
```

Check what the probe decided before blaming the code:

```console
$ make info
nvcc:    /lusr/opt/cuda-12.5/bin/nvcc
version: Build cuda_12.5.r12.5/compiler.34385749_0
host cc: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
archs:   70 75 80 86
threads: 20
```

## Generating graphs

```bash
./bin/gen_graph --n 1000000 --m 8000000 --topo rmat --wmax 1000 --out graphs/r.bin
```

| flag | default | meaning |
| --- | --- | --- |
| `--n N` | 10000 | vertices |
| `--m M` | 100000 | edges, before the undirected doubling |
| `--topo T` | uniform | `uniform`, `rmat`, `grid`, `geometric`, `chain`, `star` |
| `--wdist D` | uniform | `uniform`, `logunif`, `unit` |
| `--wmin W` | 1 | minimum weight |
| `--wmax W` | 100 | maximum weight |
| `--seed S` | 12345 | rng seed |
| `--directed` | off | emit each edge once instead of both ways |
| `--connect` / `--no-connect` | auto | force the spanning tree on or off |
| `--keep-dups` | off | do not collapse parallel edges |
| `--out PATH` | graphs/out.txt | `.bin` suffix writes packed CSR |

Four realistic topologies, chosen because they stress different parts of the
problem, plus two deliberately pathological ones:

- `uniform` Erdos-Renyi. Even degrees, small diameter. The easy case, and the
  one every SSSP paper quietly benchmarks on.
- `rmat` Graph500 parameters. Power-law degrees, so a thread-per-vertex kernel
  has one lane grinding through a hub while 31 neighbours idle. This is what
  separates `gpu-topo` from `gpu-frontier`.
- `grid` 2D mesh. `n` rounds to the nearest square and `m` is ignored. The hop
  distance from a corner grows as sqrt(n), so the frontier stays tiny and there
  are thousands of rounds. Worst case for the GPU by a wide margin.
- `geometric` random points joined within a radius. Roughly road shaped.
- `chain` a path graph. `m` is ignored. Pure sequential depth: every round moves
  the frontier by exactly one vertex. This is the floor for anything parallel.
- `star` one hub joined to everything. `m` is ignored. Pure degree skew: one
  vertex holds every edge in the graph.

`chain` and `star` are not meant for the sweep. They are microbenchmarks that
isolate one property each, so that a result on `rmat` — which mixes skew, small
diameter and hubs together — can be attributed to the right cause. See [two
microbenchmarks](#two-microbenchmarks).

Weights come from `--wdist uniform|logunif|unit` over `[--wmin, --wmax]`. The
spread matters, and not subtly: `dial` is only competitive when `maxw` is small,
and delta-stepping's bucket width is derived from it. `--wmax 1` makes every
edge weight 1, which turns the problem into BFS and makes `gpu-nearfar`
degenerate to `gpu-frontier`.

For `uniform`, `rmat` and `geometric` the generator lays a random spanning tree
over the vertices so that every vertex is reachable from vertex 0. Without it a
sparse random graph leaves a large unreachable tail and the timings measure the
wrong thing.

`grid`, `chain` and `star` are connected by construction, so they do **not** get
one. This is not a detail: on a mesh, a spanning tree is n-1 random long-range
edges, which is 50% more edges and collapses the one property the mesh exists to
test. That used to be opt-out and it silently invalidated an entire sweep — see
[the grid correction](#the-grid-topology-was-not-testing-what-it-looked-like).
`--connect` and `--no-connect` override the default in either direction, and the
generator now prints which it chose:

```console
$ ./bin/gen_graph --n 100000 --topo grid --wmax 100 --out g.bin
wrote g.bin  topo=grid n=99856 arcs=398160 avg_deg=3.98734 ... spanning_tree=no
$ ./bin/gen_graph --n 100000 --topo grid --wmax 100 --connect --out g.bin
wrote g.bin  topo=grid n=99856 arcs=597862 avg_deg=5.98724 ... spanning_tree=yes
```

Weights must be non-negative: every solver here either settles in nondecreasing
distance order or relies on relaxation converging, and neither survives a
negative edge. The loader rejects one rather than returning a plausible wrong
answer.

Output is a text edge list, or packed CSR if the filename ends in `.bin`. Use
`.bin` for anything above a few million edges; parsing text dominates otherwise.
The binary layout is a 16-byte header, then `(n+1)` int32 offsets, then `m`
int32 targets, then `m` int32 weights, so on-disk size is roughly `4n + 8m`
bytes:

| graph | arcs | .bin size |
| --- | ---: | ---: |
| `--n 100000 --m 400000` | 800k | 6.8 MB |
| `--n 1000000 --m 4000000` | 8M | 68 MB |
| `--n 4000000 --m 16000000` | 32M | 272 MB |
| `--n 4000000 --m 64000000` | 128M | 1.0 GB |

The generator is seeded, so a graph rebuilds byte for byte from its parameters.
Regenerating is usually cheaper than storing.

## Running the bench

```bash
./bin/bench graphs/r.bin --reps 5
./bin/bench graphs/r.bin --sources 20 --stats
./bin/bench graphs/r.bin --sources 100 --resident
./bin/bench graphs/r.bin --only dijkstra,gpu-frontier --csv results/run.csv --tag rmat1m
./bin/bench --list
```

| flag | default | meaning |
| --- | --- | --- |
| `--source V` | 0 | source vertex, and the first of a `--sources` list |
| `--sources N` | 1 | run N sources: `--source` plus N-1 sampled at random |
| `--reps N` | 5 | repetitions per source; the **median** is reported |
| `--resident` | off | keep the graph on the device between sources |
| `--delta D` | auto | bucket width for `delta`, `delta-omp` and `gpu-nearfar` |
| `--threads N` | physical cores | OpenMP threads |
| `--seed S` | 1 | seed for source sampling and solver order |
| `--no-shuffle` | – | run solvers in registry order |
| `--oracle` / `--no-oracle` | auto | force the O(V*E) correctness check on or off |
| `--stats` | – | print per-source search statistics |
| `--only a,b,c` | all | run only these solvers |
| `--csv PATH` | – | append one row per solver per source |
| `--tag NAME` | – | label written into the csv rows |
| `--list` | – | print the solver table and exit |

### Reading the output

```console
$ ./bin/bench graphs/rmat1m.bin --reps 5 --only dijkstra,gpu-topo,gpu-edge,gpu-frontier,gpu-nearfar
graphs/rmat1m.bin: n=1000000 m=9784328 avg_deg=9.8 max_deg=22157 deg_sd=73.6 w=[1,100]
gpu: NVIDIA T1000 8GB (sm_75, 14 SMs)
cpu: 14 omp threads
sources=1 reps=5 resident=no order=shuffled check=cross-solver

solver        dev       prep      h2d     solve      d2h     total   rounds    MTEPS  speedup  ok
----------------------------------------------------------------------------------------------
dijkstra      cpu      0.000    0.000   519.098    0.000   519.098  1000000     18.8    1.00x  yes
gpu-topo      gpu      0.000    9.303   143.331    0.446   152.850       10     68.3    3.62x  yes
gpu-edge      gpu     36.268   13.967    80.469    0.442   131.166        9    121.6    6.45x  yes
gpu-frontier  gpu      0.000    9.312    30.988    0.451    40.746       15    315.7   16.75x  yes
gpu-nearfar   gpu      0.000    9.332    31.022    0.441    40.808      169    315.4   16.73x  yes

medians over 5 reps x 1 source, speedup against dijkstra

solve ms spread
solver                min        p25     median        p75       max
dijkstra          517.012    517.903    519.098    519.994    520.715
gpu-topo          142.962    143.077    143.331    143.377    144.261
gpu-edge           80.310     80.361     80.469     80.568     82.502
gpu-frontier       30.966     30.985     30.988     31.016     32.204
gpu-nearfar        30.950     31.021     31.022     31.078     32.238
```

The cost of a run is split into four phases rather than "solve" and "transfer":

- **`prep`** is host-side work a solver does to build inputs `Graph` does not
  carry. Only `gpu-edge` has any: its per-arc source array.
- **`h2d`** is the CSR upload, measured with CUDA events.
- **`solve`** is the algorithm. For GPU solvers it is kernel time, including the
  per-round synchronisation the algorithm actually needs.
- **`d2h`** is the distance download.
- **`rounds`** means different things per solver by design: heap pops for
  `dijkstra`, bucket scans for `dial`, relaxation sweeps for `bellman-ford` and
  `gpu-topo`, frontier expansions for the worklist kernels. It is diagnostic,
  not a cost.
- **`MTEPS`** is `m / solve`, so it is a normalisation by graph size and not a
  count of edges actually traversed. A frontier solver that touches a tenth of
  the edges still gets charged all of `m`.
- **`ok`** is the correctness check. `ref` marks `dijkstra` on graphs too big
  for the oracle, where it supplies the reference and so is not itself checked.

Splitting `prep` out of transfer corrects one of this project's own conclusions.
`gpu-edge` was described as paying "triple for transfer". In the run above it
spends 40.9 ms more than the other GPU solvers before its kernel starts — and
**36.3 ms of that is a host-side `for` loop** building the per-arc source array,
against 4.7 ms of extra PCIe traffic. Nearly 90% of what was reported as a bus
cost is single-threaded CPU work, and it was invisible while the two shared a
column.

Three methodology points the numbers depend on:

**The median is reported, not the best run.** Best-of-N preferentially selects
the run that happened to hit a quiet moment, which flatters whichever solver is
most sensitive to interference. The `solve ms spread` block underneath the main
table prints min/p25/median/p75/max so the variance is visible rather than
discarded. On a shared machine it is frequently the more interesting number: a
`dial` median of 20 ms with a max of 52 ms is a different claim from "20 ms".

**Solver order is shuffled.** Registry order lets whichever solver runs first
absorb the cache and clock effects of everything before it. The order is
reshuffled per repetition from `--seed`, so it is reproducible without being
fixed. `--no-shuffle` restores the old behaviour.

**Correctness is checked against an independent oracle.** `src/core/reference.cpp`
is a textbook O(V*E) Bellman-Ford with no early exit and no data structure worth
getting wrong. The bench runs it automatically when `n*m` is small enough to
afford, and on larger graphs falls back to `dijkstra` — pinned, so the shuffled
order cannot decide which solver goes unchecked. The header line says which was
used. Cross-checking alone cannot catch an error that every solver shares, which
is why `make test` runs everything against the oracle on graphs small enough to
afford it.

**Thread count defaults to physical cores, not logical.** See
[the thread count trap](#the-thread-count-trap).

### Pinning threads

The default thread count is the physical core count, but nothing here pins
threads to cores. On the i5-14600 the 14 threads land across 6 P-cores and 8
E-cores, whose single-thread throughput differs by roughly 2x, so an OpenMP
barrier waits on whichever E-core drew the longest chunk. `delta-omp` uses
`schedule(dynamic, 64)` partly for that reason. If you want the placement fixed:

```bash
OMP_PROC_BIND=close OMP_PLACES=cores ./bin/bench graphs/r.bin
taskset -c 0-11 ./bin/bench graphs/r.bin --threads 6   # P-cores only
```

Worth doing before drawing conclusions about parallel scaling on a hybrid CPU.
It is left out of the binary on purpose: pinning is wrong on a shared machine,
which is exactly where this was developed.


## Sweeping the whole matrix

```bash
tools/sweep.sh
tools/summarize.py results/sweep_<stamp>.csv
```

Varies topology, size, degree, and weight range, and appends every run to one
csv. Override any axis from the environment:

| variable | default |
| --- | --- |
| `TOPOS` | `uniform rmat geometric grid` |
| `SIZES` | `100000 1000000 4000000` |
| `DEGREES` | `8 32` |
| `WEIGHTS` | `1 100 100000` |
| `REPS` | `5` |
| `SOURCES` | `5` |
| `RESIDENT` | `0` |
| `CSV` | a fresh `results/sweep_<stamp>.csv` |
| `KEEP` | `0` |
| `BIN`, `OUT`, `GRAPHS` | `bin`, `results`, `graphs/sweep` |

That matrix is 63 configurations, and at the default 5 sources x 5 reps it is
just under 14,000 solver runs. `grid`, `chain` and `star` ignore `--m`, so they
run once per (size, weight) rather than once per degree. Drop `SOURCES=1
REPS=3` for a quick pass.

Every row of the csv carries the graph's shape alongside the timing —
`avg_deg`, `max_deg`, `deg_stddev`, and per source `source_deg`, `levels`,
`reached`, `max_frontier`, `avg_frontier` — so runtime can be correlated against
the property that caused it rather than against the topology name. Topology
names conflate several things at once: `grid` is simultaneously low-degree,
high-eccentricity and narrow-frontiered, and only the columns tell you which one
a result is about.

The full matrix is about 16 GB of graphs, so the sweep generates each one,
benches it, and deletes it, holding at most a single graph on disk. The
generator is seeded, so any row rebuilds byte for byte from its tag. `KEEP=1`
retains them, and graphs already present when the sweep starts are never
deleted.

An interrupted sweep is finished off rather than restarted, by pointing `CSV` at
the existing file and narrowing the axes to what is missing:

```bash
CSV=results/sweep_20260908_135038.csv TOPOS=grid tools/sweep.sh
```

For the amortised measurement, run the same matrix with the graph kept on the
device:

```bash
RESIDENT=1 SOURCES=100 CSV=results/resident.csv tools/sweep.sh
```

`tools/summarize.py` pivots the csv into one row per graph and one column per
solver, marking any incorrect result rather than printing its time.

Note that `results/` and `graphs/` are both gitignored, so a clone starts with
neither; the sweep is the way to regenerate them.

## Results: the full sweep

All numbers below are from one machine:

| | |
| --- | --- |
| CPU | Intel Core i5-14600, 14 physical cores (6 P + 8 E), 20 logical, 1 socket |
| GPU | NVIDIA T1000 8GB, sm_75, 14 SMs |
| toolkit | CUDA 12.5, g++ 13.3 |

The T1000 is a small workstation card, not a datacenter part. That matters for
reading everything below: the CPU here is a current mid-range desktop chip and
the GPU is entry-level, so this is close to the least favourable hardware
pairing for the GPU. A100 numbers would look very different. What does transfer
across hardware is the *shape* of the results, not the ratios.

Every one of the runs summarised here passed the correctness check.

**These tables predate the current reporting.** They were collected with
best-of-3 from `source=0`, before the median, the source sweep and the
prep/h2d/solve/d2h split existed. The shape of the results is what matters and
that has held up on re-runs, but the individual figures are a best case rather
than a median, and the `xfer` figures for `gpu-edge` bundle in host-side work
that is now reported separately. Regenerating them is `tools/sweep.sh`, a few
hours.

### The headline

Comparing the best CPU solver against the best GPU solver in each configuration:

| comparison | median | configs won by GPU |
| --- | ---: | ---: |
| kernel time only (`solve ms`) | 1.11x | 41 / 54 |
| including transfer (`total ms`) | 0.78x | 9 / 54 |

Those 54 are the `uniform`, `rmat` and `geometric` configurations. The nine
`grid` ones are excluded and treated separately, because as generated they were
not testing what they claimed to — see [the grid
correction](#the-grid-topology-was-not-testing-what-it-looked-like). On a
corrected mesh the GPU wins **0 of 9**, on either measure: a median of 0.20x on
kernel time and 0.17x including transfer.

The GPU usually wins the kernel and usually loses the wall clock. Transfer is a
median 30% of `total ms` for the winning GPU solver, and as much as 80%. For
one-shot SSSP on a graph that starts in host memory, this hardware pairing does
not pay for itself. It only pays if the graph is already resident on the device,
or if you are running many queries against one upload — which is the actual
argument for GPU SSSP, and it is an argument about amortisation, not about
kernels.

`delta-omp` was the fastest CPU solver in **all 54** configurations. Among the
GPU solvers the winner splits by weight range: `gpu-nearfar` took 29,
`gpu-frontier` 19, `gpu-edge` 6.

### Representative numbers

Two slices of the matrix, chosen because they bracket the interesting range.
Grid is discussed separately below, for reasons that turn out to matter.

**n = 4,000,000, degree 8, weights [1,100]** (solve ms, best of 3)

| solver | uniform | rmat | geometric |
| --- | ---: | ---: | ---: |
| `dijkstra` | 2,129 | 1,593 | 2,160 |
| `dial` | 855 | 480 | 865 |
| `bellman-ford` | 1,963 | 818 | 1,978 |
| `delta` | 1,424 | 980 | 1,509 |
| `delta-omp` | 204 | **138** | 202 |
| `gpu-topo` | 1,735 | 678 | 1,964 |
| `gpu-edge` | 1,629 | 583 | 2,065 |
| `gpu-frontier` | 488 | 155 | 484 |
| `gpu-nearfar` | **159** | 141 | **151** |

Bold is the fastest solver in each column.

**n = 1,000,000, degree 32, weights [1,100]** (solve ms, best of 3)

| solver | uniform | rmat | geometric |
| --- | ---: | ---: | ---: |
| `dijkstra` | 606 | 385 | 607 |
| `dial` | 269 | 141 | 257 |
| `bellman-ford` | 797 | 226 | 823 |
| `delta` | 534 | 254 | 466 |
| `delta-omp` | **77** | 55 | **80** |
| `gpu-topo` | 1,309 | 253 | 1,514 |
| `gpu-edge` | 1,117 | 170 | 1,291 |
| `gpu-frontier` | 358 | 50 | 257 |
| `gpu-nearfar` | 89 | **48** | 81 |

Note the degree-32 row for `gpu-topo` and `gpu-edge`: at 32M arcs they are
*slower than serial Dijkstra* on uniform and geometric. Re-reading every edge
every round is not a constant factor you can optimise away, it is the wrong
complexity, and the GPU's bandwidth advantage does not cover it.

### What each GPU solver was predicted to do

These predictions were written down before the sweep ran. Scoring them against
54 non-grid configurations:

| prediction | result |
| --- | --- |
| `gpu-topo` loses on large diameter, acceptable where rounds are in the teens | **partly refuted.** It was last of the four in every uniform and rmat config and 3.78/4 on geometric, even at 9–21 rounds. "Acceptable" was too generous. |
| `gpu-edge` beats `gpu-topo` on rmat specifically, ties on uniform | **confirmed for rmat, refuted for uniform.** Median 1.79x on rmat (max 10.0x), but it still wins 1.14x on uniform rather than tying. Load balancing helps everywhere; it just helps far more under degree skew. |
| `gpu-frontier` is best of the four on small-diameter graphs | **confirmed, but only at unit weights.** It won 17 of 18 configs at `wmax=1` and 1 of 18 at each wider range. |
| `gpu-nearfar` beats `gpu-frontier` when the weight range is wide, converging at unit weights | **confirmed.** Median 1.59x at `wmax=100` and 1.79x at `wmax=100000`, falling to 0.89x at `wmax=1` — i.e. at unit weights near-far is marginally *worse*, which is what "degenerates to BFS plus bucket bookkeeping" should look like. |

The one genuinely wrong prior was about `gpu-topo`. The intuition was that a low
round count would rescue it. It does not, because the per-round cost is O(m) and
the frontier kernels' per-round cost is proportional to the frontier, which on
these graphs is a small fraction of m for all but two or three rounds.

### The transfer tax

`xfer ms` is the whole argument. For the winning GPU solver it is a median 30%
of `total ms`, ranging from 16% to 80%. `gpu-edge` is worst affected because it
uploads a per-arc source array on top of the CSR — on `rmat/n1000000/deg8/w100`
that is 32.4 ms of transfer against 80.4 ms of kernel.

So the GPU's 41/54 win on kernel time becomes 9/54 on wall clock. Every one of
those nine is a large, low-degree graph where the kernel advantage is big enough
to outrun the upload. The conclusion is not "the GPU is slow" — `gpu-nearfar` is
genuinely the fastest solver here by kernel time on most shapes. It is that
one-shot SSSP from host memory is the wrong thing to use it for.

### Does the GPU ever pay for itself?

The transfer tax is only a tax if you pay it once per query. `--resident` keeps
the CSR on the device between sources, which is what any real application doing
more than one shortest-path query would do, and the bench prints the break-even
directly:

```console
$ ./bin/bench graphs/rmat1m.bin --reps 3 --sources 5 --resident
amortised over one upload
solver           upload ms   per query ms      queries to beat dial
dijkstra             0.000        522.858                         -
dial                 0.000        145.843                         -
bellman-ford         0.000        346.217                         -
delta                0.000        340.022                         -
delta-omp            0.000        171.582                         -
gpu-topo             7.564        164.744                     never
gpu-edge            47.695         90.355                         1
gpu-frontier         7.734         39.825                         1
gpu-nearfar          7.616         35.066                         1
```

`upload ms` is the one-time cost, `per query ms` is kernel plus download, and
the last column is the smallest K for which `upload + K*gpu < K*cpu`.

On `rmat/n1000000/deg8/w100` the answer is 1: `gpu-nearfar` costs 7.6 + 35.1 =
42.7 ms for a single query against `dial`'s 145.8 ms, so the upload is paid off
before the first query finishes. On a 1000x1000 mesh, the same table says
`never` for all four GPU solvers — the per-query cost is worse than the CPU's,
so no amount of amortisation helps:

```console
$ ./bin/bench graphs/mesh1m.bin --reps 3 --sources 5 --resident
amortised over one upload
solver           upload ms   per query ms      queries to beat dial
dijkstra             0.000        106.681                         -
dial                 0.000         22.443                         -
bellman-ford         0.000       2731.282                         -
delta                0.000         35.237                         -
delta-omp            0.000         33.431                         -
gpu-topo             3.569        412.011                     never
gpu-edge            15.494        760.200                     never
gpu-frontier         3.629         38.454                     never
gpu-nearfar          3.536         97.157                     never
```

This is the experiment that the "it only pays if the graph is already resident"
claim needed. It turns out to be two separate claims, and only one of them is
about transfer: where the GPU wins per query, residency is not even required at
this size; where it loses per query, residency does not rescue it. The
interesting cases are the middle ones, where `upload / (cpu - gpu)` lands
somewhere between 2 and a few hundred, and the break-even column is the only way
to see them.

### Source sensitivity

`--sources N` runs the same graph from N sources and reports the distribution.
It exists because `source=0` is part of the experimental setup, not a property
of the algorithm, and on some graphs the difference is enormous:

```console
$ ./bin/bench graphs/mesh1m.bin --reps 3 --sources 6 --only dijkstra,dial,bellman-ford,delta-omp,gpu-frontier
solve ms spread
solver                min        p25     median        p75       max
dijkstra          103.693    105.754    111.940    113.716    122.000
dial               20.136     20.723     22.993     24.563     25.818
bellman-ford       11.285    264.099   3011.838   4038.140   5124.742
delta-omp          32.824     33.437     34.551     35.248     36.219
gpu-frontier       30.363     32.642     37.079     43.438     44.945

sources
      vertex   degree   levels     reached  max frontier avg frontier
           0        2     1999     1000000          1000         500.3
      133876        4     1743     1000000          1123         573.7
      136407        4     1456     1000000          1136         686.8
      451214        4     1334     1000000          1332         749.6
       21024        4     1954     1000000          1021         511.8
      350898        4     1548     1000000          1101         646.0
```

`bellman-ford` spans **454x** across six sources of the same graph: 11 ms from
vertex 0 and 5,125 ms from a random interior vertex. Everything else moves by
less than 1.5x. The cause is vertex numbering — a mesh is stored row-major, so
from the top-left corner every shortest path runs monotonically increasing in
vertex index and one forward sweep settles the whole graph. That is a property
of the benchmark, not of Bellman-Ford, and reporting the `source=0` number alone
would have been reporting an artifact.

The `sources` block also reports `levels`, the **hop eccentricity** of the
source: the number of BFS levels needed to reach everything from it. This is not
the graph diameter, which is the maximum over all pairs; it is a lower bound on
it. Earlier versions of this README used "diameter" for both. The distinction
shows up right here: the same 1000x1000 mesh has 1999 levels from a corner and
1334 from an interior vertex, while its diameter is a single number.

### On the CPU side

`delta-omp` was fastest in all 54 non-grid configurations, a median 6.9x over
`dijkstra` (min 3.1x, max 12.2x) on 14 cores. On a true mesh it is not: the
frontier is too narrow to be worth 14 threads, and `dial` or `bellman-ford` win
instead.

`dial` behaves exactly as its complexity says. Against `dijkstra` it is a median
2.74x at `wmax=100` but only 2.16x at `wmax=100000`, and its bucket-scan count
goes from a median of 252 to 123,167 — the ring is `maxw + 1` entries and
finding the next non-empty bucket dominates. At `wmax=1` it is barely ahead
(1.14x) because there is nothing for the bucket structure to exploit.

`bellman-ford` is the serial control for `gpu-topo` and loses for the same
reason: on `uniform/n1000000/deg32/w100` it takes 797 ms across 17 rounds, worse
than `dijkstra`'s 606 ms, because each round sweeps all 32M arcs whether or not
anything changed.

### The grid topology was not testing what it looked like

The `grid` rows in the main sweep are not measuring a mesh, and this was only
caught by noticing that Bellman-Ford was converging in 7 rounds on a graph whose
diameter should be 630.

`gen_graph` used to apply its spanning-tree connectivity pass to every topology.
For `uniform`, `rmat` and `geometric` that is necessary — a sparse random graph
otherwise leaves an unreachable tail. For `grid` it is not, because a mesh is
connected by construction, and what it actually did was layer n-1 *random
long-range edges* on top of the 199,080 real mesh edges. That is 50% more edges,
and it turns the mesh into a small-world graph. Delta-stepping needs 631 phases
on the true 316x316 mesh and 9 with the shortcuts in place.

So the main sweep's grid columns measure a topology that does not exist on
purpose, and every grid prediction in this README was untested by it.

The fix is in the generator, not in the documentation: `grid`, `chain` and
`star` no longer get a spanning tree unless `--connect` asks for one, and the
generator prints `spanning_tree=yes|no` on every run. A default that silently
invalidates a whole topology is a bug, and documenting it was never going to be
enough. The numbers below were produced with the equivalent of today's default:

**n = 1,000,000 mesh (1000x1000), weights [1,1]** — solve ms and rounds

| solver | mesh ms | mesh rounds | with shortcuts ms | shortcut rounds |
| --- | ---: | ---: | ---: | ---: |
| `dijkstra` | 50.3 | 1000000 | 104.5 | 1000000 |
| `dial` | 9.1 | 1999 | 69.1 | 11 |
| `bellman-ford` | 4.2 | 2 | 112.2 | 8 |
| `delta` | 14.4 | 1999 | 100.6 | 11 |
| `delta-omp` | 15.3 | 1999 | 19.3 | 11 |
| `gpu-topo` | 431.9 | 1998 | 25.0 | 9 |
| `gpu-edge` | 833.7 | 1825 | 26.9 | 9 |
| `gpu-frontier` | 32.4 | 1999 | 14.5 | 11 |
| `gpu-nearfar` | 93.2 | 3997 | 17.0 | 21 |

**n = 1,000,000 mesh (1000x1000), weights [1,100]** — solve ms and rounds

| solver | mesh ms | mesh rounds | with shortcuts ms | shortcut rounds |
| --- | ---: | ---: | ---: | ---: |
| `dijkstra` | 107.1 | 1000000 | 306.2 | 1000000 |
| `dial` | 23.9 | 45889 | 88.0 | 341 |
| `bellman-ford` | 745.2 | 185 | 255.7 | 20 |
| `delta` | 47.5 | 1874 | 157.0 | 24 |
| `delta-omp` | 46.3 | 1874 | 27.0 | 25 |
| `gpu-topo` | 463.3 | 2079 | 88.6 | 23 |
| `gpu-edge` | 989.3 | 2066 | 87.1 | 23 |
| `gpu-frontier` | 348.5 | 2079 | 44.9 | 28 |
| `gpu-nearfar` | 172.9 | 9180 | 23.3 | 124 |

Now the rounds match the geometry: 1999 delta-stepping phases on a 1000x1000
mesh, against its 2*(side-1) = 1998 hop diameter, and against 11 phases with the
shortcuts in place.

And the prediction holds emphatically. On a real mesh the GPU loses to the CPU
outright, on kernel time alone, before transfer is even counted:

- `gpu-topo` and `gpu-edge` collapse to 432 ms and 834 ms at unit weights, 17x
  and 31x worse than they looked with the shortcuts. Two thousand rounds of
  re-reading all 4M arcs to relax a frontier of a few thousand vertices is
  close to the worst thing you can ask a GPU to do.
- `gpu-frontier`, the one built for this, is the best GPU solver at unit weights
  and still loses to `delta-omp` (32.4 ms vs 15.3 ms) and to `dial` (9.1 ms).
  At `wmax=100` the best GPU solver, `gpu-nearfar` at 172.9 ms, is 7.2x slower
  than the best CPU one, `dial` at 23.9 ms.
- The frontier never gets wide enough to fill 14 SMs. A 1000x1000 mesh has an
  average frontier of 500 to 750 vertices and a maximum of 1,332, spread over
  1,300 to 2,000 sequential rounds depending on the source; the card has 14 SMs wanting thousands of resident threads
  each. The measurements are consistent with available parallelism rather than
  bandwidth being the binding constraint, and that is as far as they go — no
  profiler run backs it up, so it is stated as consistency, not as a measured
  occupancy figure. `--stats` prints the frontier widths if you want to check
  the premise.

The gap widens with size, as it must. On a 2000x2000 mesh (diameter 3998) at
unit weights, `gpu-topo` takes 3,185 ms and `gpu-edge` 3,365 ms against `dial`'s
44 ms — a 72x loss for the naive kernel. `gpu-frontier` holds up best at 88 ms
and still loses.

`bellman-ford` is left out of this table; see the caveat below.

| n = 4,000,000 mesh | `dial` | `delta-omp` | `gpu-topo` | `gpu-edge` | `gpu-frontier` | `gpu-nearfar` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| weights [1,1] | **44** | 76 | 3,185 | 3,365 | 88 | 209 |
| weights [1,100] | 112 | **210** | 3,422 | 5,518 | 2,674 | 388 |

This is the clean version of the result the whole project was set up to find:
the GPU's advantage is entirely contingent on the graph having a small diameter,
and a mesh is the counterexample. Note also that `delta-omp` stops being the
best CPU solver here — across the nine mesh configurations the win splits evenly
between `bellman-ford`, `dial` and `delta-omp`, because a frontier of a few
thousand vertices does not justify 14 threads.

**One caveat on the Bellman-Ford row.** Its 2 rounds at unit weights is real and
verified, but it is an artifact of the benchmark, not a property of the
algorithm. `source=0` is the grid's top-left corner, vertices are numbered
row-major, so every shortest path runs monotonically increasing in vertex index
and a single forward sweep propagates all of them. Move the source to the middle
of the same graph and it needs 231 rounds:

```console
$ ./bin/bench mesh.bin --only dijkstra,bellman-ford --source 0
bellman-ford  cpu        0.432      0.000      0.432        2     922.4   10.65x  yes
$ ./bin/bench mesh.bin --only dijkstra,bellman-ford --source 50000
bellman-ford  cpu       37.761      0.000     37.761      231      10.5    0.13x  yes
```

An 87x swing from the source vertex alone. Both are correct. Any grid result
here that involves `bellman-ford` at `wmax=1` should be read as measuring the
vertex numbering rather than the algorithm.

### The thread count trap

This one cost a day, and it was measured on a different machine: a shared
2x12-core Xeon Silver 4214 reporting 48 CPUs (24 physical cores, hyperthreaded,
two sockets). Running OpenMP with all 48:

| threads | empty parallel region | `delta-omp` on a 50k graph |
| ---: | ---: | ---: |
| 24 | 11.7 us | 8.95 ms |
| 48 | 6486.0 us | 188.64 ms |

At 48 threads the runtime oversubscribes a loaded machine, every barrier ends up
waiting on a descheduled thread, and an *empty* parallel region costs 6.5
milliseconds. That was swamping the algorithm by 20x and looked exactly like a
bad parallel implementation.

So `delta-omp` defaults to the physical core count, read from
`/sys/devices/system/cpu/*/topology/thread_siblings_list`, and only uses
`OMP_NUM_THREADS` when it is explicitly set. `--threads N` overrides both. The
second hyperthread on a core does nothing for memory-bound graph traversal
anyway.

On the i5-14600 used for the sweep the same rule picks 14 of 20 logical CPUs,
which is the right count for a different reason: 6 P-cores with hyperthreading
plus 8 E-cores. Counting siblings collapses the P-core pairs and leaves each
E-core, so the pool is one thread per physical core.

There is a second guard in the same place: any phase with fewer than a few
thousand vertices in its bucket runs serially. Forking threads to relax a few
hundred edges is never worth it, and on a loaded machine it is a catastrophe.

## Two microbenchmarks

`rmat` mixes degree skew, small diameter and hub vertices together, so a result
on it cannot be attributed to any one of them. `chain` and `star` each isolate
one property, which makes them useless as benchmarks and very good as
explanations.

**`star`, n = 1,000,000** — one hub holding all 2M arcs, everything else degree
1. Pure degree skew, hop eccentricity 2.

| solver | solve ms | rounds |
| --- | ---: | ---: |
| `dijkstra` | 224.6 | 1,000,000 |
| `dial` | 23.5 | 101 |
| `bellman-ford` | 8.2 | 2 |
| `delta` | 30.4 | 3 |
| `delta-omp` | 42.8 | 3 |
| `gpu-topo` | 485.7 | 2 |
| `gpu-edge` | **2.3** | 2 |
| `gpu-frontier` | 49.9 | 2 |
| `gpu-nearfar` | 60.4 | 6 |

`gpu-topo` and `gpu-edge` run the same algorithm over the same edges in the same
number of rounds, and differ by **212x**. Thread-per-vertex puts all 999,999 hub
edges on one thread while the other 999,999 threads finish instantly and wait.
Thread-per-edge spreads them across the whole grid. `gpu-frontier`'s
warp-per-vertex gets a 32-lane team onto the hub, which is 10x better than one
thread and still 20x worse than going per-edge.

This is the `gpu-edge` argument in isolation. On `rmat` the same effect shows up
as a median 1.79x, because the skew is real but nothing like this extreme.

**`chain`, n = 20,000** — a path graph. Pure sequential depth, frontier width 1.

| solver | solve ms | rounds |
| --- | ---: | ---: |
| `dijkstra` | **0.31** | 20,000 |
| `dial` | 4.89 | 20,000 |
| `bellman-ford` | 0.17 | 2 |
| `delta` | 1.22 | 15,191 |
| `delta-omp` | 1.29 | 15,191 |
| `gpu-topo` | 346.8 | 20,000 |
| `gpu-edge` | 385.4 | 19,999 |
| `gpu-frontier` | 374.8 | 20,000 |
| `gpu-nearfar` | 1,047.9 | 36,379 |

Both `bellman-ford` rows are the vertex-numbering artifact again — a chain
numbered in path order settles in one forward sweep. Ignore them and read the
GPU rows: every GPU solver is between 1,100x and 3,300x slower than serial
Dijkstra on a 40,000-arc graph. There is no load-balancing or memory-access story here: a
frontier of one vertex means a kernel launch and a device synchronisation per
edge, and 20,000 of those cost about 350 ms no matter what the kernel does.
`gpu-nearfar` is worst because its near/far split adds rounds without adding
width.

The mesh result is the interesting middle: wide enough that the GPU is not
absurd, narrow enough that it still loses. The chain says what the limit looks
like.

## Testing

```bash
make test
```

`tests/test_sssp.cpp` is one binary, no framework, ~5,500 assertions in about
three seconds. What it covers:

| case | what it checks |
| --- | --- |
| `random_small` | 300 seeded random graphs, n up to 64, five weight regimes, directed and undirected, random source. Every solver against the oracle. |
| `forced_delta` | the same graphs at `--delta` 1, 2, 3, 17, 4999, 5000, 100000, which exercises the bucket ring at and past its wraparound bound |
| `chain` / `star` / `mesh` | the shapes the sweep uses, at a size the oracle can verify |
| `single_vertex`, `disconnected`, `self_and_parallel_edges`, `zero_weights` | the degenerate inputs |
| `contention_*` | 30k and 50k vertex graphs built so that thousands of threads relax the same few vertices at once, with `--delta 1` to force wide buckets |
| `csr`, `dedup`, `io_*` | CSR construction, parallel-edge collapse, text and binary round trips, malformed input |
| `residency` | GPU solvers upload once, reuse, and re-upload after `release()` |
| `stats` | hop eccentricity and frontier widths on shapes with known answers |

The oracle is the point. Cross-checking solvers against each other, which is
what the bench does on large graphs, cannot catch a mistake they all share — a
misunderstanding of the weight contract, say, or a bad CSR build. `random_small`
compares against an O(V*E) Bellman-Ford that shares no code with any solver, and
the `EdgeList` overload of it does not go through `build_csr` either.

The contention cases exist because `delta-omp`'s correctness argument is the
subtle one. It relies on a CAS loop where the thread that lowers `dist[v]` is
the one that enqueues it, so a thread reading a stale higher distance does
redundant work but never loses an update. Small random graphs never test that:
`delta-omp` runs any bucket under 4,096 vertices serially, so a 60-vertex graph
exercises none of the parallel path. The funnel graph puts 7,500 vertices in one
bucket, all relaxing into the same 8 targets.

Verified by mutation: reverting the light/heavy edge boundary, shrinking the
bucket ring by one slot, replacing the CAS with a plain store, dropping the
frontier's per-round dedup, and losing the near-far pile each produce between 1
and 465 failures.

## Reproducing the study

The whole thing, from a clean clone, is three commands and a few hours:

```bash
make
make test                                        # before trusting any of it
tools/sweep.sh                                   # 63 configs, ~14k solver runs
tools/summarize.py results/sweep_<stamp>.csv
```

Peak disk is one graph (about 1 GB at the largest configuration) because the
sweep deletes as it goes. Add `KEEP=1` if you want them retained, and roughly
16 GB free.

The grid rows no longer need the `--no-connect` correction described above; the
generator applies it. To reproduce the broken version for comparison:

```bash
GENFLAGS=--connect GRAPHS=graphs/shortcut CSV=results/grid_shortcut.csv \
  TOPOS=grid tools/sweep.sh
```

The microbenchmarks:

```bash
./bin/gen_graph --n 20000 --topo chain --wmax 100 --out graphs/chain.bin
./bin/gen_graph --n 1000000 --topo star --wmax 100 --out graphs/star.bin
./bin/bench graphs/chain.bin --reps 3
./bin/bench graphs/star.bin --reps 3
```

To reproduce a single row, the tag names every parameter. `rmat/n1000000/deg8/w100`
is:

```bash
./bin/gen_graph --n 1000000 --m 4000000 --topo rmat --wmax 100 --out g.bin
./bin/bench g.bin --reps 3
```

## Troubleshooting

| symptom | cause |
| --- | --- |
| `none (built without CUDA)` | the Makefile probe found no usable nvcc. `make info` shows what it tried. |
| `skipped 4 gpu solvers: no cuda device visible` | CUDA support is compiled in but `cudaGetDeviceCount` found nothing. |
| a solver prints `NO` in the `ok` column | it disagrees with the reference. The mismatch count follows. |
| `N of M vertices unreachable` | expected with `--no-connect`; otherwise the source is in a small component. |
| a kernel launch fails on a large graph | `ARCHS` probably does not match the card, so the binary has no compatible cubin. |
| `delta-omp` slower than `delta` | thread oversubscription. See the thread count trap. |
| `negative edge weight ...` | these solvers need `w >= 0`; the loader refuses rather than returning nonsense. |
| `ref` in the `ok` column | that solver supplied the reference because the graph was too big for the oracle. Nothing checked it. |
| wide `solve ms` spread | something else is on the machine. Raise `--reps` and read the median, not the min. |
| a solver's `rounds` changes with `--source` | expected, and sometimes by orders of magnitude. See source sensitivity. |
