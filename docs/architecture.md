# Architecture

```
include/sssp/        headers
  graph.hpp          CSR graph, graph files with their parameters, content hashes
  generators.hpp     graph generators
  stats.hpp          graph and per source statistics
  reference.hpp      Bellman Ford and Dijkstra references
  solver.hpp         solver interface, timing phases, work counters
  solvers.hpp        solver factories, GPU helpers, calibration entry points
  registry.hpp       solver names
  delta.hpp          default bucket width for the delta stepping solvers
  env.hpp            machine details and NVML readings
  csv.hpp            CSV writer
  rng.hpp            sampling that does not depend on the standard library
  timer.hpp          wall clock timer
  build_info.hpp     commit and uncommitted change hash compiled into the binaries
src/core/            everything except the solvers
src/cpu/             dijkstra, dial, bellman-ford, delta, delta-omp
src/gpu/             gpu-topo, gpu-edge, gpu-frontier, gpu-nearfar, CUDA helpers,
                     calibration kernels
src/apps/            gen_graph, bench, calibrate
tests/test_sssp.cpp  tests
tools/               experiments.py, analyze.py, plot.py, profile.sh
```

Every solver implements the same interface.

```cpp
class Solver {
    virtual const char* name() const = 0;
    virtual const char* device() const = 0;             // "cpu" or "gpu"
    virtual Run run(const Graph& g, int source) = 0;    // distances, timing, counters
    virtual void set_delta(int);        // bucket width for the delta stepping solvers
    virtual void set_counters(bool);    // GPU work counters on or off
    virtual void set_instrument(bool);  // time every kernel
    virtual void release();             // free device memory
};
```

The GPU solvers keep the graph and their buffers on the device between runs on
the same graph, so only the first run pays for allocation and upload. The
benchmark gets one shot behaviour by calling `release()` after every run, and
resident behaviour by calling it only at the end of a session.

Most comparisons are between two solvers that differ in one respect.

```
gpu-topo and gpu-edge           one thread per vertex, or one thread per edge
gpu-topo and gpu-frontier       every vertex every round, or only the ones that changed
gpu-frontier and gpu-nearfar    an unordered worklist, or one split into near and far
delta and delta-omp             serial, or parallel with OpenMP
bellman-ford and gpu-topo       the same sweep over every vertex, on the CPU and on the GPU
```

`gen_graph` writes a graph file and `bench` reads it and writes CSV files to
`results/raw`. `calibrate` writes its measurements there too. `experiments.py`
runs both for every experiment, and `profile.sh` collects the hardware counters.
`analyze.py` checks `results/raw` and writes `results/published`, and `plot.py`
draws the plots from `results/published`.

A new solver needs one source file and one line in `src/core/registry.cpp`, and
the tests then check it against the reference automatically. A new graph type
is a function in `src/core/generators.cpp`, and it should be added to the list
in `test_generators` so the tests check that changing its weights leaves its
edges alone. A new experiment is an entry in `plan()` in `tools/experiments.py`.
