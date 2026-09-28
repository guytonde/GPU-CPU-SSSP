#pragma once

#include <cstdint>
#include <vector>

#include "sssp/graph.hpp"

namespace sssp {

// Phase boundaries are described in docs/methodology.md. The caller times
// release() and the whole run().
struct Timing {
    double prep_ms = 0.0;
    double alloc_ms = 0.0;
    double h2d_ms = 0.0;
    double solve_ms = 0.0;
    double d2h_ms = 0.0;
    double kernel_ms = -1;  // only when instrumented
};

// The work behind the time. The planned model was
// T ~ alpha * host_syncs + edges_touched / theta.
struct Counters {
    int64_t iterations = 0;         // the solver's own outer loop count
    int64_t sync_rounds = 0;        // rounds that end in a global wait
    int64_t host_syncs = 0;         // blocking host to device round trips
    int64_t parallel_phases = 0;    // delta-omp only
    int64_t serial_phases = 0;      // delta-omp only
    int64_t vertices_expanded = 0;  // including re-expansions
    int64_t edges_touched = 0;      // adjacency entries read
};

struct Run {
    std::vector<Weight> dist;
    Timing timing;
    Counters counters;
    int delta_used = 0;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual const char* name() const = 0;
    virtual const char* device() const = 0;
    virtual Run run(const Graph& g, int source) = 0;

    virtual void set_delta(int) {}
    // GPU work counters cost an atomic per vertex, so they can be turned off.
    virtual void set_counters(bool) {}
    // An event pair around every kernel, which changes the timing.
    virtual void set_instrument(bool) {}
    // GPU solvers keep the graph on the device between runs until this.
    virtual void release() {}
};

}  // namespace sssp
