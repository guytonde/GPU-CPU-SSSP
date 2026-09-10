#pragma once

#include <cstdint>
#include <vector>

#include "sssp/graph.hpp"

namespace sssp {

struct Timing {
    // Time spent in the actual shortest-path work.
    double solve_ms = 0.0;
    // Host/device copies of the graph and the distance array. Zero on CPU.
    double transfer_ms = 0.0;

    double total_ms() const { return solve_ms + transfer_ms; }
};

struct Run {
    std::vector<Weight> dist;
    Timing timing;
    // Outer iterations: Bellman-Ford rounds, delta-stepping buckets, BFS levels.
    int64_t rounds = 0;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual const char* name() const = 0;
    virtual const char* device() const = 0;
    virtual Run run(const Graph& g, int source) = 0;

    // Delta-stepping and near-far pick a bucket width from graph structure
    // unless the caller pins one. Ignored by the other solvers.
    virtual void set_delta(int) {}
};

}  // namespace sssp
