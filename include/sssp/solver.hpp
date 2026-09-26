#pragma once

#include <cstdint>
#include <vector>

#include "sssp/graph.hpp"

namespace sssp {

// Phases are kept apart so a GPU solver's kernel time is never confused with
// what it spent building or moving its inputs.
struct Timing {
    // Host side structures a solver derives for itself, e.g. gpu-edge's
    // per-arc source array. Zero for solvers that take CSR as it comes.
    double prep_ms = 0.0;
    double h2d_ms = 0.0;
    double solve_ms = 0.0;
    double d2h_ms = 0.0;

    double transfer_ms() const { return h2d_ms + d2h_ms; }
    double total_ms() const { return prep_ms + h2d_ms + solve_ms + d2h_ms; }
};

struct Run {
    std::vector<Weight> dist;
    Timing timing;
    // Outer iterations
    int64_t rounds = 0;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual const char* name() const = 0;
    virtual const char* device() const = 0;
    virtual Run run(const Graph& g, int source) = 0;

    // pins the bucket width for delta stepping and near-far
    virtual void set_delta(int) {}

    // Solvers holding device state keep it across runs on the same graph and
    // report prep_ms/h2d_ms of zero from the second run on. release() drops it,
    // so a caller can measure either the cold cost or the amortised one.
    virtual void release() {}
};

}  // namespace sssp
