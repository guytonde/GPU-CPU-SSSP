#include "sssp/registry.hpp"

#include "sssp/solvers.hpp"

namespace sssp {

namespace {

std::vector<SolverEntry> build_table() {
    std::vector<SolverEntry> t;

    t.push_back({"dijkstra", "cpu", "binary heap with lazy deletion",
                 make_dijkstra_heap});
    t.push_back({"dial", "cpu", "bucket queue, O(m + n*maxw)",
                 make_dijkstra_dial});
    t.push_back({"bellman-ford", "cpu", "relaxation rounds with early exit",
                 make_bellman_ford});
    t.push_back({"delta", "cpu", "delta-stepping, serial", make_delta_stepping});
    t.push_back({"delta-omp", "cpu", "delta-stepping, OpenMP",
                 make_delta_stepping_omp});

#ifdef SSSP_CUDA
    t.push_back({"gpu-topo", "gpu", "thread per vertex, every vertex per round",
                 make_gpu_bellman_ford_topo});
    t.push_back({"gpu-edge", "gpu", "thread per edge, atomicMin",
                 make_gpu_bellman_ford_edge});
    t.push_back({"gpu-frontier", "gpu", "worklist, warp per vertex",
                 make_gpu_frontier});
    t.push_back({"gpu-nearfar", "gpu", "near-far pile, GPU delta-stepping",
                 make_gpu_near_far});
#endif

    return t;
}

}  // namespace

const std::vector<SolverEntry>& solver_table() {
    static const std::vector<SolverEntry> table = build_table();
    return table;
}

const SolverEntry* find_solver(const std::string& name) {
    for (const auto& e : solver_table()) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

}  // namespace sssp
