#include "sssp/registry.hpp"

#include "sssp/solvers.hpp"

namespace sssp {

namespace {

std::vector<SolverEntry> build_table() {
    std::vector<SolverEntry> t;

    t.push_back({"dijkstra", "cpu", make_dijkstra_heap});
    t.push_back({"dial", "cpu", make_dijkstra_dial});
    t.push_back({"bellman-ford", "cpu", make_bellman_ford});
    t.push_back({"delta", "cpu", make_delta_stepping});
    t.push_back({"delta-omp", "cpu", make_delta_stepping_omp});

#ifdef SSSP_CUDA
    t.push_back({"gpu-topo", "gpu", make_gpu_bellman_ford_topo});
    t.push_back({"gpu-edge", "gpu", make_gpu_bellman_ford_edge});
    t.push_back({"gpu-frontier", "gpu", make_gpu_frontier});
    t.push_back({"gpu-nearfar", "gpu", make_gpu_near_far});
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
