#include "sssp/registry.hpp"

#include "sssp/solvers.hpp"

namespace sssp {

const std::vector<SolverEntry>& solver_table() {
    static const std::vector<SolverEntry> table = {
        {"dijkstra", "cpu", make_dijkstra_heap},
        {"dial", "cpu", make_dijkstra_dial},
        {"bellman-ford", "cpu", make_bellman_ford},
        {"delta", "cpu", make_delta_stepping},
        {"delta-omp", "cpu", make_delta_stepping_omp},
#ifdef SSSP_CUDA
        {"gpu-topo", "gpu", make_gpu_bellman_ford_topo},
        {"gpu-edge", "gpu", make_gpu_bellman_ford_edge},
        {"gpu-frontier", "gpu", make_gpu_frontier},
        {"gpu-nearfar", "gpu", make_gpu_near_far},
#endif
    };
    return table;
}

const SolverEntry* find_solver(const std::string& name) {
    for (const auto& e : solver_table()) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

}  // namespace sssp
