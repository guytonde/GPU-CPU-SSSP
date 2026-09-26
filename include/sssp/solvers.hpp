#pragma once

#include <memory>
#include <string>

#include "sssp/solver.hpp"

namespace sssp {

std::unique_ptr<Solver> make_dijkstra_heap();
std::unique_ptr<Solver> make_dijkstra_dial();
std::unique_ptr<Solver> make_bellman_ford();
std::unique_ptr<Solver> make_delta_stepping();
std::unique_ptr<Solver> make_delta_stepping_omp();

#ifdef SSSP_CUDA
std::unique_ptr<Solver> make_gpu_bellman_ford_topo();
std::unique_ptr<Solver> make_gpu_bellman_ford_edge();
std::unique_ptr<Solver> make_gpu_frontier();
std::unique_ptr<Solver> make_gpu_near_far();
#endif

// Returns 0 without CUDA support or a usable device. Never throws.
int gpu_device_count();
std::string gpu_device_name();

// Threads for the OpenMP solvers. 0 picks OMP_NUM_THREADS if set, else the
// physical core count.
void set_cpu_threads(int n);
int cpu_threads();

// warm up gpu by creating the context up front so the first timed run doesnt pay
void gpu_warmup();

}  // namespace sssp
