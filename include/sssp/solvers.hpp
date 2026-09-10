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

// Returns 0 when the binary has no CUDA support or the machine has no usable
// device. Never throws, so the bench can degrade to CPU-only.
int gpu_device_count();
std::string gpu_device_name();

// Threads used by the OpenMP solvers. Pass 0 to let it pick: the value of
// OMP_NUM_THREADS if set, otherwise the physical core count.
void set_cpu_threads(int n);
int cpu_threads();

// Forces context creation so the first timed run does not pay for it.
void gpu_warmup();

}  // namespace sssp
