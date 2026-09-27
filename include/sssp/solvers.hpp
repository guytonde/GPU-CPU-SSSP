#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

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

using KeyValues = std::vector<std::pair<std::string, std::string>>;

// 0 without CUDA or a usable device.
int gpu_device_count();
std::string gpu_device_name();
KeyValues gpu_properties();

// Creates the context and brings the clock up.
void gpu_warmup();
void gpu_flush_l2();

// Spins the GPU until a kernel of known length shows it near full clock.
// Returns the measured SM clock in MHz.
double gpu_preheat();

// 0 means OMP_NUM_THREADS if set, else one per physical core in the affinity mask.
void set_cpu_threads(int n);
int cpu_threads();

// C0 microbenchmarks.
// Mean time of a round of `launches` empty kernels, an optional memset and
// `syncs` blocking 4 byte copies.
double gpu_round_overhead_us(int launches, int syncs, bool memset, int iterations);
double gpu_copy_ms(size_t bytes, bool pinned, bool to_device, int reps);
// Kernel time and whole round time of one gpu-frontier round on `queue`.
struct RoundTiming {
    double kernel_us = 0.0;
    double round_us = 0.0;
    int64_t edges = 0;
};
RoundTiming gpu_frontier_round(const Graph& g, const std::vector<int>& queue, int reps);

}  // namespace sssp
