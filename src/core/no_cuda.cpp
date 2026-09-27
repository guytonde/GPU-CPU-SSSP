// Stubs for builds without nvcc; src/gpu has the real ones.
#ifndef SSSP_CUDA

#include "sssp/solvers.hpp"

namespace sssp {

int gpu_device_count() { return 0; }
std::string gpu_device_name() { return "none (built without CUDA)"; }
KeyValues gpu_properties() { return {}; }
void gpu_warmup() {}
void gpu_flush_l2() {}
double gpu_preheat() { return 0.0; }
double gpu_round_overhead_us(int, int, bool, int) { return 0.0; }
double gpu_copy_ms(size_t, bool, bool, int) { return 0.0; }
RoundTiming gpu_frontier_round(const Graph&, const std::vector<int>&, int) { return {}; }

}  // namespace sssp

#endif
