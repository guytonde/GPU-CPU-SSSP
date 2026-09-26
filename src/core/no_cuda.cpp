// Stubs for builds without nvcc as src/gpu/device.cu has the real ones
#ifndef SSSP_CUDA

#include "sssp/solvers.hpp"

namespace sssp {

int gpu_device_count() { return 0; }

std::string gpu_device_name() { return "none (built without CUDA)"; }

void gpu_warmup() {}

}  // namespace sssp

#endif
