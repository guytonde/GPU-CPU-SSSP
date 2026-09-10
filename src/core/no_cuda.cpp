// Stubs used when the project is built without nvcc. src/gpu/device.cu
// provides the real implementations otherwise.
#ifndef SSSP_CUDA

#include "sssp/solvers.hpp"

namespace sssp {

int gpu_device_count() { return 0; }

std::string gpu_device_name() { return "none (built without CUDA)"; }

void gpu_warmup() {}

}  // namespace sssp

#endif
