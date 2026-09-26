#include <cuda_runtime.h>

#include "sssp/solvers.hpp"

namespace sssp {

int gpu_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count;
}

std::string gpu_device_name() {
    if (gpu_device_count() == 0) return "none";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "unknown";
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
           std::to_string(prop.minor) + ", " +
           std::to_string(prop.multiProcessorCount) + " SMs)";
}

void gpu_warmup() {
    if (gpu_device_count() == 0) return;
    void* p = nullptr;
    if (cudaMalloc(&p, 16) == cudaSuccess) cudaFree(p);
    cudaGetLastError();
}

}  // namespace sssp
