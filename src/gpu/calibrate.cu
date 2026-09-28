#include <algorithm>
#include <cstdlib>
#include <vector>

#include "cuda_common.cuh"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

__global__ void noop(int* p) {
    if (p && threadIdx.x == 0 && blockIdx.x == 0 && *p == -12345) *p = 0;
}

}  // namespace

double gpu_round_overhead_us(int launches, int syncs, bool memset, int iterations) {
    DeviceBuffer<int> word;
    word.alloc(1);
    word.fill_zero();
    int host = 0;
    auto round = [&]() {
        if (memset) word.fill_zero();
        for (int l = 0; l < launches; ++l) noop<<<1, kWarp>>>(word.get());
        for (int s = 0; s < syncs; ++s) word.copy_to(&host, 1);
    };
    for (int i = 0; i < 200; ++i) round();
    CUDA_CHECK(cudaDeviceSynchronize());

    Timer t;
    for (int i = 0; i < iterations; ++i) round();
    CUDA_CHECK(cudaDeviceSynchronize());
    return t.ms() * 1e3 / iterations;
}

double gpu_copy_ms(size_t bytes, bool pinned, bool to_device, int reps) {
    DeviceBuffer<char> dev;
    dev.alloc(bytes);
    char* host = nullptr;
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&host, bytes));
    } else {
        host = static_cast<char*>(std::malloc(bytes));
        if (!host) throw std::runtime_error("host allocation failed");
    }
    std::fill(host, host + bytes, char(1));

    std::vector<double> ms;
    GpuTimer t;
    for (int rep = 0; rep < reps + 1; ++rep) {
        gpu_preheat();
        t.start();
        if (to_device) {
            CUDA_CHECK(cudaMemcpy(dev.get(), host, bytes, cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(host, dev.get(), bytes, cudaMemcpyDeviceToHost));
        }
        float x = t.stop();
        if (rep > 0) ms.push_back(x);
    }
    if (pinned) {
        cudaFreeHost(host);
    } else {
        std::free(host);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

}  // namespace sssp
