#include <cuda_runtime.h>

#include <string>

#include "cuda_common.cuh"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// Spins for about `cycles` SM clocks.
__global__ void spin(long long cycles, int* sink) {
    long long start = clock64();
    int x = threadIdx.x;
    while (clock64() - start < cycles) x = x * 1103515245 + 12345;
    if (x == 42 && sink) *sink = x;
}

int attr(cudaDeviceAttr a) {
    int v = 0;
    cudaDeviceGetAttribute(&v, a, 0);
    return v;
}

}  // namespace

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

KeyValues gpu_properties() {
    KeyValues kv;
    if (gpu_device_count() == 0) return kv;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    kv.emplace_back("gpu_name", prop.name);
    kv.emplace_back("gpu_cc", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    kv.emplace_back("gpu_sms", std::to_string(attr(cudaDevAttrMultiProcessorCount)));
    kv.emplace_back("gpu_max_threads_per_sm",
                    std::to_string(attr(cudaDevAttrMaxThreadsPerMultiProcessor)));
    kv.emplace_back("gpu_l2_bytes", std::to_string(attr(cudaDevAttrL2CacheSize)));
    kv.emplace_back("gpu_mem_bytes", std::to_string(prop.totalGlobalMem));
    kv.emplace_back("gpu_mem_bus_bits", std::to_string(attr(cudaDevAttrGlobalMemoryBusWidth)));
    kv.emplace_back("cuda_runtime", std::to_string(runtime));
    kv.emplace_back("cuda_driver", std::to_string(driver));
    return kv;
}

void gpu_warmup() {
    if (gpu_device_count() == 0) return;
    CUDA_CHECK(cudaFree(nullptr));
    gpu_preheat();
}

// Copies 16 MB to wake the PCIe link, then spins in chunks of about 1 ms until
// cycles over elapsed time reach 90% of the rated clock, or 100 chunks pass.
double gpu_preheat() {
    static DeviceBuffer<char> link;
    static std::vector<char> host(16 << 20, 1);
    if (link.count() == 0) link.alloc(host.size());
    link.copy_from(host.data(), host.size());

    const int sms = attr(cudaDevAttrMultiProcessorCount);
    const double rated_mhz = attr(cudaDevAttrClockRate) / 1000.0;
    const long long chunk = (long long)(rated_mhz * 1000.0);
    GpuTimer t;
    double mhz = 0.0;
    for (int i = 0; i < 100; ++i) {
        t.start();
        spin<<<sms * 4, kBlock>>>(chunk, nullptr);
        float ms = t.stop();
        mhz = ms > 0 ? double(chunk) / (ms * 1000.0) : 0.0;
        if (mhz >= 0.9 * rated_mhz && i >= 2) break;
    }
    return mhz;
}

void gpu_flush_l2() {
    static DeviceBuffer<char> junk;
    static unsigned char fill = 0;
    if (junk.count() == 0) junk.alloc(size_t(256) << 20);
    CUDA_CHECK(cudaMemset(junk.get(), ++fill, junk.count()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace sssp
