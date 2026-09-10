#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "sssp/graph.hpp"

namespace sssp {

inline void cuda_check(cudaError_t e, const char* expr, const char* file,
                       int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" +
                                 std::to_string(line) + " " + expr + ": " +
                                 cudaGetErrorString(e));
    }
}

#define CUDA_CHECK(call) ::sssp::cuda_check((call), #call, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { alloc(count); }
    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void alloc(size_t count) {
        free();
        if (count == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        count_ = count;
    }

    void free() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& host) {
        alloc(host.size());
        upload(host.data(), host.size());
    }
    void download(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
    void fill_zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Wall time of a stretch of device work, measured with events so that host-side
// launch overhead is not folded into the number.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

struct DeviceCsr {
    DeviceBuffer<int> offsets;
    DeviceBuffer<int> targets;
    DeviceBuffer<Weight> weights;
    int n = 0;
    int64_t m = 0;
};

// Uploads the CSR arrays and reports how long that took, so the bench can keep
// transfer cost separate from solve cost.
inline double upload_csr(const Graph& g, DeviceCsr& d) {
    GpuTimer t;
    t.start();
    d.n = g.n;
    d.m = g.num_edges();
    d.offsets.upload(g.offsets);
    d.targets.upload(g.targets);
    d.weights.upload(g.weights);
    return t.stop();
}

constexpr int kBlock = 256;
constexpr int kWarp = 32;

inline int grid_for(int64_t work, int block = kBlock) {
    int64_t blocks = (work + block - 1) / block;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

// Warp-aggregated queue push. The lanes with something to enqueue take a single
// atomicAdd between them and then write to their own slot, turning up to 32
// contended atomics on the queue counter into one. Every lane of the warp must
// reach this call, which is why the expansion loops that use it are padded out
// to a whole number of warp steps.
__device__ inline void warp_push(int* queue, int* size, bool push, int v) {
    unsigned mask = __ballot_sync(0xffffffffu, push);
    if (mask == 0) return;

    int lane = threadIdx.x & (kWarp - 1);
    int leader = __ffs(mask) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(size, __popc(mask));
    base = __shfl_sync(0xffffffffu, base, leader);

    if (push) queue[base + __popc(mask & ((1u << lane) - 1))] = v;
}

__device__ inline int round_up(int x, int mult) {
    return (x + mult - 1) / mult * mult;
}

inline void device_set(int* dev, int value) {
    CUDA_CHECK(cudaMemcpy(dev, &value, sizeof(int), cudaMemcpyHostToDevice));
}

inline int device_get(const int* dev) {
    int value = 0;
    CUDA_CHECK(cudaMemcpy(&value, dev, sizeof(int), cudaMemcpyDeviceToHost));
    return value;
}

}  // namespace sssp
