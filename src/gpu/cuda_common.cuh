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

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

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
        if (count > count_) throw std::runtime_error("upload past end of buffer");
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& host) {
        alloc(host.size());
        upload(host.data(), host.size());
    }
    void download(T* host, size_t count) const {
        if (count > count_) throw std::runtime_error("download past end of buffer");
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

// Device-side wall time, so host launch overhead stays out of the number.
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

// Uploads the CSR arrays and returns how long it took.
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

// Holds the CSR on the device between runs on the same graph, so repeated
// queries pay the upload once. release() puts the next one back on the clock.
class ResidentCsr {
public:
    // Adds the upload cost to h2d_ms. True if this call did the upload.
    bool bind(const Graph& g, double& h2d_ms) {
        if (owner_ == &g && csr_.n == g.n && csr_.m == g.num_edges()) {
            return false;
        }
        release();
        h2d_ms += upload_csr(g, csr_);
        owner_ = &g;
        return true;
    }

    void release() {
        csr_ = DeviceCsr{};
        owner_ = nullptr;
    }

    const DeviceCsr& get() const { return csr_; }

private:
    DeviceCsr csr_;
    const Graph* owner_ = nullptr;
};

constexpr int kBlock = 256;
constexpr int kWarp = 32;

inline int grid_for(int64_t work, int block = kBlock) {
    int64_t blocks = (work + block - 1) / block;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

// Warp-aggregated queue push: the lanes with something to enqueue share one
// atomicAdd, turning up to 32 contended atomics into one. Every lane must reach
// this call, so the expansion loops pad out to whole warp steps.
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
