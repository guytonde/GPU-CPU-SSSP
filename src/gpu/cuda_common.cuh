#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "sssp/graph.hpp"
#include "sssp/solver.hpp"
#include "sssp/timer.hpp"

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

    void copy_from(const T* host, size_t count) {
        if (count > count_) throw std::runtime_error("upload past end of buffer");
        if (count) {
            CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }
    void copy_to(T* host, size_t count) const {
        if (count > count_) throw std::runtime_error("download past end of buffer");
        if (count) {
            CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
    }
    void fill_zero() {
        if (count_) CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

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

// Sums the time of the kernels it brackets, in instrumented runs only.
class KernelClock {
public:
    ~KernelClock() {
        for (cudaEvent_t e : events_) cudaEventDestroy(e);
    }
    void enable(bool on) { on_ = on; }
    bool enabled() const { return on_; }
    void reset() {
        used_ = 0;
        total_ms_ = 0.0;
    }
    void start() { record(); }
    void stop() { record(); }
    // Only after a blocking sync, when every event has completed.
    void drain() {
        for (size_t i = 0; i + 1 < used_; i += 2) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, events_[i], events_[i + 1]));
            total_ms_ += ms;
        }
        used_ = 0;
    }
    double total_ms() const { return total_ms_; }

private:
    void record() {
        if (!on_) return;
        if (used_ == events_.size()) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreate(&e));
            events_.push_back(e);
        }
        CUDA_CHECK(cudaEventRecord(events_[used_++]));
    }

    bool on_ = false;
    std::vector<cudaEvent_t> events_;
    size_t used_ = 0;
    double total_ms_ = 0.0;
};

struct DeviceCsr {
    DeviceBuffer<int> offsets;
    DeviceBuffer<int> targets;
    DeviceBuffer<Weight> weights;
    int n = 0;
    int64_t m = 0;
};

// Keeps the CSR on the device between runs on the same graph. bind() returns
// true when it uploaded, so the caller's own buffers need reallocating too.
class ResidentCsr {
public:
    bool bind(const Graph& g, Timing& t) {
        if (owner_ == &g && csr_.n == g.n && csr_.m == g.num_edges()) return false;
        release();

        Timer alloc;
        csr_.offsets.alloc(g.offsets.size());
        csr_.targets.alloc(g.targets.size());
        csr_.weights.alloc(g.weights.size());
        t.alloc_ms += alloc.ms();

        GpuTimer copy;
        copy.start();
        csr_.offsets.copy_from(g.offsets.data(), g.offsets.size());
        csr_.targets.copy_from(g.targets.data(), g.targets.size());
        csr_.weights.copy_from(g.weights.data(), g.weights.size());
        t.h2d_ms += copy.stop();

        csr_.n = g.n;
        csr_.m = g.num_edges();
        owner_ = &g;
        return true;
    }

    void release() {
        csr_.offsets.free();
        csr_.targets.free();
        csr_.weights.free();
        csr_.n = 0;
        csr_.m = 0;
        owner_ = nullptr;
    }

    const DeviceCsr& get() const { return csr_; }

private:
    DeviceCsr csr_;
    const Graph* owner_ = nullptr;
};

// Includes allocating the host array, as the CPU solvers do.
template <typename T>
double download_result(const DeviceBuffer<T>& d, std::vector<T>& host, int n) {
    Timer t;
    host.resize(n);
    d.copy_to(host.data(), n);
    return t.ms();
}

constexpr int kBlock = 256;
constexpr int kWarp = 32;

inline int grid_for(int64_t work, int block = kBlock) {
    int64_t blocks = (work + block - 1) / block;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

// Pushes the flagged lanes' values with one atomicAdd per warp. Every lane of
// the warp must reach it.
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

// Adds the warp's sum of x to *counter. Every lane of the warp must reach it.
__device__ inline void warp_count(unsigned long long* counter, unsigned long long x) {
    for (int off = kWarp / 2; off > 0; off /= 2) x += __shfl_down_sync(0xffffffffu, x, off);
    if ((threadIdx.x & (kWarp - 1)) == 0 && x) atomicAdd(counter, x);
}

__device__ inline int round_up(int x, int mult) {
    return (x + mult - 1) / mult * mult;
}

// Shared by gpu-frontier and gpu-nearfar.
static __global__ void seed(Weight* dist, int* queued, int n, int source, int* queue,
                            int* size) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dist[i] = (i == source) ? 0 : kInf;
        queued[i] = (i == source) ? 1 : 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        queue[0] = source;
        *size = 1;
    }
}

// Lets vertices about to expand be queued again if they improve this round.
static __global__ void clear_queued(const int* __restrict__ queue, int size,
                                    int* __restrict__ queued) {
    int stride = blockDim.x * gridDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < size; k += stride) {
        queued[queue[k]] = 0;
    }
}

inline void device_set(int* dev, int value) {
    CUDA_CHECK(cudaMemcpy(dev, &value, sizeof(int), cudaMemcpyHostToDevice));
}

inline int device_get(const int* dev) {
    int value = 0;
    CUDA_CHECK(cudaMemcpy(&value, dev, sizeof(int), cudaMemcpyDeviceToHost));
    return value;
}

inline int64_t read_counter(const DeviceBuffer<unsigned long long>& c, int index = 0) {
    std::vector<unsigned long long> host(c.count());
    c.copy_to(host.data(), host.size());
    return int64_t(host[index]);
}

}  // namespace sssp
