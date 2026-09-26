#include <algorithm>
#include <utility>

#include "../cpu/delta.hpp"
#include "cuda_common.cuh"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// Near-far split: the frontier expansion, except a vertex landing past the
// threshold is parked in the far pile instead of expanded now. Delta-stepping's
// bucket rule with two buckets.
//
// One `queued` flag per vertex means a vertex sits in at most one queue, so both
// queues are bounded by n. A vertex that improves while parked is not moved
// eagerly; the next split re-reads its distance and routes it.
__global__ void expand_split(const int* __restrict__ offsets,
                             const int* __restrict__ targets,
                             const Weight* __restrict__ weights,
                             Weight* __restrict__ dist,
                             const int* __restrict__ in_queue, int in_size,
                             Weight threshold, int* __restrict__ queued,
                             int* __restrict__ near_queue,
                             int* __restrict__ near_size,
                             int* __restrict__ far_queue,
                             int* __restrict__ far_size) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
    const int warps = (blockDim.x * gridDim.x) / kWarp;

    for (int k = warp_id; k < in_size; k += warps) {
        int u = in_queue[k];
        Weight du = dist[u];
        int begin = offsets[u];
        int len = offsets[u + 1] - begin;

        for (int off = lane; off < round_up(len, kWarp); off += kWarp) {
            int v = -1;
            Weight nd = 0;
            bool claimed = false;
            if (off < len) {
                int i = begin + off;
                v = targets[i];
                nd = du + weights[i];
                claimed = nd < atomicMin(&dist[v], nd) &&
                          atomicExch(&queued[v], 1) == 0;
            }
            warp_push(near_queue, near_size, claimed && nd < threshold, v);
            warp_push(far_queue, far_size, claimed && nd >= threshold, v);
        }
    }
}

// Rebuilds the near set from the far pile after the threshold moves. Entries now
// under the threshold go near and give up their flag, the rest stay far.
__global__ void split_far(const Weight* __restrict__ dist,
                          const int* __restrict__ in_queue, int in_size,
                          Weight threshold, int* __restrict__ queued,
                          int* __restrict__ near_queue,
                          int* __restrict__ near_size,
                          int* __restrict__ far_queue,
                          int* __restrict__ far_size) {
    const int stride = blockDim.x * gridDim.x;
    const int start = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = start; k < round_up(in_size, stride); k += stride) {
        int v = -1;
        Weight d = kInf;
        if (k < in_size) {
            v = in_queue[k];
            d = dist[v];
        }
        bool near = k < in_size && d < threshold;
        if (near) queued[v] = 0;
        warp_push(near_queue, near_size, near, v);
        warp_push(far_queue, far_size, k < in_size && !near, v);
    }
}

// Clears the flags of the vertices about to expand, so an improvement found this
// round can re-queue them. O(|frontier|), not O(n).
__global__ void clear_flags(const int* __restrict__ queue, int size,
                            int* __restrict__ queued) {
    int stride = blockDim.x * gridDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < size; k += stride) {
        queued[queue[k]] = 0;
    }
}

__global__ void seed(Weight* dist, int* queued, int n, int source, int* queue,
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

class GpuNearFar : public Solver {
public:
    const char* name() const override { return "gpu-nearfar"; }
    const char* device() const override { return "gpu"; }
    void set_delta(int d) override { forced_delta_ = d; }

    Run run(const Graph& g, int source) override {
        Run r;
        r.dist.assign(g.n, kInf);

        csr_.bind(g, r.timing.h2d_ms);
        const DeviceCsr& d = csr_.get();

        const Weight delta = forced_delta_ > 0 ? forced_delta_ : pick_delta(g);

        DeviceBuffer<Weight> d_dist(g.n);
        DeviceBuffer<int> d_queued(g.n);
        DeviceBuffer<int> near_a(g.n), near_b(g.n), far_a(g.n), far_b(g.n);
        DeviceBuffer<int> counters(3);

        int* d_near = counters.get();
        int* d_far = counters.get() + 1;
        int* d_far_next = counters.get() + 2;

        GpuTimer solve;
        solve.start();

        seed<<<grid_for(g.n), kBlock>>>(d_dist.get(), d_queued.get(), g.n,
                                        source, near_a.get(), d_near);
        device_set(d_far, 0);

        int* near_cur = near_a.get();
        int* near_new = near_b.get();
        int* far_cur = far_a.get();
        int* far_new = far_b.get();

        int near = 1;
        Weight threshold = delta;
        // Doubles on an empty split, so a wide gap in the distance
        // distribution costs log(gap) splits instead of gap/delta.
        Weight step = delta;

        while (true) {
            while (near > 0) {
                device_set(d_near, 0);
                clear_flags<<<grid_for(near), kBlock>>>(near_cur, near,
                                                       d_queued.get());
                expand_split<<<grid_for(int64_t(near) * kWarp), kBlock>>>(
                    d.offsets.get(), d.targets.get(), d.weights.get(),
                    d_dist.get(), near_cur, near, threshold, d_queued.get(),
                    near_new, d_near, far_cur, d_far);

                near = device_get(d_near);
                std::swap(near_cur, near_new);
                r.rounds++;
            }

            int far = device_get(d_far);
            if (far == 0) break;

            threshold += step;
            device_set(d_near, 0);
            device_set(d_far_next, 0);
            split_far<<<grid_for(far), kBlock>>>(d_dist.get(), far_cur, far,
                                                 threshold, d_queued.get(),
                                                 near_cur, d_near, far_new,
                                                 d_far_next);

            near = device_get(d_near);
            int remaining = device_get(d_far_next);
            device_set(d_far, remaining);
            std::swap(far_cur, far_new);
            step = (near == 0) ? step * 2 : delta;
            r.rounds++;
        }

        r.timing.solve_ms = solve.stop();

        GpuTimer back;
        back.start();
        d_dist.download(r.dist.data(), g.n);
        r.timing.d2h_ms = back.stop();
        return r;
    }

    void release() override { csr_.release(); }

private:
    int forced_delta_ = 0;
    ResidentCsr csr_;
};

}  // namespace

std::unique_ptr<Solver> make_gpu_near_far() {
    return std::make_unique<GpuNearFar>();
}

}  // namespace sssp
