#include <algorithm>
#include <utility>

#include "cuda_common.cuh"
#include "sssp/delta.hpp"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// Like the frontier expansion, but a vertex whose new distance is past the
// threshold goes to the far queue instead. A vertex is in at most one queue.
__global__ void expand_split(const int* __restrict__ offsets,
                             const int* __restrict__ targets,
                             const Weight* __restrict__ weights,
                             Weight* __restrict__ dist,
                             const int* __restrict__ in_queue, int in_size,
                             Weight threshold, int* __restrict__ queued,
                             int* __restrict__ near_queue,
                             int* __restrict__ near_size,
                             int* __restrict__ far_queue,
                             int* __restrict__ far_size,
                             unsigned long long* __restrict__ work) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
    const int warps = (blockDim.x * gridDim.x) / kWarp;

    for (int k = warp_id; k < in_size; k += warps) {
        int u = in_queue[k];
        Weight du = dist[u];
        int begin = offsets[u];
        int len = offsets[u + 1] - begin;
        if (work && lane == 0) atomicAdd(work, (unsigned long long)len);

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

// After the threshold moves, sends far vertices now under it to the near queue.
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

class GpuNearFar : public Solver {
public:
    const char* name() const override { return "gpu-nearfar"; }
    const char* device() const override { return "gpu"; }
    void set_delta(int d) override { forced_delta_ = d; }
    void set_counters(bool on) override { counters_ = on; }
    void set_instrument(bool on) override { clock_.enable(on); }

    Run run(const Graph& g, int source) override {
        Run r;
        Counters& c = r.counters;

        if (csr_.bind(g, r.timing)) {
            Timer alloc;
            d_dist_.alloc(g.n);
            d_queued_.alloc(g.n);
            near_a_.alloc(g.n);
            near_b_.alloc(g.n);
            far_a_.alloc(g.n);
            far_b_.alloc(g.n);
            counters_buf_.alloc(3);
            d_work_.alloc(1);
            r.timing.alloc_ms += alloc.ms();
        }
        const DeviceCsr& d = csr_.get();
        const Weight delta = forced_delta_ > 0 ? forced_delta_ : pick_delta(g);
        r.delta_used = delta;
        unsigned long long* work = counters_ ? d_work_.get() : nullptr;
        if (work) d_work_.fill_zero();
        clock_.reset();

        int* d_near = counters_buf_.get();
        int* d_far = counters_buf_.get() + 1;
        int* d_far_next = counters_buf_.get() + 2;

        GpuTimer solve;
        solve.start();

        clock_.start();
        seed<<<grid_for(g.n), kBlock>>>(d_dist_.get(), d_queued_.get(), g.n,
                                        source, near_a_.get(), d_near);
        clock_.stop();
        device_set(d_far, 0);
        c.host_syncs++;

        int* near_cur = near_a_.get();
        int* near_new = near_b_.get();
        int* far_cur = far_a_.get();
        int* far_new = far_b_.get();

        int near = 1;
        Weight threshold = delta;
        // Doubles after an empty split, so a gap in distances costs log(gap) splits.
        Weight step = delta;

        while (true) {
            while (near > 0) {
                c.vertices_expanded += near;
                device_set(d_near, 0);
                clock_.start();
                clear_queued<<<grid_for(near), kBlock>>>(near_cur, near,
                                                       d_queued_.get());
                clock_.stop();
                clock_.start();
                expand_split<<<grid_for(int64_t(near) * kWarp), kBlock>>>(
                    d.offsets.get(), d.targets.get(), d.weights.get(),
                    d_dist_.get(), near_cur, near, threshold, d_queued_.get(),
                    near_new, d_near, far_cur, d_far, work);
                clock_.stop();

                near = device_get(d_near);
                c.host_syncs += 2;
                clock_.drain();
                std::swap(near_cur, near_new);
                c.iterations++;
                c.sync_rounds++;
            }

            int far = device_get(d_far);
            c.host_syncs++;
            if (far == 0) break;

            threshold += step;
            device_set(d_near, 0);
            device_set(d_far_next, 0);
            clock_.start();
            split_far<<<grid_for(far), kBlock>>>(d_dist_.get(), far_cur, far,
                                                 threshold, d_queued_.get(),
                                                 near_cur, d_near, far_new,
                                                 d_far_next);
            clock_.stop();

            near = device_get(d_near);
            int remaining = device_get(d_far_next);
            device_set(d_far, remaining);
            c.host_syncs += 5;
            clock_.drain();
            std::swap(far_cur, far_new);
            step = (near == 0) ? step * 2 : delta;
            c.iterations++;
            c.sync_rounds++;
        }

        r.timing.solve_ms = solve.stop();
        if (clock_.enabled()) r.timing.kernel_ms = clock_.total_ms();

        r.timing.d2h_ms = download_result(d_dist_, r.dist, g.n);
        if (work) c.edges_touched = read_counter(d_work_);
        return r;
    }

    void release() override {
        csr_.release();
        d_dist_.free();
        d_queued_.free();
        near_a_.free();
        near_b_.free();
        far_a_.free();
        far_b_.free();
        counters_buf_.free();
        d_work_.free();
    }

private:
    int forced_delta_ = 0;
    bool counters_ = true;
    KernelClock clock_;
    ResidentCsr csr_;
    DeviceBuffer<Weight> d_dist_;
    DeviceBuffer<int> d_queued_, near_a_, near_b_, far_a_, far_b_, counters_buf_;
    DeviceBuffer<unsigned long long> d_work_;
};

}  // namespace

std::unique_ptr<Solver> make_gpu_near_far() {
    return std::make_unique<GpuNearFar>();
}

}  // namespace sssp
