#include <algorithm>
#include <utility>

#include "cuda_common.cuh"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// One warp per frontier vertex. `queued` keeps a vertex from entering the next
// queue twice. `work`, when set, counts adjacency entries read.
__global__ void expand(const int* __restrict__ offsets,
                       const int* __restrict__ targets,
                       const Weight* __restrict__ weights,
                       Weight* __restrict__ dist,
                       const int* __restrict__ in_queue, int in_size,
                       int* __restrict__ out_queue, int* __restrict__ out_size,
                       int* __restrict__ queued,
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
            bool push = false;
            if (off < len) {
                int i = begin + off;
                v = targets[i];
                Weight nd = du + weights[i];
                push = nd < atomicMin(&dist[v], nd) &&
                       atomicExch(&queued[v], 1) == 0;
            }
            warp_push(out_queue, out_size, push, v);
        }
    }
}

// Bellman-Ford over a worklist of the vertices that changed in the last round.
// With weights, a vertex can be expanded again when a shorter path arrives.
class GpuFrontier : public Solver {
public:
    const char* name() const override { return "gpu-frontier"; }
    const char* device() const override { return "gpu"; }
    void set_counters(bool on) override { counters_ = on; }
    void set_instrument(bool on) override { clock_.enable(on); }

    Run run(const Graph& g, int source) override {
        Run r;
        Counters& c = r.counters;

        if (csr_.bind(g, r.timing)) {
            Timer alloc;
            d_dist_.alloc(g.n);
            d_queued_.alloc(g.n);
            qa_.alloc(g.n);
            qb_.alloc(g.n);
            d_size_.alloc(1);
            d_work_.alloc(1);
            r.timing.alloc_ms += alloc.ms();
        }
        const DeviceCsr& d = csr_.get();
        unsigned long long* work = counters_ ? d_work_.get() : nullptr;
        if (work) d_work_.fill_zero();
        clock_.reset();

        GpuTimer solve;
        solve.start();

        clock_.start();
        seed<<<grid_for(g.n), kBlock>>>(d_dist_.get(), d_queued_.get(), g.n,
                                        source, qa_.get(), d_size_.get());
        clock_.stop();

        int* cur = qa_.get();
        int* next = qb_.get();
        int frontier = 1;

        while (frontier > 0 && c.iterations < g.n) {
            c.vertices_expanded += frontier;
            d_size_.fill_zero();
            clock_.start();
            clear_queued<<<grid_for(frontier), kBlock>>>(cur, frontier,
                                                         d_queued_.get());
            clock_.stop();
            clock_.start();
            expand<<<grid_for(int64_t(frontier) * kWarp), kBlock>>>(
                d.offsets.get(), d.targets.get(), d.weights.get(), d_dist_.get(),
                cur, frontier, next, d_size_.get(), d_queued_.get(), work);
            clock_.stop();

            d_size_.copy_to(&frontier, 1);
            c.host_syncs++;
            clock_.drain();
            std::swap(cur, next);
            c.iterations++;
        }
        c.sync_rounds = c.iterations;

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
        qa_.free();
        qb_.free();
        d_size_.free();
        d_work_.free();
    }

private:
    bool counters_ = true;
    KernelClock clock_;
    ResidentCsr csr_;
    DeviceBuffer<Weight> d_dist_;
    DeviceBuffer<int> d_queued_, qa_, qb_, d_size_;
    DeviceBuffer<unsigned long long> d_work_;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

std::unique_ptr<Solver> make_gpu_frontier() {
    return std::make_unique<GpuFrontier>();
}

// One gpu-frontier round from a queue of vertices at distance 0, timed on the
// host around the same calls the solver makes, and separately with events
// around the kernel.
RoundTiming gpu_frontier_round(const Graph& g, const std::vector<int>& queue, int reps) {
    ResidentCsr csr;
    Timing unused;
    csr.bind(g, unused);
    const DeviceCsr& d = csr.get();

    std::vector<Weight> host_dist(g.n, kInf);
    RoundTiming out;
    for (int v : queue) {
        host_dist[v] = 0;
        out.edges += g.degree(v);
    }
    const int size = int(queue.size());

    DeviceBuffer<Weight> dist;
    DeviceBuffer<int> queued, in, outq, count;
    dist.alloc(g.n);
    queued.alloc(g.n);
    in.alloc(std::max(1, size));
    outq.alloc(g.n);
    count.alloc(1);
    in.copy_from(queue.data(), size);

    auto reset = [&]() {
        dist.copy_from(host_dist.data(), g.n);
        queued.fill_zero();
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto launch = [&]() {
        expand<<<grid_for(int64_t(size) * kWarp), kBlock>>>(
            d.offsets.get(), d.targets.get(), d.weights.get(), dist.get(), in.get(),
            size, outq.get(), count.get(), queued.get(), nullptr);
    };

    // Two passes, so the event sync never falls inside a timed round. Rep 0 is a warm up.
    std::vector<double> kernel, round;
    for (int rep = 0; rep < reps + 1; ++rep) {
        reset();
        Timer wall;
        count.fill_zero();
        clear_queued<<<grid_for(size), kBlock>>>(in.get(), size, queued.get());
        launch();
        int pushed = 0;
        count.copy_to(&pushed, 1);
        if (rep > 0) round.push_back(wall.ms() * 1e3);
    }
    GpuTimer events;
    for (int rep = 0; rep < reps + 1; ++rep) {
        reset();
        count.fill_zero();
        clear_queued<<<grid_for(size), kBlock>>>(in.get(), size, queued.get());
        events.start();
        launch();
        float k = events.stop();
        if (rep > 0) kernel.push_back(k * 1e3);
    }
    out.kernel_us = median(kernel);
    out.round_us = median(round);
    return out;
}

}  // namespace sssp
