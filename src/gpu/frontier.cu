#include <utility>

#include "cuda_common.cuh"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// One warp per frontier vertex, lanes striding its adjacency, so work inside a
// warp is uniform even when the degree distribution is not. `queued` keeps a
// vertex out of the output queue twice in one round.
__global__ void expand(const int* __restrict__ offsets,
                       const int* __restrict__ targets,
                       const Weight* __restrict__ weights,
                       Weight* __restrict__ dist,
                       const int* __restrict__ in_queue, int in_size,
                       int* __restrict__ out_queue, int* __restrict__ out_size,
                       int* __restrict__ queued) {
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

// Clears the dedup flags of the vertices about to expand, so one that improves
// again this round can re-queue. O(|frontier|), not O(n).
__global__ void clear_queued(const int* __restrict__ queue, int size,
                             int* __restrict__ queued) {
    int stride = blockDim.x * gridDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < size; k += stride) {
        queued[queue[k]] = 0;
    }
}

__global__ void seed(Weight* dist, int* queued, int n, int source,
                     int* queue, int* size) {
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

// Data-driven Bellman-Ford: a round only touches vertices whose distance changed
// in the previous one. At small diameter that is close to O(m) total.
class GpuFrontier : public Solver {
public:
    const char* name() const override { return "gpu-frontier"; }
    const char* device() const override { return "gpu"; }

    Run run(const Graph& g, int source) override {
        Run r;
        r.dist.assign(g.n, kInf);

        csr_.bind(g, r.timing.h2d_ms);
        const DeviceCsr& d = csr_.get();

        DeviceBuffer<Weight> d_dist(g.n);
        DeviceBuffer<int> d_queued(g.n);
        // `queued` caps a vertex at one push per round, so n slots is a hard
        // bound.
        DeviceBuffer<int> qa(g.n), qb(g.n);
        DeviceBuffer<int> d_size(1);

        GpuTimer solve;
        solve.start();

        seed<<<grid_for(g.n), kBlock>>>(d_dist.get(), d_queued.get(), g.n,
                                        source, qa.get(), d_size.get());

        int* cur = qa.get();
        int* next = qb.get();
        int frontier = 1;

        while (frontier > 0 && r.rounds < g.n) {
            d_size.fill_zero();
            clear_queued<<<grid_for(frontier), kBlock>>>(cur, frontier,
                                                         d_queued.get());

            int grid = grid_for(int64_t(frontier) * kWarp);
            expand<<<grid, kBlock>>>(d.offsets.get(), d.targets.get(),
                                     d.weights.get(), d_dist.get(), cur,
                                     frontier, next, d_size.get(),
                                     d_queued.get());

            d_size.download(&frontier, 1);
            std::swap(cur, next);
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
    ResidentCsr csr_;
};

}  // namespace

std::unique_ptr<Solver> make_gpu_frontier() {
    return std::make_unique<GpuFrontier>();
}

}  // namespace sssp
