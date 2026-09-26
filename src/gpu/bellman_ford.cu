#include "cuda_common.cuh"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// One thread per vertex, every vertex every round. No frontier, no queue, so the
// cost stays O(rounds * m) however few vertices changed.
__global__ void relax_by_vertex(const int* __restrict__ offsets,
                                const int* __restrict__ targets,
                                const Weight* __restrict__ weights,
                                Weight* __restrict__ dist, int n,
                                int* __restrict__ changed) {
    int stride = blockDim.x * gridDim.x;
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += stride) {
        Weight du = dist[u];
        if (du == kInf) continue;
        for (int i = offsets[u]; i < offsets[u + 1]; ++i) {
            Weight nd = du + weights[i];
            if (nd < atomicMin(&dist[targets[i]], nd)) *changed = 1;
        }
    }
}

// One thread per edge. Same asymptotic cost, but a skewed degree distribution
// spreads evenly instead of one lane grinding a hub while 31 idle.
__global__ void relax_by_edge(const int* __restrict__ sources,
                              const int* __restrict__ targets,
                              const Weight* __restrict__ weights,
                              Weight* __restrict__ dist, int64_t m,
                              int* __restrict__ changed) {
    int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; e < m;
         e += stride) {
        Weight du = dist[sources[e]];
        if (du == kInf) continue;
        Weight nd = du + weights[e];
        if (nd < atomicMin(&dist[targets[e]], nd)) *changed = 1;
    }
}

__global__ void init_dist(Weight* dist, int n, int source) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dist[i] = (i == source) ? 0 : kInf;
    }
}

class GpuBellmanFord : public Solver {
public:
    explicit GpuBellmanFord(bool edge_parallel) : edge_parallel_(edge_parallel) {}

    const char* name() const override {
        return edge_parallel_ ? "gpu-edge" : "gpu-topo";
    }
    const char* device() const override { return "gpu"; }

    Run run(const Graph& g, int source) override {
        Run r;
        r.dist.assign(g.n, kInf);

        bool uploaded = csr_.bind(g, r.timing.h2d_ms);
        const DeviceCsr& d = csr_.get();

        if (edge_parallel_ && uploaded) {
            Timer host;
            std::vector<int> sources = g.edge_sources();
            r.timing.prep_ms = host.ms();

            GpuTimer up;
            up.start();
            d_sources_.upload(sources);
            r.timing.h2d_ms += up.stop();
        }

        DeviceBuffer<Weight> d_dist(g.n);
        DeviceBuffer<int> d_changed(1);

        GpuTimer solve;
        solve.start();

        const int vgrid = grid_for(g.n);
        const int egrid = grid_for(d.m);
        init_dist<<<vgrid, kBlock>>>(d_dist.get(), g.n, source);

        int changed = 0;
        for (int round = 0; round < g.n; ++round) {
            d_changed.fill_zero();
            if (edge_parallel_) {
                relax_by_edge<<<egrid, kBlock>>>(d_sources_.get(),
                                                 d.targets.get(),
                                                 d.weights.get(), d_dist.get(),
                                                 d.m, d_changed.get());
            } else {
                relax_by_vertex<<<vgrid, kBlock>>>(
                    d.offsets.get(), d.targets.get(), d.weights.get(),
                    d_dist.get(), g.n, d_changed.get());
            }
            d_changed.download(&changed, 1);
            r.rounds++;
            if (changed == 0) break;
        }

        r.timing.solve_ms = solve.stop();

        GpuTimer back;
        back.start();
        d_dist.download(r.dist.data(), g.n);
        r.timing.d2h_ms = back.stop();
        return r;
    }

    void release() override {
        csr_.release();
        d_sources_.free();
    }

private:
    bool edge_parallel_;
    ResidentCsr csr_;
    DeviceBuffer<int> d_sources_;
};

}  // namespace

std::unique_ptr<Solver> make_gpu_bellman_ford_topo() {
    return std::make_unique<GpuBellmanFord>(false);
}

std::unique_ptr<Solver> make_gpu_bellman_ford_edge() {
    return std::make_unique<GpuBellmanFord>(true);
}

}  // namespace sssp
