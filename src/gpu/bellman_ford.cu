#include "cuda_common.cuh"
#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// One thread per vertex, every reached vertex every round. `work`, when set,
// counts the edges and vertices scanned.
__global__ void relax_by_vertex(const int* __restrict__ offsets,
                                const int* __restrict__ targets,
                                const Weight* __restrict__ weights,
                                Weight* __restrict__ dist, int n,
                                int* __restrict__ changed,
                                unsigned long long* __restrict__ work) {
    unsigned long long edges = 0, verts = 0;
    int stride = blockDim.x * gridDim.x;
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += stride) {
        Weight du = dist[u];
        if (du == kInf) continue;
        int begin = offsets[u], end = offsets[u + 1];
        edges += end - begin;
        verts++;
        for (int i = begin; i < end; ++i) {
            Weight nd = du + weights[i];
            if (nd < atomicMin(&dist[targets[i]], nd)) *changed = 1;
        }
    }
    if (work) {
        warp_count(&work[0], edges);
        warp_count(&work[1], verts);
    }
}

// One thread per edge, so a hub's edges are spread over many threads.
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
    void set_counters(bool on) override { counters_ = on; }
    void set_instrument(bool on) override { clock_.enable(on); }

    Run run(const Graph& g, int source) override {
        Run r;
        Counters& c = r.counters;

        if (csr_.bind(g, r.timing)) {
            if (edge_parallel_) {
                // The source of every edge, built on the host.
                Timer host;
                std::vector<int> sources = g.edge_sources();
                r.timing.prep_ms = host.ms();
                Timer alloc;
                d_sources_.alloc(sources.size());
                r.timing.alloc_ms += alloc.ms();
                GpuTimer up;
                up.start();
                d_sources_.copy_from(sources.data(), sources.size());
                r.timing.h2d_ms += up.stop();
            }
            Timer alloc;
            d_dist_.alloc(g.n);
            d_changed_.alloc(1);
            d_work_.alloc(2);
            r.timing.alloc_ms += alloc.ms();
        }
        const DeviceCsr& d = csr_.get();
        unsigned long long* work = counters_ && !edge_parallel_ ? d_work_.get() : nullptr;
        if (work) d_work_.fill_zero();
        clock_.reset();

        GpuTimer solve;
        solve.start();

        const int vgrid = grid_for(g.n);
        const int egrid = grid_for(d.m);
        clock_.start();
        init_dist<<<vgrid, kBlock>>>(d_dist_.get(), g.n, source);
        clock_.stop();

        int changed = 0;
        for (int round = 0; round < g.n; ++round) {
            d_changed_.fill_zero();
            clock_.start();
            if (edge_parallel_) {
                relax_by_edge<<<egrid, kBlock>>>(d_sources_.get(), d.targets.get(),
                                                 d.weights.get(), d_dist_.get(),
                                                 d.m, d_changed_.get());
            } else {
                relax_by_vertex<<<vgrid, kBlock>>>(
                    d.offsets.get(), d.targets.get(), d.weights.get(),
                    d_dist_.get(), g.n, d_changed_.get(), work);
            }
            clock_.stop();
            d_changed_.copy_to(&changed, 1);
            c.host_syncs++;
            clock_.drain();
            c.iterations++;
            if (changed == 0) break;
        }
        c.sync_rounds = c.iterations;

        r.timing.solve_ms = solve.stop();
        if (clock_.enabled()) r.timing.kernel_ms = clock_.total_ms();

        r.timing.d2h_ms = download_result(d_dist_, r.dist, g.n);

        if (edge_parallel_) {
            c.edges_touched = d.m * c.iterations;
        } else if (work) {
            c.edges_touched = read_counter(d_work_, 0);
            c.vertices_expanded = read_counter(d_work_, 1);
        }
        return r;
    }

    void release() override {
        csr_.release();
        d_sources_.free();
        d_dist_.free();
        d_changed_.free();
        d_work_.free();
    }

private:
    bool edge_parallel_;
    bool counters_ = true;
    KernelClock clock_;
    ResidentCsr csr_;
    DeviceBuffer<int> d_sources_;
    DeviceBuffer<Weight> d_dist_;
    DeviceBuffer<int> d_changed_;
    DeviceBuffer<unsigned long long> d_work_;
};

}  // namespace

std::unique_ptr<Solver> make_gpu_bellman_ford_topo() {
    return std::make_unique<GpuBellmanFord>(false);
}

std::unique_ptr<Solver> make_gpu_bellman_ford_edge() {
    return std::make_unique<GpuBellmanFord>(true);
}

}  // namespace sssp
