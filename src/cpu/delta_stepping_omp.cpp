#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "delta.hpp"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

using Buckets = std::vector<std::vector<int>>;
using Staged = std::vector<Buckets>;

// Below this many vertices a phase runs serially: forking threads to relax a
// few hundred edges costs more than it saves.
constexpr size_t kSerialCutoff = 4096;

inline Weight load_dist(const Weight* dist, int v) {
    return __atomic_load_n(&dist[v], __ATOMIC_RELAXED);
}

// Lock-free min-write. True if this thread lowered dist[v], which is who gets
// to enqueue v.
inline bool relax_atomic(Weight* dist, int v, Weight nd) {
    Weight old = __atomic_load_n(&dist[v], __ATOMIC_RELAXED);
    while (nd < old) {
        if (__atomic_compare_exchange_n(&dist[v], &old, nd, true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return true;
        }
    }
    return false;
}

// Relaxes one vertex's light or heavy edges into `out`: the shared bucket ring
// on the serial path, a private staging area on the parallel one.
inline void relax_vertex(const Graph& g, Weight* dist, int u, Weight du,
                         int delta, int nb, bool light, Buckets& out,
                         int64_t& highest) {
    for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
        if ((g.weights[i] <= delta) != light) continue;
        Weight nd = du + g.weights[i];
        if (relax_atomic(dist, g.targets[i], nd)) {
            int64_t b = nd / delta;
            out[b % nb].push_back(g.targets[i]);
            highest = std::max(highest, b);
        }
    }
}

// Buckets are independent, so one bucket per thread needs no synchronisation.
// Draining serially would leave an O(m) sequential section per phase.
void merge_staged(Staged& staged, Buckets& bucket, int nb) {
#pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; ++b) {
        for (auto& mine : staged) {
            if (mine[b].empty()) continue;
            bucket[b].insert(bucket[b].end(), mine[b].begin(), mine[b].end());
            mine[b].clear();
        }
    }
}

// One pass over `work`. A light pass (settled != null) drops vertices that moved
// to a lower bucket and appends the survivors to *settled; a heavy pass just
// fires. Returns the highest absolute bucket index written.
int64_t expand(const Graph& g, Weight* dist, const std::vector<int>& work,
               int delta, int nb, int64_t idx, Staged& staged, Buckets& bucket,
               std::vector<int>* settled) {
    const bool light = settled != nullptr;
    int64_t highest = 0;

    if (work.size() < kSerialCutoff) {
        for (int u : work) {
            Weight du = load_dist(dist, u);
            if (light) {
                if (du / delta != idx) continue;
                settled->push_back(u);
            }
            relax_vertex(g, dist, u, du, delta, nb, light, bucket, highest);
        }
        return highest;
    }

    const size_t base = light ? settled->size() : 0;
    if (light) settled->resize(base + work.size());
    int kept = 0;

    // num_threads pins the team to the staging areas allocated for it: a larger
    // team would index past the end of `staged`.
#pragma omp parallel num_threads(int(staged.size())) reduction(max : highest)
    {
        Buckets& mine = staged[omp_get_thread_num()];
        std::vector<int> keep;

#pragma omp for schedule(dynamic, 64) nowait
        for (size_t k = 0; k < work.size(); ++k) {
            int u = work[k];
            Weight du = load_dist(dist, u);
            if (light) {
                if (du / delta != idx) continue;
                keep.push_back(u);
            }
            relax_vertex(g, dist, u, du, delta, nb, light, mine, highest);
        }

        if (light && !keep.empty()) {
            int off;
#pragma omp atomic capture
            {
                off = kept;
                kept += static_cast<int>(keep.size());
            }
            std::copy(keep.begin(), keep.end(), settled->begin() + base + off);
        }
    }

    if (light) settled->resize(base + kept);
    merge_staged(staged, bucket, nb);
    return highest;
}

// Serial bucket structure, parallel phases. The only shared writes on the hot
// path are the CAS loops on dist; queue insertions stage privately first.
class DeltaSteppingOmp : public Solver {
public:
    const char* name() const override { return "delta-omp"; }
    const char* device() const override { return "cpu"; }
    void set_delta(int d) override { forced_delta_ = d; }

    Run run(const Graph& g, int source) override {
        Timer t;
        Run r;
        r.dist.assign(g.n, kInf);

        const int delta = forced_delta_ > 0 ? forced_delta_ : pick_delta(g);
        const int nb = bucket_count(g, delta);

        Buckets bucket(nb);
        Staged staged(cpu_threads(), Buckets(nb));

        Weight* dist = r.dist.data();
        dist[source] = 0;
        bucket[0].push_back(source);

        std::vector<int> frontier;
        std::vector<int> settled;
        int64_t highest = 0;

        for (int64_t idx = 0; idx <= highest; ++idx) {
            if (bucket[idx % nb].empty()) continue;
            r.rounds++;
            settled.clear();

            while (!bucket[idx % nb].empty()) {
                frontier.swap(bucket[idx % nb]);
                bucket[idx % nb].clear();
                highest = std::max(highest, expand(g, dist, frontier, delta, nb,
                                                   idx, staged, bucket,
                                                   &settled));
            }

            highest = std::max(highest, expand(g, dist, settled, delta, nb, idx,
                                               staged, bucket, nullptr));
        }

        r.timing.solve_ms = t.ms();
        return r;
    }

private:
    int forced_delta_ = 0;
};

}  // namespace

std::unique_ptr<Solver> make_delta_stepping_omp() {
    return std::make_unique<DeltaSteppingOmp>();
}

}  // namespace sssp
