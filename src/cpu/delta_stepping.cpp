#include <algorithm>
#include <vector>

#include "sssp/delta.hpp"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// Meyer and Sanders delta stepping. Each bucket relaxes its light edges
// (w <= delta) until it stays empty, then its heavy edges once. Bucket i only
// writes into buckets i to i + maxw/delta, so a ring of maxw/delta + 2 suffices.
class DeltaStepping : public Solver {
public:
    const char* name() const override { return "delta"; }
    const char* device() const override { return "cpu"; }
    void set_delta(int d) override { forced_delta_ = d; }

    Run run(const Graph& g, int source) override {
        Run r;
        Timer alloc;
        const int delta = forced_delta_ > 0 ? forced_delta_ : pick_delta(g);
        const int nb = bucket_count(g, delta);
        std::vector<std::vector<int>> bucket(nb);
        r.dist.reserve(g.n);
        r.timing.alloc_ms = alloc.ms();
        r.delta_used = delta;

        Timer t;
        r.dist.assign(g.n, kInf);
        Counters& c = r.counters;

        int64_t highest = 0;
        auto relax = [&](int v, Weight nd) {
            if (nd < r.dist[v]) {
                r.dist[v] = nd;
                int64_t b = nd / delta;
                bucket[b % nb].push_back(v);
                highest = std::max(highest, b);
            }
        };

        r.dist[source] = 0;
        bucket[0].push_back(source);

        std::vector<int> frontier;
        std::vector<int> settled;

        for (int64_t idx = 0; idx <= highest; ++idx) {
            auto& b = bucket[idx % nb];
            if (b.empty()) continue;
            c.iterations++;
            settled.clear();

            while (!b.empty()) {
                frontier.swap(b);
                b.clear();
                c.sync_rounds++;
                for (int u : frontier) {
                    if (r.dist[u] / delta != idx) continue;  // moved lower
                    settled.push_back(u);
                    c.vertices_expanded++;
                    c.edges_touched += g.offsets[u + 1] - g.offsets[u];
                    for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                        if (g.weights[i] > delta) continue;
                        relax(g.targets[i], r.dist[u] + g.weights[i]);
                    }
                }
            }

            c.sync_rounds++;
            for (int u : settled) {
                c.edges_touched += g.offsets[u + 1] - g.offsets[u];
                for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                    if (g.weights[i] <= delta) continue;
                    relax(g.targets[i], r.dist[u] + g.weights[i]);
                }
            }
        }

        std::vector<std::vector<int>>().swap(bucket);
        r.timing.solve_ms = t.ms();
        return r;
    }

private:
    int forced_delta_ = 0;
};

}  // namespace

std::unique_ptr<Solver> make_delta_stepping() {
    return std::make_unique<DeltaStepping>();
}

}  // namespace sssp
