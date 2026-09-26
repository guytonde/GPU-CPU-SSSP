#include <algorithm>
#include <vector>

#include "delta.hpp"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// Meyer and Sanders delta-stepping. A bucket of width delta is relaxed to a
// fixed point over its light edges (w <= delta), then its heavy edges fire once.
// The ring is maxw/delta + 2 wide: bucket i only ever writes into buckets i
// through i + maxw/delta.
class DeltaStepping : public Solver {
public:
    const char* name() const override { return "delta"; }
    const char* device() const override { return "cpu"; }
    void set_delta(int d) override { forced_delta_ = d; }

    Run run(const Graph& g, int source) override {
        Timer t;
        Run r;
        r.dist.assign(g.n, kInf);

        const int delta = forced_delta_ > 0 ? forced_delta_ : pick_delta(g);
        const int nb = bucket_count(g, delta);
        std::vector<std::vector<int>> bucket(nb);

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
            r.rounds++;
            settled.clear();

            while (!b.empty()) {
                frontier.swap(b);
                b.clear();
                for (int u : frontier) {
                    if (r.dist[u] / delta != idx) continue;  // moved to a lower bucket
                    settled.push_back(u);
                    for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                        if (g.weights[i] > delta) continue;
                        relax(g.targets[i], r.dist[u] + g.weights[i]);
                    }
                }
            }

            for (int u : settled) {
                for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                    if (g.weights[i] <= delta) continue;
                    relax(g.targets[i], r.dist[u] + g.weights[i]);
                }
            }
        }

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
