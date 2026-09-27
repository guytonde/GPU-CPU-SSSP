#include <algorithm>
#include <vector>

#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// Dial's algorithm with a ring of maxw+1 buckets. Finding the next nonempty
// bucket costs O(max distance), so it suits small integer weights.
class DijkstraDial : public Solver {
public:
    const char* name() const override { return "dial"; }
    const char* device() const override { return "cpu"; }

    Run run(const Graph& g, int source) override {
        Run r;
        Timer alloc;
        const Weight maxw = std::max<Weight>(1, g.max_weight());
        const int nb = maxw + 1;
        std::vector<std::vector<int>> bucket(nb);
        r.dist.reserve(g.n);
        r.timing.alloc_ms = alloc.ms();

        Timer t;
        r.dist.assign(g.n, kInf);
        r.dist[source] = 0;
        bucket[0].push_back(source);

        Counters& c = r.counters;
        Weight d = 0;
        int remaining = 1;
        while (remaining > 0) {
            while (bucket[d % nb].empty()) ++d;
            auto& b = bucket[d % nb];

            c.iterations++;
            while (!b.empty()) {
                int u = b.back();
                b.pop_back();
                --remaining;
                if (r.dist[u] != d) continue;  // stale entry
                c.vertices_expanded++;
                c.edges_touched += g.offsets[u + 1] - g.offsets[u];

                for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                    int v = g.targets[i];
                    Weight nd = d + g.weights[i];
                    if (nd < r.dist[v]) {
                        r.dist[v] = nd;
                        bucket[nd % nb].push_back(v);
                        ++remaining;
                    }
                }
            }
        }

        // Freeing the ring counts as part of the query.
        std::vector<std::vector<int>>().swap(bucket);
        r.timing.solve_ms = t.ms();
        return r;
    }
};

}  // namespace

std::unique_ptr<Solver> make_dijkstra_dial() {
    return std::make_unique<DijkstraDial>();
}

}  // namespace sssp
