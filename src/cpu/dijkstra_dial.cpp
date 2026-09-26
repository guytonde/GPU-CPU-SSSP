#include <algorithm>
#include <vector>

#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// Dial's algorithm: settle in increasing order out of a ring of maxw+1 buckets,
// no heap and no log factor. Scanning for the next non-empty bucket costs
// O(n * maxw), so this only pays off for small integer weights.
class DijkstraDial : public Solver {
public:
    const char* name() const override { return "dial"; }
    const char* device() const override { return "cpu"; }

    Run run(const Graph& g, int source) override {
        Timer t;
        Run r;
        r.dist.assign(g.n, kInf);

        const Weight maxw = std::max<Weight>(1, g.max_weight());
        const int nb = maxw + 1;
        std::vector<std::vector<int>> bucket(nb);

        r.dist[source] = 0;
        bucket[0].push_back(source);

        Weight d = 0;
        int remaining = 1;
        while (remaining > 0) {
            while (bucket[d % nb].empty()) ++d;
            auto& b = bucket[d % nb];

            r.rounds++;
            while (!b.empty()) {
                int u = b.back();
                b.pop_back();
                --remaining;
                if (r.dist[u] != d) continue;  // stale entry from an earlier relax

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

        r.timing.solve_ms = t.ms();
        return r;
    }
};

}  // namespace

std::unique_ptr<Solver> make_dijkstra_dial() {
    return std::make_unique<DijkstraDial>();
}

}  // namespace sssp
