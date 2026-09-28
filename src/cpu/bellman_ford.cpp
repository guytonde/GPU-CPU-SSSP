#include <vector>

#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// Sweeps every vertex in order each round, updating in place, until a round
// changes nothing. The serial counterpart of gpu-topo.
class BellmanFord : public Solver {
public:
    const char* name() const override { return "bellman-ford"; }
    const char* device() const override { return "cpu"; }

    Run run(const Graph& g, int source) override {
        Run r;
        Timer alloc;
        r.dist.reserve(g.n);
        r.timing.alloc_ms = alloc.ms();

        Timer t;
        r.dist.assign(g.n, kInf);
        r.dist[source] = 0;

        Counters& c = r.counters;
        for (int round = 0; round < g.n; ++round) {
            bool changed = false;
            for (int u = 0; u < g.n; ++u) {
                Weight du = r.dist[u];
                if (du == kInf) continue;
                c.vertices_expanded++;
                c.edges_touched += g.offsets[u + 1] - g.offsets[u];
                for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                    Weight nd = du + g.weights[i];
                    int v = g.targets[i];
                    if (nd < r.dist[v]) {
                        r.dist[v] = nd;
                        changed = true;
                    }
                }
            }
            c.iterations++;
            if (!changed) break;
        }
        c.sync_rounds = c.iterations;

        r.timing.solve_ms = t.ms();
        return r;
    }
};

}  // namespace

std::unique_ptr<Solver> make_bellman_ford() {
    return std::make_unique<BellmanFord>();
}

}  // namespace sssp
