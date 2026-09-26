#include <vector>

#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

// Sweeps every vertex per round, stops when a round changes nothing. Sequential
// counterpart of gpu-topo.
class BellmanFord : public Solver {
public:
    const char* name() const override { return "bellman-ford"; }
    const char* device() const override { return "cpu"; }

    Run run(const Graph& g, int source) override {
        Timer t;
        Run r;
        r.dist.assign(g.n, kInf);
        r.dist[source] = 0;

        for (int round = 0; round < g.n; ++round) {
            bool changed = false;
            for (int u = 0; u < g.n; ++u) {
                Weight du = r.dist[u];
                if (du == kInf) continue;
                for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                    Weight nd = du + g.weights[i];
                    int v = g.targets[i];
                    if (nd < r.dist[v]) {
                        r.dist[v] = nd;
                        changed = true;
                    }
                }
            }
            r.rounds++;
            if (!changed) break;
        }

        r.timing.solve_ms = t.ms();
        return r;
    }
};

}  // namespace

std::unique_ptr<Solver> make_bellman_ford() {
    return std::make_unique<BellmanFord>();
}

}  // namespace sssp
