#include <queue>
#include <utility>

#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

namespace sssp {

namespace {

class DijkstraHeap : public Solver {
public:
    const char* name() const override { return "dijkstra"; }
    const char* device() const override { return "cpu"; }

    Run run(const Graph& g, int source) override {
        Timer t;
        Run r;
        r.dist.assign(g.n, kInf);

        using Item = std::pair<Weight, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

        r.dist[source] = 0;
        pq.emplace(0, source);

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();
            if (d > r.dist[u]) continue;
            r.rounds++;

            for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                int v = g.targets[i];
                Weight nd = d + g.weights[i];
                if (nd < r.dist[v]) {
                    r.dist[v] = nd;
                    pq.emplace(nd, v);
                }
            }
        }

        r.timing.solve_ms = t.ms();
        return r;
    }
};

}  // namespace

std::unique_ptr<Solver> make_dijkstra_heap() {
    return std::make_unique<DijkstraHeap>();
}

}  // namespace sssp
