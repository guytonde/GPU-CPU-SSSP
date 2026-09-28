#include "sssp/reference.hpp"

#include <functional>
#include <queue>
#include <tuple>

namespace sssp {

std::vector<Weight> reference_sssp(const Graph& g, int source) {
    std::vector<Weight> dist(g.n, kInf);
    dist[source] = 0;

    for (int round = 0; round + 1 < g.n; ++round) {
        for (int u = 0; u < g.n; ++u) {
            if (dist[u] == kInf) continue;
            for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                Weight nd = dist[u] + g.weights[i];
                if (nd < dist[g.targets[i]]) dist[g.targets[i]] = nd;
            }
        }
    }
    return dist;
}

std::vector<Weight> reference_sssp(const EdgeList& el, int source) {
    std::vector<Weight> dist(el.n, kInf);
    dist[source] = 0;

    for (int round = 0; round + 1 < el.n; ++round) {
        for (size_t i = 0; i < el.size(); ++i) {
            if (dist[el.u[i]] == kInf) continue;
            Weight nd = dist[el.u[i]] + el.w[i];
            if (nd < dist[el.v[i]]) dist[el.v[i]] = nd;
        }
    }
    return dist;
}

// (distance, hops) is a nonnegative lexicographic weight whenever w >= 0, so
// Dijkstra over it settles every vertex with its shortest distance and, among
// the shortest paths, the fewest hops.
RefResult reference_dijkstra(const Graph& g, int source) {
    RefResult r;
    r.dist.assign(g.n, kInf);
    r.hops.assign(g.n, -1);

    using Key = std::tuple<Weight, int, int>;  // dist, hops, vertex
    std::priority_queue<Key, std::vector<Key>, std::greater<Key>> open;
    r.dist[source] = 0;
    r.hops[source] = 0;
    open.emplace(0, 0, source);

    std::vector<char> done(g.n, 0);
    while (!open.empty()) {
        auto [d, h, u] = open.top();
        open.pop();
        if (done[u]) continue;
        done[u] = 1;
        for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
            int v = g.targets[i];
            Weight nd = d + g.weights[i];
            int nh = h + 1;
            if (nd < r.dist[v] || (nd == r.dist[v] && nh < r.hops[v])) {
                r.dist[v] = nd;
                r.hops[v] = nh;
                open.emplace(nd, nh, v);
            }
        }
    }
    return r;
}

}  // namespace sssp
