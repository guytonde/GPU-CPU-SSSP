#include "sssp/reference.hpp"

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

}  // namespace sssp
