#include "sssp/stats.hpp"

#include <algorithm>
#include <numeric>

namespace sssp {

namespace {

int find_root(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

}  // namespace

std::vector<int> component_ids(const Graph& g, int* count) {
    std::vector<int> parent(g.n);
    std::iota(parent.begin(), parent.end(), 0);
    for (int u = 0; u < g.n; ++u) {
        for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
            int a = find_root(parent, u), b = find_root(parent, g.targets[i]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
    }

    std::vector<int> root(g.n);
    std::vector<int64_t> size(g.n, 0);
    for (int u = 0; u < g.n; ++u) {
        root[u] = find_root(parent, u);
        size[root[u]]++;
    }

    std::vector<int> roots;
    for (int u = 0; u < g.n; ++u) {
        if (root[u] == u) roots.push_back(u);
    }
    std::stable_sort(roots.begin(), roots.end(),
                     [&](int a, int b) { return size[a] > size[b]; });
    std::vector<int> dense(g.n, -1);
    for (size_t k = 0; k < roots.size(); ++k) dense[roots[k]] = int(k);

    std::vector<int> id(g.n);
    for (int u = 0; u < g.n; ++u) id[u] = dense[root[u]];
    if (count) *count = int(roots.size());
    return id;
}

GraphStats graph_stats(const Graph& g) {
    GraphStats s;
    s.n = g.n;
    s.m = g.num_edges();
    if (g.n == 0) return s;
    for (int u = 0; u < g.n; ++u) s.deg_max = std::max(s.deg_max, g.degree(u));

    std::vector<int> comp = component_ids(g, &s.components);
    s.giant_frac = double(std::count(comp.begin(), comp.end(), 0)) / g.n;

    if (!g.weights.empty()) {
        auto mm = std::minmax_element(g.weights.begin(), g.weights.end());
        s.w_min = *mm.first;
        s.w_max = *mm.second;
    }
    return s;
}

SourceStats source_stats(const Graph& g, int source, const RefResult& ref,
                         const std::vector<int>& comp) {
    SourceStats s;
    s.source = source;
    if (g.n == 0) return s;
    s.source_degree = g.degree(source);
    s.in_giant = !comp.empty() && comp[source] == 0;

    std::vector<char> seen(g.n, 0);
    std::vector<int> cur{source}, next;
    seen[source] = 1;
    double sum = 0.0, sumsq = 0.0;
    while (!cur.empty()) {
        double f = double(cur.size());
        s.bfs_frontier_max = std::max<int>(s.bfs_frontier_max, int(cur.size()));
        sum += f;
        sumsq += f * f;
        s.bfs_levels++;

        next.clear();
        for (int u : cur) {
            for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
                int v = g.targets[i];
                if (seen[v]) continue;
                seen[v] = 1;
                next.push_back(v);
            }
        }
        cur.swap(next);
    }
    s.bfs_frontier_wmean = sum > 0 ? sumsq / sum : 0.0;

    double hop_sum = 0.0;
    for (int u = 0; u < g.n; ++u) {
        if (ref.dist[u] == kInf) continue;
        s.reached++;
        s.m_reached += g.degree(u);
        s.sp_depth_max = std::max(s.sp_depth_max, ref.hops[u]);
        hop_sum += ref.hops[u];
        s.max_dist = std::max<int64_t>(s.max_dist, ref.dist[u]);
    }
    s.sp_depth_mean = s.reached ? hop_sum / s.reached : 0.0;
    return s;
}

}  // namespace sssp
