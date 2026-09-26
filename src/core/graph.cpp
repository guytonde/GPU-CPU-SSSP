#include "sssp/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sssp {

namespace {

constexpr char kMagic[4] = {'S', 'S', 'S', 'P'};
constexpr uint32_t kVersion = 1;

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

Weight Graph::max_weight() const {
    Weight m = 0;
    for (Weight w : weights) m = std::max(m, w);
    return m;
}

double Graph::avg_degree() const {
    if (n == 0) return 0.0;
    return static_cast<double>(num_edges()) / n;
}

GraphStats graph_stats(const Graph& g) {
    GraphStats s;
    s.n = g.n;
    s.m = g.num_edges();
    s.avg_degree = g.avg_degree();

    double sumsq = 0.0;
    for (int u = 0; u < g.n; ++u) {
        int d = g.degree(u);
        s.max_degree = std::max(s.max_degree, d);
        double dev = d - s.avg_degree;
        sumsq += dev * dev;
    }
    if (g.n > 0) s.degree_stddev = std::sqrt(sumsq / g.n);

    if (!g.weights.empty()) {
        auto mm = std::minmax_element(g.weights.begin(), g.weights.end());
        s.min_weight = *mm.first;
        s.max_weight = *mm.second;
    }
    return s;
}

SourceStats source_stats(const Graph& g, int source) {
    SourceStats s;
    if (g.n == 0) return s;

    std::vector<char> seen(g.n, 0);
    std::vector<int> cur{source}, next;
    seen[source] = 1;
    s.reached = 1;

    int64_t total = 0;
    while (!cur.empty()) {
        s.max_frontier = std::max<int>(s.max_frontier, int(cur.size()));
        total += int64_t(cur.size());
        s.levels++;

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

    s.reached = 0;
    for (char c : seen) s.reached += c;
    s.avg_frontier = s.levels ? double(total) / s.levels : 0.0;
    return s;
}

void check_weights(const Graph& g) {
    for (Weight w : g.weights) {
        if (w < 0) {
            throw std::runtime_error("negative edge weight " +
                                     std::to_string(w) +
                                     ": these solvers require w >= 0");
        }
    }
}

std::vector<int> Graph::edge_sources() const {
    std::vector<int> src(num_edges());
    for (int u = 0; u < n; ++u) {
        for (int i = offsets[u]; i < offsets[u + 1]; ++i) src[i] = u;
    }
    return src;
}

Graph build_csr(const EdgeList& el) {
    Graph g;
    g.n = el.n;
    g.offsets.assign(g.n + 1, 0);

    for (size_t i = 0; i < el.size(); ++i) {
        if (el.u[i] < 0 || el.u[i] >= g.n || el.v[i] < 0 || el.v[i] >= g.n) {
            throw std::runtime_error("edge endpoint out of range for n=" +
                                     std::to_string(g.n));
        }
        g.offsets[el.u[i] + 1]++;
    }
    for (int u = 0; u < g.n; ++u) g.offsets[u + 1] += g.offsets[u];

    g.targets.resize(el.size());
    g.weights.resize(el.size());
    std::vector<int> cursor(g.offsets.begin(), g.offsets.end() - 1);
    for (size_t i = 0; i < el.size(); ++i) {
        int pos = cursor[el.u[i]]++;
        g.targets[pos] = el.v[i];
        g.weights[pos] = el.w[i];
    }
    return g;
}

void dedup(EdgeList& el) {
    std::vector<size_t> idx(el.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (el.u[a] != el.u[b]) return el.u[a] < el.u[b];
        if (el.v[a] != el.v[b]) return el.v[a] < el.v[b];
        return el.w[a] < el.w[b];
    });

    EdgeList out;
    out.n = el.n;
    int last_u = -1, last_v = -1;
    for (size_t i : idx) {
        if (el.u[i] == last_u && el.v[i] == last_v) continue;
        last_u = el.u[i];
        last_v = el.v[i];
        out.add(el.u[i], el.v[i], el.w[i]);
    }
    el = std::move(out);
}

EdgeList read_edge_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    EdgeList el;
    long long max_vertex = -1;
    std::string line;

    // "u v" is an unweighted edge, not a header: a header that looked like one
    // used to swallow the first edge of an unweighted file. Metadata goes on a
    // "#" line, which is what write_edge_list emits.
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;

        std::istringstream ss(line);
        long long a, b, w;
        if (!(ss >> a >> b)) continue;
        if (!(ss >> w)) w = 1;

        if (a < 0 || b < 0 || a > INT32_MAX || b > INT32_MAX) {
            throw std::runtime_error("vertex id out of range in " + path);
        }
        el.add(static_cast<int>(a), static_cast<int>(b),
               static_cast<Weight>(w));
        max_vertex = std::max(max_vertex, std::max(a, b));
    }

    el.n = static_cast<int>(max_vertex + 1);
    return el;
}

void write_edge_list(const std::string& path, const EdgeList& el) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);

    out << "# vertices=" << el.n << " edges=" << el.size() << "\n";
    // ostream per line is the bottleneck on large graphs so buffer and flush in chunks
    std::string buf;
    buf.reserve(1 << 20);
    char tmp[64];
    for (size_t i = 0; i < el.size(); ++i) {
        int len = std::snprintf(tmp, sizeof(tmp), "%d %d %d\n", el.u[i], el.v[i],
                                el.w[i]);
        buf.append(tmp, len);
        if (buf.size() > (1u << 20)) {
            out.write(buf.data(), buf.size());
            buf.clear();
        }
    }
    out.write(buf.data(), buf.size());
}

void write_binary(const std::string& path, const Graph& g) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);

    int64_t m = g.num_edges();
    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char*>(&g.n), sizeof(g.n));
    out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    out.write(reinterpret_cast<const char*>(g.offsets.data()),
              (g.n + 1) * sizeof(int));
    out.write(reinterpret_cast<const char*>(g.targets.data()), m * sizeof(int));
    out.write(reinterpret_cast<const char*>(g.weights.data()),
              m * sizeof(Weight));
}

Graph read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    char magic[4];
    uint32_t version;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (std::memcmp(magic, kMagic, 4) != 0) {
        throw std::runtime_error(path + " is not an sssp binary graph");
    }
    if (version != kVersion) {
        throw std::runtime_error("unsupported graph version " +
                                 std::to_string(version));
    }

    Graph g;
    int64_t m = 0;
    in.read(reinterpret_cast<char*>(&g.n), sizeof(g.n));
    in.read(reinterpret_cast<char*>(&m), sizeof(m));

    g.offsets.resize(g.n + 1);
    g.targets.resize(m);
    g.weights.resize(m);
    in.read(reinterpret_cast<char*>(g.offsets.data()), (g.n + 1) * sizeof(int));
    in.read(reinterpret_cast<char*>(g.targets.data()), m * sizeof(int));
    in.read(reinterpret_cast<char*>(g.weights.data()), m * sizeof(Weight));
    if (!in) throw std::runtime_error("truncated graph file " + path);

    return g;
}

Graph load_graph(const std::string& path) {
    if (ends_with(path, ".bin")) return read_binary(path);
    return build_csr(read_edge_list(path));
}

void save_graph(const std::string& path, const EdgeList& el) {
    if (ends_with(path, ".bin")) {
        write_binary(path, build_csr(el));
    } else {
        write_edge_list(path, el);
    }
}

}  // namespace sssp
