#include "sssp/graph.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sssp {

namespace {

constexpr char kMagic[4] = {'S', 'S', 'S', 'P'};
constexpr uint32_t kVersion = 2;

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

void Meta::set(const std::string& key, const std::string& value) {
    if (key.find_first_of("=\n") != std::string::npos ||
        value.find('\n') != std::string::npos) {
        throw std::runtime_error("meta key/value may not contain '=' or newline: " + key);
    }
    for (auto& kv : items) {
        if (kv.first == key) {
            kv.second = value;
            return;
        }
    }
    items.emplace_back(key, value);
}

std::string Meta::get(const std::string& key, const std::string& fallback) const {
    for (const auto& kv : items) {
        if (kv.first == key) return kv.second;
    }
    return fallback;
}

bool Meta::has(const std::string& key) const {
    for (const auto& kv : items) {
        if (kv.first == key) return true;
    }
    return false;
}

std::string Meta::serialize() const {
    std::string out;
    for (const auto& kv : items) out += kv.first + "=" + kv.second + "\n";
    return out;
}

Meta Meta::parse(const std::string& text) {
    Meta m;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        m.items.emplace_back(line.substr(0, eq), line.substr(eq + 1));
    }
    return m;
}

Weight Graph::max_weight() const {
    Weight m = 0;
    for (Weight w : weights) m = std::max(m, w);
    return m;
}

double Graph::avg_degree() const {
    if (n == 0) return 0.0;
    return static_cast<double>(num_edges()) / n;
}

std::vector<int> Graph::edge_sources() const {
    std::vector<int> src(num_edges());
    for (int u = 0; u < n; ++u) {
        for (int i = offsets[u]; i < offsets[u + 1]; ++i) src[i] = u;
    }
    return src;
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

    std::vector<std::pair<int, Weight>> row;
    for (int u = 0; u < g.n; ++u) {
        int b = g.offsets[u], e = g.offsets[u + 1];
        if (e - b < 2) continue;
        row.clear();
        for (int i = b; i < e; ++i) row.emplace_back(g.targets[i], g.weights[i]);
        std::sort(row.begin(), row.end());
        for (int i = b; i < e; ++i) {
            g.targets[i] = row[i - b].first;
            g.weights[i] = row[i - b].second;
        }
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

// Layout: "SSSP", u32 version, i32 n, i64 m, u32 meta bytes, meta text,
// (n+1) i32 offsets, m i32 targets, m i32 weights.
void write_binary(const std::string& path, const Graph& g) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);

    int64_t m = g.num_edges();
    std::string meta = g.meta.serialize();
    uint32_t meta_len = static_cast<uint32_t>(meta.size());
    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char*>(&g.n), sizeof(g.n));
    out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    out.write(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len));
    out.write(meta.data(), meta_len);
    out.write(reinterpret_cast<const char*>(g.offsets.data()),
              (g.n + 1) * sizeof(int));
    out.write(reinterpret_cast<const char*>(g.targets.data()), m * sizeof(int));
    out.write(reinterpret_cast<const char*>(g.weights.data()),
              m * sizeof(Weight));
    if (!out) throw std::runtime_error("short write to " + path);
}

Graph read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    char magic[4];
    uint32_t version = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || std::memcmp(magic, kMagic, 4) != 0) {
        throw std::runtime_error(path + " is not an sssp binary graph");
    }
    if (version != kVersion) {
        throw std::runtime_error("unsupported graph version " +
                                 std::to_string(version) +
                                 "; regenerate it with bin/gen_graph");
    }

    Graph g;
    int64_t m = 0;
    uint32_t meta_len = 0;
    in.read(reinterpret_cast<char*>(&g.n), sizeof(g.n));
    in.read(reinterpret_cast<char*>(&m), sizeof(m));
    in.read(reinterpret_cast<char*>(&meta_len), sizeof(meta_len));
    if (!in || g.n < 0 || m < 0) throw std::runtime_error("bad header in " + path);

    std::string meta(meta_len, '\0');
    in.read(meta.data(), meta_len);
    g.meta = Meta::parse(meta);

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
    Graph g = build_csr(read_edge_list(path));
    g.meta.set("topology", "file");
    g.meta.set("source_file", path);
    return g;
}

void save_graph(const std::string& path, const Graph& g) {
    if (ends_with(path, ".bin")) {
        write_binary(path, g);
        return;
    }
    EdgeList el;
    el.n = g.n;
    for (int u = 0; u < g.n; ++u) {
        for (int i = g.offsets[u]; i < g.offsets[u + 1]; ++i) {
            el.add(u, g.targets[i], g.weights[i]);
        }
    }
    write_edge_list(path, el);
}

uint64_t hash_words(const void* data, size_t bytes, uint64_t seed) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t h = seed ^ (bytes * 0x9e3779b97f4a7c15ull);
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t k;
        std::memcpy(&k, p + i, 8);
        k *= 0xff51afd7ed558ccdull;
        k ^= k >> 33;
        h ^= k;
        h *= 0xc4ceb9fe1a85ec53ull;
    }
    for (; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

uint64_t topo_hash(const Graph& g) {
    uint64_t h = hash_words(&g.n, sizeof(g.n), 0x5353535053535350ull);
    h = hash_words(g.offsets.data(), g.offsets.size() * sizeof(int), h);
    return hash_words(g.targets.data(), g.targets.size() * sizeof(int), h);
}

uint64_t weights_hash(const Graph& g) {
    return hash_words(g.weights.data(), g.weights.size() * sizeof(Weight),
                      0x7765696768747321ull);
}

uint64_t graph_hash(const Graph& g) {
    return hash_words(&g.n, 0, topo_hash(g) ^ (weights_hash(g) * 0x9e3779b97f4a7c15ull));
}

uint64_t dist_hash(const std::vector<Weight>& dist) {
    return hash_words(dist.data(), dist.size() * sizeof(Weight), 0x6469737468617368ull);
}

std::string hex64(uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

}  // namespace sssp
