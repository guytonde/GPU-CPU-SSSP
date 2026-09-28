#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sssp {

using Weight = int32_t;

// Half of INT32_MAX, so dist[u] + w cannot overflow.
constexpr Weight kInf = 0x3fffffff;

// Ordered key=value parameters that travel with a .bin graph.
struct Meta {
    std::vector<std::pair<std::string, std::string>> items;

    void set(const std::string& key, const std::string& value);
    std::string get(const std::string& key, const std::string& fallback = "") const;
    bool has(const std::string& key) const;

    std::string serialize() const;
    static Meta parse(const std::string& text);
};

// CSR: the edges of u are [offsets[u], offsets[u+1]) in targets and weights.
struct Graph {
    int n = 0;
    std::vector<int> offsets;
    std::vector<int> targets;
    std::vector<Weight> weights;
    Meta meta;

    int64_t num_edges() const { return static_cast<int64_t>(targets.size()); }
    int degree(int u) const { return offsets[u + 1] - offsets[u]; }

    Weight max_weight() const;
    double avg_degree() const;

    std::vector<int> edge_sources() const;
};

struct EdgeList {
    int n = 0;
    std::vector<int> u;
    std::vector<int> v;
    std::vector<Weight> w;

    void add(int a, int b, Weight weight) {
        u.push_back(a);
        v.push_back(b);
        w.push_back(weight);
    }
    size_t size() const { return u.size(); }
};

// Each adjacency list comes out sorted by (target, weight), so the result does
// not depend on the order of the input edges.
Graph build_csr(const EdgeList& el);
void dedup(EdgeList& el);

// Throws on a negative weight.
void check_weights(const Graph& g);

// .bin files hold the CSR and its Meta. Any other path is a text edge list,
// one "u v [w]" per line, with w defaulting to 1 and # lines ignored.
Graph load_graph(const std::string& path);
void save_graph(const std::string& path, const Graph& g);

EdgeList read_edge_list(const std::string& path);
void write_edge_list(const std::string& path, const EdgeList& el);
Graph read_binary(const std::string& path);
void write_binary(const std::string& path, const Graph& g);

// topo_hash ignores the weights, so graphs that differ only in weights share it.
uint64_t hash_words(const void* data, size_t bytes, uint64_t seed);
uint64_t topo_hash(const Graph& g);
uint64_t weights_hash(const Graph& g);
uint64_t graph_hash(const Graph& g);
uint64_t dist_hash(const std::vector<Weight>& dist);
std::string hex64(uint64_t h);

}  // namespace sssp
