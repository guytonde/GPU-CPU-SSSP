#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sssp {

using Weight = int32_t;

// Half of INT32_MAX so that dist[u] + w never overflows during relaxation.
constexpr Weight kInf = 0x3fffffff;

// Compressed sparse row. Edges of vertex u live at indices
// [offsets[u], offsets[u+1]) of targets/weights.
struct Graph {
    int n = 0;
    std::vector<int> offsets;
    std::vector<int> targets;
    std::vector<Weight> weights;

    int64_t num_edges() const { return static_cast<int64_t>(targets.size()); }
    int degree(int u) const { return offsets[u + 1] - offsets[u]; }

    Weight max_weight() const;
    double avg_degree() const;

    // Row index of every edge. Needed by edge-parallel GPU kernels, which
    // cannot recover the source vertex from the CSR arrays alone.
    std::vector<int> edge_sources() const;
};

// Intermediate form produced by the generators and by the text reader.
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

Graph build_csr(const EdgeList& el);
void dedup(EdgeList& el);

// ".bin" reads and writes the packed CSR format, anything else the text
// edge list ("u v w" per line, "# ..." and blank lines ignored).
Graph load_graph(const std::string& path);
void save_graph(const std::string& path, const EdgeList& el);

EdgeList read_edge_list(const std::string& path);
void write_edge_list(const std::string& path, const EdgeList& el);
Graph read_binary(const std::string& path);
void write_binary(const std::string& path, const Graph& g);

}  // namespace sssp
