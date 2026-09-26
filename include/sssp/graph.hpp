#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sssp {

using Weight = int32_t;

// Half of INT32_MAX so that dist[u] + w never overflows during relaxation.
constexpr Weight kInf = 0x3fffffff;

// compressed sparse row so edges of vertex u live at indices
// [offsets[u], offsets[u+1]) of targets/weights
struct Graph {
    int n = 0;
    std::vector<int> offsets;
    std::vector<int> targets;
    std::vector<Weight> weights;

    int64_t num_edges() const { return static_cast<int64_t>(targets.size()); }
    int degree(int u) const { return offsets[u + 1] - offsets[u]; }

    Weight max_weight() const;
    double avg_degree() const;

    std::vector<int> edge_sources() const;
};

// Intermediate form used by the generators and the text reader
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

// Shape of the graph, independent of any source.
struct GraphStats {
    int n = 0;
    int64_t m = 0;
    double avg_degree = 0.0;
    double degree_stddev = 0.0;
    int max_degree = 0;
    Weight min_weight = 0;
    Weight max_weight = 0;
};

// Shape of the search from one source. `levels` is the hop eccentricity of the
// source, which is a lower bound on the graph diameter and not the same thing.
struct SourceStats {
    int levels = 0;
    int64_t reached = 0;
    int max_frontier = 0;
    double avg_frontier = 0.0;
};

GraphStats graph_stats(const Graph& g);
SourceStats source_stats(const Graph& g, int source);

Graph build_csr(const EdgeList& el);
void dedup(EdgeList& el);

// Every solver here settles vertices in nondecreasing distance order or relies
// on relaxation converging, both of which need w >= 0. Throws on a negative
// weight rather than returning a plausible wrong answer.
void check_weights(const Graph& g);

// the .bin reads and writes the packed CSR format, anything else the text
// edge list ("u v w" per line, w defaulting to 1; "# ..." and blank lines
// ignored).
Graph load_graph(const std::string& path);
void save_graph(const std::string& path, const EdgeList& el);

EdgeList read_edge_list(const std::string& path);
void write_edge_list(const std::string& path, const EdgeList& el);
Graph read_binary(const std::string& path);
void write_binary(const std::string& path, const Graph& g);

}  // namespace sssp
