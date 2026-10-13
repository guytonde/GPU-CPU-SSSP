#pragma once

#include <cstdint>
#include <vector>

#include "sssp/graph.hpp"
#include "sssp/reference.hpp"

namespace sssp {

struct SourceStats {
    int source = 0;
    int64_t reached = 0;
    int64_t m_reached = 0;            // edges leaving reached vertices
    int bfs_levels = 0;
    double bfs_frontier_wmean = 0.0;  // sum f^2 / sum f over BFS levels
    // Hop depth of the shortest path tree, ties broken toward fewer hops.
    int sp_depth_max = 0;
};

// Component id per vertex, numbered by size so the largest is 0.
std::vector<int> component_ids(const Graph& g, int* count = nullptr);

SourceStats source_stats(const Graph& g, int source, const RefResult& ref);

}  // namespace sssp
