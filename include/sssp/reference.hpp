#pragma once

#include <vector>

#include "sssp/graph.hpp"

namespace sssp {

// Textbook O(V*E) Bellman-Ford with no early exit. The EdgeList version never
// touches the CSR, so it also checks build_csr.
std::vector<Weight> reference_sssp(const Graph& g, int source);
std::vector<Weight> reference_sssp(const EdgeList& el, int source);

// Dijkstra ordered by (distance, hops), so it also gives the hop depth of the
// shortest path tree. Used where the oracle is too slow.
struct RefResult {
    std::vector<Weight> dist;
    std::vector<int> hops;  // -1 where unreached
};
RefResult reference_dijkstra(const Graph& g, int source);

}  // namespace sssp
