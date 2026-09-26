#pragma once

#include <vector>

#include "sssp/graph.hpp"

namespace sssp {

// Textbook O(V*E) Bellman-Ford, no early exit and no data structure worth
// getting wrong. Slow on purpose: it exists to be the independent answer the
// real solvers are checked against, not to compete with them. The EdgeList
// overload never touches CSR, so it also catches build_csr bugs.
std::vector<Weight> reference_sssp(const Graph& g, int source);
std::vector<Weight> reference_sssp(const EdgeList& el, int source);

}  // namespace sssp
