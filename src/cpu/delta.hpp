#pragma once

#include <algorithm>

#include "sssp/graph.hpp"

namespace sssp {

// Bucket width for delta-stepping. Buckets that hold roughly one vertex worth
// of out-edges give the parallel variants enough work per phase without letting
// too many out-of-order settles through.
inline int pick_delta(const Graph& g) {
    double deg = std::max(1.0, g.avg_degree());
    Weight maxw = std::max<Weight>(1, g.max_weight());
    return std::max(1, static_cast<int>(maxw / deg));
}

inline int bucket_count(const Graph& g, int delta) {
    return static_cast<int>(std::max<Weight>(1, g.max_weight()) / delta) + 2;
}

}  // namespace sssp
