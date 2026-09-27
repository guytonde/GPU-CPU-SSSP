#pragma once

#include <cstdint>
#include <string>

#include "sssp/graph.hpp"

namespace sssp {

// Topology, weights and vertex numbering use separate seeds, so changing the
// weights never changes the edges.
struct GenSpec {
    std::string topology = "uniform";
    int64_t n = 1 << 16;
    int64_t m = 0;                       // undirected edges: uniform, rmat, geometric, hub
    int rows = 0, cols = 0;              // grid; 0 means square from n
    int layer_width = 0;                 // layered
    int layer_links = 4;                 // layered: next-layer targets per vertex
    double hub_fraction = 0.0;           // hub: share of m attached to one vertex
    std::string wdist = "uniform";       // unit | uniform | logunif
    Weight wmin = 1, wmax = 100;
    double decades = -1.0;               // logunif over [1, 10^decades] when >= 0
    uint64_t seed = 1;
    uint64_t wseed = 0;                  // 0 derives it from seed
    uint64_t rseed = 0;                  // 0 derives it from seed
    bool relabel = true;
    bool directed = false;
};

// Topologies: uniform (G(n,m)), rmat (a,b,c = .57,.19,.19), grid (row major),
// geometric (points in the unit square within a radius), chain, star,
// layered (n/W layers of width W, each vertex linked to layer_links vertices of
// the next layer, layer 0 recorded in meta "sources"), and hub (G(n,m) with
// hub_fraction of the edges moved onto vertex 0). None get a spanning tree.
Graph generate(const GenSpec& spec);

}  // namespace sssp
