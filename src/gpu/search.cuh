#pragma once
#include <vector>
#include <limits>
#include <string>
#include <iostream>


// Adjacency list structure for GPU
struct GPUEdge {
    int to;
    int weight;
};

// Device-side data structures
struct GPUGraph {
    int* vertices;      // vertex indices
    int* edges;         // edge destinations
    int* weights;       // edge weights
    int num_vertices;
    int num_edges;
};

// GPU SSSP algorithms
// CSR-based GPU interfaces: `row_ptr` size = num_vertices+1, `col_idx` and `weights` size = num_edges
std::vector<int> gpu_dijkstra(const std::vector<int>& row_ptr,
                               const std::vector<int>& col_idx,
                               const std::vector<int>& weights,
                               int source, int num_vertices);

std::vector<int> gpu_bellman_ford(const std::vector<int>& row_ptr,
                                   const std::vector<int>& col_idx,
                                   const std::vector<int>& weights,
                                   int source, int num_vertices);

std::vector<int> gpu_bfs_sssp(const std::vector<int>& row_ptr,
                              const std::vector<int>& col_idx,
                              const std::vector<int>& weights,
                              int source, int num_vertices);

// Utility functions
// void print_distances(const std::vector<int>& distances);
void print_gpu_results(const std::vector<int>& distances, const std::string& algorithm_name);
