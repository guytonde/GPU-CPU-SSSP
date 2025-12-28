#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <iomanip>

using namespace std;

inline constexpr int INF = numeric_limits<int>::max();
inline constexpr int NEG_INF = -INF;

inline string dist_to_string(int d) {
    if (d == INF) return "INF";
    if (d == NEG_INF) return "-INF";
    return to_string(d);
}

inline void print_distances(const vector<int>& distances) {
    for (size_t i = 0; i < distances.size(); ++i) {
        if (distances[i] == INF) {
            cout << "Vertex " << i << ": INF\n";
        } else if (distances[i] == NEG_INF) {
            cout << "Vertex " << i << ": -INF (negative cycle)\n";
        } else {
            cout << "Vertex " << i << ": " << distances[i] << "\n";
        }
    }
}

inline void print_results_info(const vector<int>& cpu,
                               const vector<int>& gpu,
                               const string& algorithm_name,
                               size_t max_diffs_to_show) {
    cout << "===== Results: " << algorithm_name << " =====\n";
    if (cpu.size() != gpu.size()) {
        cout << "Size mismatch: cpu=" << cpu.size() << " gpu=" << gpu.size() << "\n";
    }
    size_t n = min(cpu.size(), gpu.size());
    size_t diffs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (cpu[i] != gpu[i]) {
            ++diffs;
            if (diffs <= max_diffs_to_show) {
                cout << "Vertex " << i << ": cpu=" << dist_to_string(cpu[i])
                          << " gpu=" << dist_to_string(gpu[i]) << "\n";
            }
        }
    }
    if (cpu.size() != gpu.size()) diffs += (cpu.size() > n ? cpu.size()-n : gpu.size()-n);
    cout << "Total differing vertices: " << diffs << "\n";
    if (diffs == 0) {
        cout << "[MATCH] CPU and GPU outputs identical\n";
    } else {
        cout << "[DIFFER] CPU and GPU outputs differ\n";
    }
    cout << "=================================\n";
}

// Convert adjacency-list Graph (vector<vector<Edge>>) into CSR arrays
// Caller provides row_ptr (size n+1), col_idx (size m), weights (size m)
template <typename EdgeType>
inline void graph_to_csr(const vector<vector<EdgeType>>& graph,
                         vector<int>& row_ptr,
                         vector<int>& col_idx,
                         vector<int>& weights) {
    int n = (int)graph.size();
    row_ptr.assign(n + 1, 0);
    for (int u = 0; u < n; ++u) row_ptr[u+1] = row_ptr[u] + (int)graph[u].size();
    size_t m = row_ptr[n];
    col_idx.clear(); col_idx.reserve(m);
    weights.clear(); weights.reserve(m);
    for (int u = 0; u < n; ++u) {
        for (const EdgeType& e : graph[u]) {
            col_idx.push_back(e.to);
            weights.push_back(e.weight);
        }
    }
}
