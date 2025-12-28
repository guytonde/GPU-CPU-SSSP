#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <functional>
#include <fstream>
#include <sstream>
#include "search.hpp"
#include "../common/utils.hpp"

using namespace std;

vector<int> dijkstra(const Graph& graph, int source) {
    int n = graph.size();
    vector<int> dist(n, INF);

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    dist[source] = 0;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (d > dist[u]) continue;
        
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            int weight = edge.weight;
            
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}

vector<int> bellman_ford(const Graph& graph, int source) {
    int n = graph.size();
    vector<int> dist(n, INF);
    dist[source] = 0;
    
    // Relax edges n-1 times
    for (int i = 0; i < n - 1; i++) {
        for (int u = 0; u < n; u++) {
            if (dist[u] == INF) continue;
            for (const Edge& edge : graph[u]) {
                int v = edge.to;
                int weight = edge.weight;
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                }
            }
        }
    }
    
    // Check for negative cycles
    for (int u = 0; u < n; u++) {
        if (dist[u] == INF) continue;
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            int weight = edge.weight;
            if (dist[u] + weight < dist[v]) {
                // Negative cycle detected
                fill(dist.begin(), dist.end(), -INF);
                return dist;
            }
        }
    }
    
    return dist;
}

vector<int> bfs_sssp(const Graph& graph, int source) {
    int n = graph.size();
    vector<int> dist(n, INF);
    queue<int> q;
    
    dist[source] = 0;
    q.push(source);
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            // BFS assumes unweighted graph or all weights = 1
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    
    return dist;
}

vector<vector<int>> johnson(const Graph& graph) {
    int n = graph.size();
    
    // Step 1: Add a new vertex (vertex n) connected to all vertices with weight 0
    Graph extended_graph = graph;
    extended_graph.resize(n + 1);
    for (int i = 0; i < n; i++) {
        extended_graph[n].push_back({i, 0});
    }
    
    // Step 2: Run Bellman-Ford from the new vertex
    vector<int> h = bellman_ford(extended_graph, n);
    
    // Check for negative cycles
    if (!h.empty() && h[0] == -INF) {
        // Negative cycle detected, return empty matrix
        return vector<vector<int>>();
    }
    
    // Step 3: Reweight all edges
    Graph reweighted_graph(n);
    for (int u = 0; u < n; u++) {
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            int new_weight = edge.weight + h[u] - h[v];
            reweighted_graph[u].push_back({v, new_weight});
        }
    }
    
    // Step 4: Run Dijkstra from each vertex on reweighted graph
    vector<vector<int>> result(n);
    for (int i = 0; i < n; i++) {
        vector<int> dist = dijkstra(reweighted_graph, i);
        
        // Step 5: Restore original weights
        for (int j = 0; j < n; j++) {
            if (dist[j] != INF) {
                dist[j] = dist[j] - h[i] + h[j];
            }
        }
        result[i] = dist;
    }
    
    return result;
}

vector<int> astar(const Graph& graph, int source, int target, const vector<int>& heuristic) {
    int n = graph.size();
    vector<int> dist(n, INF);
    vector<int> parent(n, -1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    dist[source] = 0;
    pq.push({heuristic[source], source});
    
    while (!pq.empty()) {
        int f = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (u == target) {
            // Reconstruct path
            vector<int> path;
            int current = target;
            while (current != -1) {
                path.push_back(current);
                current = parent[current];
            }
            reverse(path.begin(), path.end());
            return path;
        }
        
        int g = f - heuristic[u];
        if (g > dist[u]) continue;
        
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            int weight = edge.weight;
            
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                int f_new = dist[v] + heuristic[v];
                pq.push({f_new, v});
            }
        }
    }
    
    // No path found
    return vector<int>();
}

/**** Utility Functions ****/


bool has_negative_cycle(const Graph& graph, int source) {
    vector<int> dist = bellman_ford(graph, source);
    return !dist.empty() && dist[0] == -INF;
}

bool has_negative_weight(const Graph& graph) {
    for (const auto& adj_list : graph) {
        for (const Edge& edge : adj_list) {
            if (edge.weight < 0) {
                return true;
            }
        }
    }
    return false;
}

Graph load_graph_from_file(const string& filename, int num_vertices) {
    Graph graph(num_vertices);
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        istringstream iss(line);
        int u, v, weight;
        if (iss >> u >> v >> weight) {
            graph[u].push_back({v, weight});
        }
    }
    
    return graph;
}

void print_graph(const Graph& graph) {
    for (size_t u = 0; u < graph.size(); u++) {
        cout << "Vertex " << u << ": ";
        for (const Edge& edge : graph[u]) {
            cout << "(" << edge.to << ", " << edge.weight << ") ";
        }
        cout << endl;
    }
}

void print_all_pairs_distances(const vector<vector<int>>& distances) {
    size_t n = distances.size();
    cout << "All-pairs shortest distances:" << endl;
    cout << "   ";
    for (size_t j = 0; j < n; j++) {
        cout << setw(8) << j;
    }
    cout << endl;
    
    for (size_t i = 0; i < n; i++) {
        cout << setw(2) << i << ":";
        for (size_t j = 0; j < n; j++) {
            if (distances[i][j] == INF) {
                cout << setw(8) << "INF";
            } else if (distances[i][j] == -INF) {
                cout << setw(8) << "-INF";
            } else {
                cout << setw(8) << distances[i][j];
            }
        }
        cout << endl;
    }
}

void print_path(const vector<int>& path) {
    if (path.empty()) {
        cout << "No path found" << endl;
        return;
    }
    
    cout << "Path: ";
    for (size_t i = 0; i < path.size(); i++) {
        cout << path[i];
        if (i < path.size() - 1) {
            cout << " -> ";
        }
    }
    cout << endl;
}

bool is_connected(const Graph& graph) {
    if (graph.empty()) return true;
    
    size_t n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    
    q.push(0);
    visited[0] = true;
    size_t count = 1;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
                count++;
            }
        }
    }
    
    return count == n;
}

// print_distances and print_results_info are provided by ../common/utils.hpp