#include "../inc/algos.h"
#include <queue>
#include <cstring>
#include <algorithm>
#include <deque>
#include <chrono>

Result DijkstraSerial::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<bool> visited(g.n, false);
    
    dist[source] = 0;
    
    for (int iter = 0; iter < g.n; iter++) {
        int u = -1;
        for (int v = 0; v < g.n; v++) {
            if (!visited[v] && (u == -1 || dist[v] < dist[u])) {
                u = v;
            }
        }
        
        if (dist[u] == INF) break;
        visited[u] = true;
        
        for (const auto& edge : g.adj[u]) {
            if (dist[u] + edge.weight < dist[edge.to]) {
                dist[edge.to] = dist[u] + edge.weight;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "Dijkstra", "CPU Serial"};
}

Result BellmanFordSerial::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Weight> dist(g.n, INF);
    dist[source] = 0;
    
    // Relax edges n-1 times
    for (int i = 0; i < g.n - 1; i++) {
        for (int u = 0; u < g.n; u++) {
            if (dist[u] == INF) continue;
            for (const auto& edge : g.adj[u]) {
                if (dist[u] + edge.weight < dist[edge.to]) {
                    dist[edge.to] = dist[u] + edge.weight;
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "Bellman-Ford", "CPU Serial"};
}

// Result BFSSerial::solve(const Graph& g, int source) {
//     auto start = std::chrono::high_resolution_clock::now();
    
//     std::vector<Weight> dist(g.n, INF);
//     std::queue<int> q;
    
//     dist[source] = 0;
//     q.push(source);
    
//     while (!q.empty()) {
//         int u = q.front();
//         q.pop();
        
//         for (const auto& edge : g.adj[u]) {
//             if (dist[edge.to] > dist[u] + 1) {
//                 dist[edge.to] = dist[u] + 1;
//                 q.push(edge.to);
//             }
//         }
//     }
    
//     auto end = std::chrono::high_resolution_clock::now();
//     double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
//     return {dist, elapsed, "BFS", "CPU Serial"};
// }
class BFSExpanded : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        Graph expanded = g.expandForBFS(g);
        std::vector<Weight> dist(expanded.n, INF);
        std::queue<int> q;
        dist[source] = 0;
        q.push(source);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (const auto& edge : expanded.adj[u]) {
                if (dist[edge.to] == INF) {
                    dist[edge.to] = dist[u] + 1;
                    q.push(edge.to);
                }
            }
        }
        
        std::vector<Weight> result_dist(g.n);
        for (int i = 0; i < g.n; i++) {
            result_dist[i] = dist[i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        
        return {result_dist, elapsed, "BFS (Expanded)", "CPU Serial"};
    }
    
    std::string getName() override { return "BFS (Expanded)"; }
    std::string getType() override { return "CPU Serial"; }
};

Result SPFASerial::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<int> inQueue(g.n, 0);
    std::deque<int> q;
    
    dist[source] = 0;
    q.push_back(source);
    inQueue[source] = 1;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop_front();
        inQueue[u] = 0;
        
        for (const auto& edge : g.adj[u]) {
            if (dist[u] != INF && dist[u] + edge.weight < dist[edge.to]) {
                dist[edge.to] = dist[u] + edge.weight;
                if (!inQueue[edge.to]) {
                    q.push_back(edge.to);
                    inQueue[edge.to] = 1;
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "SPFA", "CPU Serial"};
}

Result LevitSerial::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<int> state(g.n, 0);  // 0: untouched, 1: active, 2: scanned
    std::deque<int> active;
    
    dist[source] = 0;
    state[source] = 1;
    active.push_back(source);
    
    while (!active.empty()) {
        int u = active.front();
        
        bool process_front = true;
        for (int v = 0; v < g.n; v++) {
            if (state[v] == 1 && dist[v] < dist[u]) {
                u = v;
                process_front = false;
                break;
            }
        }
        
        if (process_front) {
            active.pop_front();
        } else {
            active.erase(std::find(active.begin(), active.end(), u));
            active.push_front(u);
            continue;
        }
        
        state[u] = 2;
        // hopefully works lol
        for (const auto& edge : g.adj[u]) {
            if (dist[u] + edge.weight < dist[edge.to]) {
                dist[edge.to] = dist[u] + edge.weight;
                if (state[edge.to] == 0) {
                    active.push_back(edge.to);
                    state[edge.to] = 1;
                } else if (state[edge.to] == 2) {
                    state[edge.to] = 1;
                    active.push_back(edge.to);
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "Levit", "CPU Serial"};
}