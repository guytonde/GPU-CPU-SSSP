#include "../inc/algos.h"
#include <omp.h>
#include <queue>
#include <deque>
#include <algorithm>
#include <chrono>

Result DijkstraParallel::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<bool> visited(g.n, false);
    
    dist[source] = 0;
    
    for (int iter = 0; iter < g.n; iter++) {
        int u = -1;
        Weight minDist = INF;
        
        // pick next vertex with min dist but in parallel
        #pragma omp parallel for reduction(min:minDist) private(u)
        for (int v = 0; v < g.n; v++) {
            if (!visited[v] && dist[v] < minDist) {
                minDist = dist[v];
                u = v;
            }
        }
        
        if (minDist == INF) break;
        visited[u] = true;
        
        //edge relaxation, TODO: see if this is right idk i think it is rn
        #pragma omp parallel for
        for (int e = 0; e < (int)g.adj[u].size(); e++) {
            const auto& edge = g.adj[u][e];
            if (dist[u] + edge.weight < dist[edge.to]) {
                #pragma omp critical
                {
                    if (dist[u] + edge.weight < dist[edge.to]) {
                        dist[edge.to] = dist[u] + edge.weight;
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "Dijkstra", "CPU Parallel (OpenMP)"};
}

Result BellmanFordParallel::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
    
    std::vector<Weight> dist(g.n, INF);
    dist[source] = 0;
    
    //  edge relaxation, n-1 times i think
    for (int i = 0; i < g.n - 1; i++) {
        #pragma omp parallel for
        for (int u = 0; u < g.n; u++) {
            if (dist[u] == INF) continue;
            for (const auto& edge : g.adj[u]) {
                if (dist[u] + edge.weight < dist[edge.to]) {
                    #pragma omp critical
                    {
                        if (dist[u] + edge.weight < dist[edge.to]) {
                            dist[edge.to] = dist[u] + edge.weight;
                        }
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "Bellman-Ford", "CPU Parallel (OpenMP)"};
}

Result BFSParallel::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<int> current_level, next_level;
    
    dist[source] = 0;
    current_level.push_back(source);
    
    while (!current_level.empty()) {
        next_level.clear();
        
        #pragma omp parallel for
        for (int idx = 0; idx < (int)current_level.size(); idx++) {
            int u = current_level[idx];
            for (const auto& edge : g.adj[u]) {
                if (dist[edge.to] == INF) {
                    #pragma omp critical
                    {
                        if (dist[edge.to] == INF) {
                            dist[edge.to] = dist[u] + 1;
                            next_level.push_back(edge.to);
                        }
                    }
                }
            }
        }
        
        current_level = next_level;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "BFS", "CPU Parallel (OpenMP)"};
}

Result SPFAParallel::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<int> inQueue(g.n, 0);
    std::vector<std::deque<int>> buckets(4); // bucketing, might be bad tho
    
    dist[source] = 0;
    buckets[0].push_back(source);
    inQueue[source] = 1;
    
    for (int bucket_idx = 0; bucket_idx < 4; bucket_idx++) {
        while (!buckets[bucket_idx].empty()) {
            std::vector<int> batch(buckets[bucket_idx].begin(), buckets[bucket_idx].end());
            buckets[bucket_idx].clear();
            
            #pragma omp parallel for
            for (int idx = 0; idx < (int)batch.size(); idx++) {
                int u = batch[idx];
                inQueue[u] = 0;
                
                for (const auto& edge : g.adj[u]) {
                    if (dist[u] != INF && dist[u] + edge.weight < dist[edge.to]) {
                        #pragma omp critical
                        {
                            if (dist[u] + edge.weight < dist[edge.to]) {
                                dist[edge.to] = dist[u] + edge.weight;
                                if (!inQueue[edge.to]) {
                                    buckets[(dist[edge.to] >> 8) % 4].push_back(edge.to);
                                    inQueue[edge.to] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "SPFA", "CPU Parallel (OpenMP)"};
}

Result LevitParallel::solve(const Graph& g, int source) {
    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
    
    std::vector<Weight> dist(g.n, INF);
    std::vector<int> state(g.n, 0);  // 0: untouched, 1: active, 2: scanned
    std::vector<std::deque<int>> active_set(num_threads);
    
    dist[source] = 0;
    state[source] = 1;
    active_set[0].push_back(source);
    
    bool has_active = true;
    while (has_active) {
        has_active = false;
        
        #pragma omp parallel for
        for (int tid = 0; tid < num_threads; tid++) {
            while (!active_set[tid].empty()) {
                int u = active_set[tid].front();
                active_set[tid].pop_front();
                state[u] = 2;
                
                for (const auto& edge : g.adj[u]) {
                    if (dist[u] + edge.weight < dist[edge.to]) {
                        #pragma omp critical
                        {
                            if (dist[u] + edge.weight < dist[edge.to]) {
                                dist[edge.to] = dist[u] + edge.weight;
                                if (state[edge.to] == 0 || state[edge.to] == 2) {
                                    state[edge.to] = 1;
                                    active_set[tid % num_threads].push_back(edge.to);
                                    has_active = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {dist, elapsed, "Levit", "CPU Parallel (OpenMP)"};
}