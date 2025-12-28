#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include "cpu/search.hpp"

#ifdef USE_GPU
#include "gpu/search.cuh"
#endif

using namespace std;

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " <algorithm> <graph_file> <num_vertices> [source_vertex]\n";
    cout << "Algorithms: dijkstra, bellman-ford, bfs, johnson, gpu-dijkstra, gpu-bellman-ford, gpu-bfs\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    string algorithm = argv[1];
    string graph_file = argv[2];
    int num_vertices = stoi(argv[3]);
    int source = (argc > 4) ? stoi(argv[4]) : 0;
    
    // Load graph
    Graph graph = load_graph_from_file(graph_file, num_vertices);
    
    if (graph.empty()) {
        cerr << "Error loading graph from file\n";
        return 1;
    }
    
    cout << "Loaded graph with " << num_vertices << " vertices\n";
    cout << "Running " << algorithm << " from source " << source << "\n";
    cout << "================================\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    if (algorithm == "dijkstra") {
        auto distances = dijkstra(graph, source);
        auto end = chrono::high_resolution_clock::now();
        cout << "Dijkstra's Algorithm (CPU)\n";
        print_distances(distances);
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
    else if (algorithm == "bellman-ford") {
        auto distances = bellman_ford(graph, source);
        auto end = chrono::high_resolution_clock::now();
        cout << "Bellman-Ford Algorithm (CPU)\n";
        print_distances(distances);
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
    else if (algorithm == "bfs") {
        auto distances = bfs_sssp(graph, source);
        auto end = chrono::high_resolution_clock::now();
        cout << "BFS-based SSSP (CPU)\n";
        print_distances(distances);
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
    else if (algorithm == "johnson") {
        auto distances = johnson(graph);
        auto end = chrono::high_resolution_clock::now();
        cout << "Johnson's Algorithm (CPU)\n";
        if (!distances.empty()) {
            print_all_pairs_distances(distances);
        } else {
            cout << "Negative cycle detected - womp womp\n";
        }
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
#ifdef USE_GPU
    else if (algorithm == "gpu-dijkstra") {
        // Convert Graph to CSR for GPU
        std::vector<int> row_ptr, col_idx, weights_flat;
        graph_to_csr(graph, row_ptr, col_idx, weights_flat);
        auto distances = gpu_dijkstra(row_ptr, col_idx, weights_flat, source, num_vertices);
        auto end = chrono::high_resolution_clock::now();
        print_gpu_results(distances, "Dijkstra's Algorithm");
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
    else if (algorithm == "gpu-bellman-ford") {
        std::vector<int> row_ptr, col_idx, weights_flat;
        graph_to_csr(graph, row_ptr, col_idx, weights_flat);
        auto distances = gpu_bellman_ford(row_ptr, col_idx, weights_flat, source, num_vertices);
        auto end = chrono::high_resolution_clock::now();
        print_gpu_results(distances, "Bellman-Ford Algorithm");
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
    else if (algorithm == "gpu-bfs") {
        std::vector<int> row_ptr, col_idx, weights_flat;
        graph_to_csr(graph, row_ptr, col_idx, weights_flat);
        auto distances = gpu_bfs_sssp(row_ptr, col_idx, weights_flat, source, num_vertices);
        auto end = chrono::high_resolution_clock::now();
        print_gpu_results(distances, "BFS-based SSSP");
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "Time: " << duration.count() << " microseconds\n";
    }
#endif
    else {
        cerr << "Unknown algorithm: " << algorithm << "\n";
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
