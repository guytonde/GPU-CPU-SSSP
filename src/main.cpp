#include "../inc/algos.h"
#include <iostream>
#include <fstream>
// #include <memory>
#include <iomanip>



bool verifyResults(const std::vector<Weight>& dist1, const std::vector<Weight>& dist2) {
    if (dist1.size() != dist2.size()) return false;
    for (size_t i = 0; i < dist1.size(); i++) {
        if (dist1[i] != dist2[i]) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <graph_file> [source_vertex]\n";
        return 1;
    }

    std::string graph_file = argv[1];
    int source = (argc > 2) ? atoi(argv[2]) : 0;

    try {
        // Load graph
        std::cout << "[INFO] Loading graph from: " << graph_file << "\n";
        Graph g = Graph::loadFromFile(graph_file);
        std::cout << "[INFO] Graph loaded: " << g.n << " vertices, " << g.m << " edges\n";
        
        if (source < 0 || source >= g.n) {
            std::cerr << "[ERR] Invalid source vertex\n";
            return 1;
        }

        // Create solvers
        std::vector<std::unique_ptr<SSSPSolver>> solvers;
        
        // CPU Serial
        solvers.push_back(std::make_unique<DijkstraSerial>());
        solvers.push_back(std::make_unique<BellmanFordSerial>());
        solvers.push_back(std::make_unique<BFSSerial>());
        solvers.push_back(std::make_unique<SPFASerial>());
        solvers.push_back(std::make_unique<LevitSerial>());
        
        // CPU Parallel (8 threads)
        solvers.push_back(std::make_unique<DijkstraParallel>(8));
        solvers.push_back(std::make_unique<BellmanFordParallel>(8));
        solvers.push_back(std::make_unique<BFSParallel>(8));
        solvers.push_back(std::make_unique<SPFAParallel>(8));
        solvers.push_back(std::make_unique<LevitParallel>(8));

        // Benchmark and store results
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "RUNNING BENCHMARKS\n";
        std::cout << std::string(100, '=') << "\n\n";

        std::vector<Result> results;
        Result baseline;
        bool first = true;

        for (auto& solver : solvers) {
            std::cout << "Running " << solver->getType() << " - " << solver->getName() << "... ";
            std::cout.flush();
            
            auto result = solver->solve(g, source);
            results.push_back(result);

            if (first) {
                baseline = result;
                first = false;
            } else {
                if (!verifyResults(baseline.distances, result.distances)) {
                    std::cout << "INCORRECT RESULT!\n";
                    continue;
                }
            }

            std::cout << std::fixed << std::setprecision(3) << result.time_ms << " ms\n";
        }


        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "RESULTS SUMMARY\n";
        std::cout << std::string(100, '=') << "\n\n";

        std::cout << std::left 
                  << std::setw(25) << "Implementation" 
                  << std::setw(20) << "Algorithm"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Speedup vs Serial"
                  << "\n";
        
        std::cout << std::string(75, '-') << "\n";

        double serial_time = results[0].time_ms;

        for (const auto& result : results) {
            double speedup = serial_time / result.time_ms;
            std::cout << std::left
                      << std::setw(25) << result.implementation_type
                      << std::setw(20) << result.algorithm_name
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x"
                      << "\n";
        }

        std::cout << "\n[INFO] Baseline (reference): " << results[0].implementation_type 
                  << " - " << results[0].algorithm_name << " = " 
                  << std::fixed << std::setprecision(3) << results[0].time_ms << " ms\n";

        // Sample distances for verification
        std::cout << "\n[INFO] Sample distances from source " << source << ":\n";
        for (int i = 0; i < std::min(10, g.n); i++) {
            if (baseline.distances[i] == INF) {
                std::cout << "  dist[" << i << "] = INF\n";
            } else {
                std::cout << "  dist[" << i << "] = " << baseline.distances[i] << "\n";
            }
        }

        std::cout << "\n[INFO] Benchmark complete.\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "[DONE] we done yippee!\n";
    return 0;
}