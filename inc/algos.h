#ifndef ALGOS_H
#define ALGOS_H

#include "graph.h"
#include <chrono>

using namespace std;

struct Result {
    vector<Weight> distances;
    double time_ms;
    string algorithm_name;
    string implementation_type;
};

// a basic interface for all the algos
class SSSPSolver {
public:
    virtual ~SSSPSolver() = default;
    virtual Result solve(const Graph& g, int source) = 0;
    virtual string getName() = 0;
    virtual string getType() = 0;
};

///////////////////////////////////////////////////////
// CPU serial

class DijkstraSerial : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Dijkstra"; }
    string getType() override { return "CPU Serial"; }
};

class BellmanFordSerial : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Bellman-Ford"; }
    string getType() override { return "CPU Serial"; }
};

class BFSSerial : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "BFS"; }
    string getType() override { return "CPU Serial"; }
};

class SPFASerial : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "SPFA"; }
    string getType() override { return "CPU Serial"; }
};

class LevitSerial : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Levit"; }
    string getType() override { return "CPU Serial"; }
};

///////////////////////////////////////////////////////
// CPU parallel (using openmp)

class DijkstraParallel : public SSSPSolver {
private:
    int num_threads;
public:
    DijkstraParallel(int threads = 8) : num_threads(threads) {}
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Dijkstra"; }
    string getType() override { return "CPU Parallel (OpenMP)"; }
};

class BellmanFordParallel : public SSSPSolver {
private:
    int num_threads;
public:
    BellmanFordParallel(int threads = 8) : num_threads(threads) {}
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Bellman-Ford"; }
    string getType() override { return "CPU Parallel (OpenMP)"; }
};

class BFSParallel : public SSSPSolver {
private:
    int num_threads;
public:
    BFSParallel(int threads = 8) : num_threads(threads) {}
    Result solve(const Graph& g, int source) override;
    string getName() override { return "BFS"; }
    string getType() override { return "CPU Parallel (OpenMP)"; }
};

class SPFAParallel : public SSSPSolver {
private:
    int num_threads;
public:
    SPFAParallel(int threads = 8) : num_threads(threads) {}
    Result solve(const Graph& g, int source) override;
    string getName() override { return "SPFA"; }
    string getType() override { return "CPU Parallel (OpenMP)"; }
};

class LevitParallel : public SSSPSolver {
private:
    int num_threads;
public:
    LevitParallel(int threads = 8) : num_threads(threads) {}
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Levit"; }
    string getType() override { return "CPU Parallel (OpenMP)"; }
};

////////////////////////////////////////////////////////////////
// GPU

#ifdef __CUDACC__

// naive
class DijkstraNaiveGPU : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Dijkstra"; }
    string getType() override { return "GPU Naive"; }
};

class BellmanFordNaiveGPU : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Bellman-Ford"; }
    string getType() override { return "GPU Naive"; }
};

class BFSNaiveGPU : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "BFS"; }
    string getType() override { return "GPU Naive"; }
};

//  Optimal
class DijkstraOptimalGPU : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Dijkstra"; }
    string getType() override { return "GPU Optimal"; }
};

class BellmanFordOptimalGPU : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "Bellman-Ford"; }
    string getType() override { return "GPU Optimal"; }
};

class BFSOptimalGPU : public SSSPSolver {
public:
    Result solve(const Graph& g, int source) override;
    string getName() override { return "BFS"; }
    string getType() override { return "GPU Optimal"; }
};

#endif // __CUDACC__

#endif // ALGOS_H