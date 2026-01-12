#include <iostream>
#include <fstream>
#include <set>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cout << "Usage: ./gen_graph <vertices> <edges> <min_weight> <max_weight> <output_file>\n";
        return 1;
    }

    long long n = atoll(argv[1]); // vertices
    long long m = atoll(argv[2]); // edges
    int min_w = atoi(argv[3]);
    int max_w = atoi(argv[4]);
    string filename = argv[5];

    if (n <= 0 || m < 0 || min_w > max_w) {
        cerr << "[ERR] Invalid input parameters.\n";
        cerr << "  n must be > 0\n";
        cerr << "  m must be >= 0\n";
        cerr << "  min_weight must be <= max_weight\n";
        return 1;
    }

    long long max_edges = (long long)n * (n - 1) / 2;
    if (m > max_edges) {
        cerr << "[ERR] Too many edges for the number of vertices.\n";
        cerr << "  Max edges for " << n << " vertices: " << max_edges << "\n";
        cerr << "  Requested: " << m << "\n";
        return 1;
    }

    mt19937_64 rng(std::random_device{}());
    uniform_int_distribution<long long> vertex_dist(0, n - 1);
    uniform_int_distribution<int> weight_dist(min_w, max_w);

    set<pair<int, int>> edges;
    
    cerr << "[INFO] Generating graph with " << n << " vertices and " << m << " edges\n";
    cerr << "[INFO] Weight range: [" << min_w << ", " << max_w << "]\n";

    long long attempts = 0;
    long long max_attempts = m * 10;  // Safety limit
    
    while ((long long)edges.size() < m && attempts < max_attempts) {
        int u = vertex_dist(rng);
        int v = vertex_dist(rng);
        attempts++;

        if (u == v) continue;  // no self-loops
        if (u > v) swap(u, v);  // normalize to avoid duplicates
        
        if (edges.count({u, v})) continue;  // no duplicate edges
        
        edges.insert({u, v});
    }

    if ((long long)edges.size() < m) {
        cerr << "[WARN] Could only generate " << edges.size() << " edges out of " 
             << m << " requested\n";
        m = edges.size();
    }

    // Open output file
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "[ERR] Failed to open file: " << filename << "\n";
        return 1;
    }

    outfile << n << " " << m << "\n";
    
    // Write edges to file
    for (auto e : edges) {
        int w = weight_dist(rng);
        outfile << e.first << " " << e.second << " " << w << "\n";
    }
    outfile.close();
    
    cerr << "[DONE] Graph generation complete.\n";
    cerr << "  File: " << filename << "\n";
    cerr << "  Vertices: " << n << "\n";
    cerr << "  Edges: " << edges.size() << "\n";
    cerr << "  Density: " << (double)edges.size() * 2 / (double)(n * (n - 1)) * 100 << "%\n";
    
    return 0;
}