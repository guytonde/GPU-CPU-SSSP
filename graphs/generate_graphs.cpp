#include <iostream>
#include <fstream>
#include <set>
#include <cstdlib>
#include <ctime>
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
        return 1;
    }

    long long max_edges = (long long)n * (n - 1) / 2;
    if (m > max_edges) {
        cerr << "[ERR] Too many edges for the number of vertices.\n";
        return 1;
    }

    set<pair<int, int>> edges;
    srand(time(0));

    while ((int) edges.size() < m) {
        int u = rand() % n;
        int v = rand() % n;
        if (u == v) continue; // no self-loops
        if (edges.count({u, v})) continue; // no duplicate edges
        edges.insert({u, v});
    }

    cerr << "[INFO] Generating graph with " << n << " vertices and " << m << " edges\n";
    
    // Open output file
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "[ERR] Failed to open file: " << filename << "\n";
        return 1;
    }
    
    // Write edges to file
    for (auto e : edges) {
        int w = min_w + rand() % (max_w - min_w + 1);
        outfile << e.first << " " << e.second << " " << w << "\n";
    }
    outfile.close();
    
    cerr << "[DONE] Graph generation complete.\n";
    return 0;
}
