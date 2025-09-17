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

    int n = atoi(argv[1]); // vertices
    int m = atoi(argv[2]); // edges
    int min_w = atoi(argv[3]);
    int max_w = atoi(argv[4]);
    string filename = argv[5];

    if (n <= 0 || m < 0 || min_w > max_w) {
        cerr << "Invalid input parameters.\n";
        return 1;
    }

    if (m > n * (n - 1) / 2) {
        cerr << "Too many edges for the number of vertices.\n";
        return 1;
    }

    set<pair<int, int>> edges;
    srand(time(0));

    while (edges.size() < m) {
        int u = rand() % n;
        int v = rand() % n;
        if (u == v) continue; // no self-loops
        if (edges.count({u, v})) continue; // no duplicate edges
        edges.insert({u, v});
    }

    cerr << "Generating graph with " << n << " vertices and " << m << " edges" << endl;
    for (auto e : edges) {
        int w = min_w + rand() % (max_w - min_w + 1);
        cout << e.first << " " << e.second << " " << w << endl;
    }
    cerr << "Graph generation complete." << endl;
    return 0;
}
