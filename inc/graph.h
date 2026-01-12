#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <tuple>
#include <limits>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>

using namespace std;

typedef long long ll;
typedef int Weight;
const Weight INF = numeric_limits<Weight>::max() / 2;

// adj list
struct Edge {
    int to;
    Weight weight;
};

struct Graph {
    int n;  // num verts
    ll m;   // num edges
    vector<vector<Edge>> adj;

    // Graph() : n(0), m(0) {}

    Graph(int vertices) : n(vertices), m(0), adj(vertices) {}

    void addEdge(int from, int to, Weight weight) {
        adj[from].push_back({to, weight});
        m++;
    }

    // load graph (u v w format)
    static Graph loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("[ERR] Cannot open file: " + filename);
        }

        int maxVertex = -1;
        vector<tuple<int, int, Weight>> edges;
        
        string line;
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            istringstream iss(line);
            int u, v;
            Weight w;
            if (iss >> u >> v >> w) {
                edges.emplace_back(u, v, w);
                maxVertex = max(maxVertex, max(u, v));
            }
        }
        file.close();

        Graph g(maxVertex + 1);
        for (auto& [u, v, w] : edges) {
            g.addEdge(u, v, w);
        }
        return g;
    }

    Graph Graph::expandForBFS() const {
        Graph expanded(n);
        expanded.adj.assign(n, {});  // start with original size
        expanded.m = 0;

        int next = n;

        for (int u = 0; u < n; ++u) {
            for (auto &e : adj[u]) {
                int v = e.to;
                int w = e.weight;
                if (w == 1) {
                    expanded.addEdge(u, v, 1);
                } else {
                    int cur = u;
                    for (int i = 0; i < w - 1; ++i) {
                        int dummy = next++;
                        if ((int)expanded.adj.size() <= dummy)
                            expanded.adj.push_back({});
                        expanded.addEdge(cur, dummy, 1);
                        cur = dummy;
                    }
                    expanded.addEdge(cur, v, 1);
                }
            }
        }

        expanded.n = next;
        return expanded;
    }


    // apparently Compressed Sparse Row format is better for GPU
    struct CSRFormat {
        vector<int> rowPtr;
        vector<int> colIdx;
        vector<Weight> values;
        int n, nnz;

        CSRFormat(const Graph& g) : n(g.n), nnz(g.m) {
            rowPtr.resize(n + 1, 0);
            colIdx.resize(nnz);
            values.resize(nnz);

            for (int u = 0; u < n; u++) {
                rowPtr[u + 1] = rowPtr[u] + g.adj[u].size();
            }

            int idx = 0;
            for (int u = 0; u < n; u++) {
                for (const auto& edge : g.adj[u]) {
                    colIdx[idx] = edge.to;
                    values[idx] = edge.weight;
                    idx++;
                }
            }
        }
    };

    CSRFormat toCSR() const {
        return CSRFormat(*this);
    }
};

#endif // GRAPH_H