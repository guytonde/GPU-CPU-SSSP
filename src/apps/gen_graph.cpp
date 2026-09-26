#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "sssp/graph.hpp"

using namespace sssp;

namespace {

struct Options {
    long long n = 10000;
    long long m = 100000;
    std::string topology = "uniform";
    std::string wdist = "uniform";
    Weight wmin = 1;
    Weight wmax = 100;
    uint64_t seed = 12345;
    bool directed = false;
    int connect = -1;  // -1 auto: on only where the topology needs it
    bool dedup_edges = true;
    std::string out = "graphs/out.txt";
};

void usage() {
    std::cout << R"(usage: gen_graph [options]

  --n N            vertices (default 10000)
  --m M            edges, before the undirected doubling (default 100000)
  --topo T         uniform | rmat | grid | geometric | chain | star
                   (default uniform)
  --wdist D        uniform | logunif | unit (default uniform)
  --wmin W         minimum weight (default 1)
  --wmax W         maximum weight (default 100)
  --seed S         rng seed (default 12345)
  --directed       emit each edge once instead of both ways
  --connect        force the spanning tree on
  --no-connect     force it off
  --keep-dups      do not collapse parallel edges
  --out PATH       output file; a .bin suffix writes packed CSR (default graphs/out.txt)

topologies
  uniform    Erdos-Renyi. Uniform degree, small diameter.
  rmat       Graph500 parameters. Heavy degree skew, which separates
             thread-per-vertex from warp-per-vertex.
  grid       2D mesh. n rounds to the nearest square and m is ignored. Large
             hop eccentricity, tiny frontiers.
  geometric  Random points in the unit square joined within a radius picked to
             hit m edges. Roughly road shaped.
  chain      Path graph. m is ignored. Maximal sequential depth: every round
             advances the frontier by one vertex.
  star       One hub joined to everything. m is ignored. Maximal degree skew.

The spanning tree that makes vertex 0 reach everything is added only for the
random topologies. grid, chain and star are connected by construction, and
laying n-1 random edges over one collapses the very property it exists to test.
)";
}

class Weights {
public:
    Weights(const std::string& dist, Weight lo, Weight hi)
        : dist_(dist), lo_(lo), hi_(hi), uniform_(lo, hi),
          log_(std::log(std::max(1.0, double(lo))),
               std::log(std::max(2.0, double(hi)))) {}

    Weight operator()(std::mt19937_64& rng) {
        if (dist_ == "unit") return 1;
        if (dist_ == "logunif") {
            double w = std::exp(log_(rng));
            return std::clamp<Weight>(Weight(std::lround(w)), lo_, hi_);
        }
        return uniform_(rng);
    }

private:
    std::string dist_;
    Weight lo_, hi_;
    std::uniform_int_distribution<Weight> uniform_;
    std::uniform_real_distribution<double> log_;
};

// Random spanning tree rooted at 0, so every vertex is reachable from the
// default source. Without it a sparse random graph leaves a large unreachable
// tail and the timings measure the wrong thing.
void add_spanning_tree(EdgeList& el, long long n, std::mt19937_64& rng,
                       Weights& w) {
    std::vector<int> order(n);
    for (long long i = 0; i < n; ++i) order[i] = int(i);
    std::shuffle(order.begin() + 1, order.end(), rng);

    for (long long i = 1; i < n; ++i) {
        std::uniform_int_distribution<long long> pick(0, i - 1);
        el.add(order[pick(rng)], order[i], w(rng));
    }
}

void gen_uniform(EdgeList& el, long long n, long long m, std::mt19937_64& rng,
                 Weights& w) {
    std::uniform_int_distribution<long long> pick(0, n - 1);
    for (long long i = 0; i < m; ++i) {
        int u = int(pick(rng));
        int v = int(pick(rng));
        if (u == v) continue;
        el.add(u, v, w(rng));
    }
}

// RMAT with the Graph500 quadrant probabilities. Subdividing the adjacency
// matrix with a skewed split is what gives the power-law degrees.
void gen_rmat(EdgeList& el, long long n, long long m, std::mt19937_64& rng,
              Weights& w) {
    const double a = 0.57, b = 0.19, c = 0.19;
    int scale = 1;
    while ((1LL << scale) < n) ++scale;

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (long long i = 0; i < m; ++i) {
        long long u = 0, v = 0;
        for (int bit = 0; bit < scale; ++bit) {
            double r = unit(rng);
            long long step = 1LL << bit;
            if (r < a) {
                // stay
            } else if (r < a + b) {
                v += step;
            } else if (r < a + b + c) {
                u += step;
            } else {
                u += step;
                v += step;
            }
        }
        if (u >= n || v >= n || u == v) continue;
        el.add(int(u), int(v), w(rng));
    }
}

void gen_grid(EdgeList& el, long long& n, std::mt19937_64& rng, Weights& w) {
    long long side = std::max<long long>(2, (long long)std::llround(std::sqrt(double(n))));
    n = side * side;
    auto id = [side](long long r, long long c) { return int(r * side + c); };

    for (long long r = 0; r < side; ++r) {
        for (long long c = 0; c < side; ++c) {
            if (c + 1 < side) el.add(id(r, c), id(r, c + 1), w(rng));
            if (r + 1 < side) el.add(id(r, c), id(r + 1, c), w(rng));
        }
    }
}

void gen_chain(EdgeList& el, long long n, std::mt19937_64& rng, Weights& w) {
    for (long long i = 0; i + 1 < n; ++i) el.add(int(i), int(i + 1), w(rng));
}

void gen_star(EdgeList& el, long long n, std::mt19937_64& rng, Weights& w) {
    for (long long i = 1; i < n; ++i) el.add(0, int(i), w(rng));
}

// Random geometric graph over a grid of cells sized to the connection radius, so
// each point tests 9 cells instead of all n.
void gen_geometric(EdgeList& el, long long n, long long m, std::mt19937_64& rng,
                   Weights& w) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<double> x(n), y(n);
    for (long long i = 0; i < n; ++i) {
        x[i] = unit(rng);
        y[i] = unit(rng);
    }

    // Expected degree 2m/n over area pi*r^2 gives this radius.
    double radius = std::sqrt(double(m) / (3.14159265358979 * double(n) * double(n)) * 2.0);
    radius = std::clamp(radius, 1e-4, 0.5);

    int cells = std::max(1, int(1.0 / radius));
    std::vector<std::vector<int>> grid(size_t(cells) * cells);
    auto cell_of = [&](double p) {
        return std::min(cells - 1, std::max(0, int(p * cells)));
    };
    for (long long i = 0; i < n; ++i) {
        grid[size_t(cell_of(y[i])) * cells + cell_of(x[i])].push_back(int(i));
    }

    double r2 = radius * radius;
    for (long long i = 0; i < n; ++i) {
        int cx = cell_of(x[i]), cy = cell_of(y[i]);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || ny < 0 || nx >= cells || ny >= cells) continue;
                for (int j : grid[size_t(ny) * cells + nx]) {
                    if (j <= int(i)) continue;
                    double ddx = x[i] - x[j], ddy = y[i] - y[j];
                    if (ddx * ddx + ddy * ddy > r2) continue;
                    el.add(int(i), j, w(rng));
                }
            }
        }
    }
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };

        if (a == "-h" || a == "--help") return false;
        else if (a == "--n") o.n = std::stoll(next());
        else if (a == "--m") o.m = std::stoll(next());
        else if (a == "--topo") o.topology = next();
        else if (a == "--wdist") o.wdist = next();
        else if (a == "--wmin") o.wmin = std::stoi(next());
        else if (a == "--wmax") o.wmax = std::stoi(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--directed") o.directed = true;
        else if (a == "--connect") o.connect = 1;
        else if (a == "--no-connect") o.connect = 0;
        else if (a == "--keep-dups") o.dedup_edges = false;
        else if (a == "--out") o.out = next();
        else throw std::runtime_error("unknown option " + a);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        if (!parse(argc, argv, o)) {
            usage();
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        usage();
        return 1;
    }

    if (o.n <= 1 || o.m < 0 || o.wmin < 0 || o.wmax < o.wmin) {
        std::cerr << "error: need n > 1, m >= 0, 0 <= wmin <= wmax\n";
        return 1;
    }

    std::mt19937_64 rng(o.seed);
    Weights w(o.wdist, o.wmin, o.wmax);
    EdgeList el;

    // grid, chain and star are connected already; laying n-1 random edges over
    // one collapses the property it exists to test.
    const bool self_connected = o.topology == "grid" || o.topology == "chain" ||
                                o.topology == "star";
    const bool connect = o.connect < 0 ? !self_connected : o.connect == 1;

    try {
        if (o.topology == "uniform") {
            gen_uniform(el, o.n, o.m, rng, w);
        } else if (o.topology == "rmat") {
            gen_rmat(el, o.n, o.m, rng, w);
        } else if (o.topology == "grid") {
            gen_grid(el, o.n, rng, w);
        } else if (o.topology == "geometric") {
            gen_geometric(el, o.n, o.m, rng, w);
        } else if (o.topology == "chain") {
            gen_chain(el, o.n, rng, w);
        } else if (o.topology == "star") {
            gen_star(el, o.n, rng, w);
        } else {
            std::cerr << "error: unknown topology " << o.topology << "\n";
            return 1;
        }

        if (connect) add_spanning_tree(el, o.n, rng, w);

        if (!o.directed) {
            size_t forward = el.size();
            for (size_t i = 0; i < forward; ++i) el.add(el.v[i], el.u[i], el.w[i]);
        }

        el.n = int(o.n);
        if (o.dedup_edges) dedup(el);

        save_graph(o.out, el);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "wrote " << o.out << "  topo=" << o.topology
              << " n=" << el.n << " arcs=" << el.size()
              << " avg_deg=" << double(el.size()) / double(el.n)
              << " weights=" << o.wdist << "[" << o.wmin << "," << o.wmax
              << "] seed=" << o.seed
              << " spanning_tree=" << (connect ? "yes" : "no") << "\n";
    return 0;
}
