#include "sssp/generators.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "sssp/build_info.hpp"
#include "sssp/rng.hpp"

namespace sssp {

namespace {

uint64_t key(int64_t u, int64_t v) { return (uint64_t(u) << 32) | uint64_t(v); }
int key_u(uint64_t k) { return int(k >> 32); }
int key_v(uint64_t k) { return int(k & 0xffffffffu); }

class Edges {
public:
    explicit Edges(bool directed) : directed_(directed) {}
    void add(int64_t u, int64_t v) {
        if (u == v) return;
        if (!directed_ && u > v) std::swap(u, v);
        keys.push_back(key(u, v));
    }
    void finish() {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }
    std::vector<uint64_t> keys;

private:
    bool directed_;
};

// Two statements, because argument evaluation order is unspecified.
void random_edge(Edges& e, int64_t n, std::mt19937_64& rng) {
    int64_t u = int64_t(below(rng, n));
    int64_t v = int64_t(below(rng, n));
    e.add(u, v);
}

void gen_uniform(Edges& e, int64_t n, int64_t m, std::mt19937_64& rng) {
    for (int64_t i = 0; i < m; ++i) random_edge(e, n, rng);
}

void gen_rmat(Edges& e, int64_t n, int64_t m, std::mt19937_64& rng) {
    const double a = 0.57, b = 0.19, c = 0.19;
    int scale = 0;
    while ((int64_t(1) << scale) < n) ++scale;
    for (int64_t i = 0; i < m; ++i) {
        int64_t u = 0, v = 0;
        for (int bit = 0; bit < scale; ++bit) {
            double r = unit(rng);
            int64_t step = int64_t(1) << bit;
            if (r < a) {
            } else if (r < a + b) {
                v += step;
            } else if (r < a + b + c) {
                u += step;
            } else {
                u += step;
                v += step;
            }
        }
        if (u < n && v < n) e.add(u, v);
    }
}

void gen_grid(Edges& e, int rows, int cols) {
    auto id = [cols](int64_t r, int64_t c) { return r * cols + c; };
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            if (c + 1 < cols) e.add(id(r, c), id(r, c + 1));
            if (r + 1 < rows) e.add(id(r, c), id(r + 1, c));
        }
    }
}

// Points are binned into cells the size of the radius.
void gen_geometric(Edges& e, int64_t n, int64_t m, std::mt19937_64& rng) {
    std::vector<double> x(n), y(n);
    for (int64_t i = 0; i < n; ++i) {
        x[i] = unit(rng);
        y[i] = unit(rng);
    }
    // About n^2/2 * pi r^2 edges, ignoring the boundary.
    const double pi = 3.14159265358979323846;
    double radius = std::sqrt(2.0 * double(m) / (pi * double(n) * double(n)));
    radius = std::clamp(radius, 1e-5, 0.5);
    int cells = std::max(1, int(1.0 / radius));
    std::vector<std::vector<int>> grid(size_t(cells) * cells);
    auto cell_of = [&](double p) { return std::min(cells - 1, int(p * cells)); };
    for (int64_t i = 0; i < n; ++i) {
        grid[size_t(cell_of(y[i])) * cells + cell_of(x[i])].push_back(int(i));
    }
    double r2 = radius * radius;
    for (int64_t i = 0; i < n; ++i) {
        int cx = cell_of(x[i]), cy = cell_of(y[i]);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || ny < 0 || nx >= cells || ny >= cells) continue;
                for (int j : grid[size_t(ny) * cells + nx]) {
                    if (j <= i) continue;
                    double ddx = x[i] - x[j], ddy = y[i] - y[j];
                    if (ddx * ddx + ddy * ddy <= r2) e.add(i, j);
                }
            }
        }
    }
}

std::vector<int> gen_layered(Edges& e, int64_t n, int width, int links,
                             std::mt19937_64& rng) {
    if (width <= 0) throw std::runtime_error("layered needs --layer-width > 0");
    int64_t layers = (n + width - 1) / width;
    std::vector<int64_t> picks;
    for (int64_t layer = 0; layer + 1 < layers; ++layer) {
        int64_t next = (layer + 1) * width;
        int64_t size = std::min<int64_t>(width, n - next);
        int k = int(std::min<int64_t>(links, size));
        for (int64_t u = layer * width; u < next; ++u) {
            picks.clear();
            while (int(picks.size()) < k) {
                int64_t v = next + int64_t(below(rng, size));
                if (std::find(picks.begin(), picks.end(), v) == picks.end()) {
                    picks.push_back(v);
                }
            }
            for (int64_t v : picks) e.add(u, v);
        }
    }
    // Up to 256 candidate sources from layer 0.
    std::vector<int> layer0(std::min<int64_t>(width, n));
    for (size_t i = 0; i < layer0.size(); ++i) layer0[i] = int(i);
    shuffle(layer0, rng);
    layer0.resize(std::min<size_t>(256, layer0.size()));
    return layer0;
}

void gen_hub(Edges& e, int64_t n, int64_t m, double fraction, std::mt19937_64& rng) {
    int64_t spokes = std::min<int64_t>(n - 1, std::llround(fraction * double(m)));
    std::vector<int> others(n - 1);
    for (int64_t i = 0; i < n - 1; ++i) others[i] = int(i + 1);
    for (int64_t i = 0; i < spokes; ++i) {
        std::swap(others[i], others[i + below(rng, n - 1 - i)]);
        e.add(0, others[i]);
    }
    for (int64_t i = spokes; i < m; ++i) random_edge(e, n, rng);
}

class WeightSampler {
public:
    WeightSampler(const GenSpec& s) : dist_(s.wdist), lo_(s.wmin), hi_(s.wmax) {
        if (dist_ == "logunif" && s.decades >= 0) {
            lo_ = 1;
            hi_ = Weight(std::llround(std::pow(10.0, s.decades)));
        }
        if (dist_ != "unit" && dist_ != "uniform" && dist_ != "logunif") {
            throw std::runtime_error("unknown weight distribution " + dist_);
        }
        if (lo_ < 0 || hi_ < lo_) throw std::runtime_error("need 0 <= wmin <= wmax");
        if (dist_ == "logunif" && lo_ < 1) lo_ = 1;
    }
    Weight operator()(std::mt19937_64& rng) const {
        if (dist_ == "unit") return 1;
        if (dist_ == "uniform") return Weight(lo_ + int64_t(below(rng, uint64_t(hi_) - lo_ + 1)));
        double l = std::log(double(lo_)), h = std::log(double(hi_));
        double w = std::exp(l + (h - l) * unit(rng));
        return std::clamp<Weight>(Weight(std::llround(w)), lo_, hi_);
    }
    Weight lo() const { return lo_; }
    Weight hi() const { return hi_; }

private:
    std::string dist_;
    Weight lo_, hi_;
};

std::string join(const std::vector<int>& v) {
    std::ostringstream out;
    for (size_t i = 0; i < v.size(); ++i) out << (i ? "," : "") << v[i];
    return out.str();
}

std::string num(double x) {
    std::ostringstream out;
    out.precision(10);
    out << x;
    return out.str();
}

}  // namespace

Graph generate(const GenSpec& s) {
    int64_t n = s.n;
    int rows = s.rows, cols = s.cols;
    if (s.topology == "grid") {
        if (rows <= 0 || cols <= 0) {
            rows = std::max(1, int(std::llround(std::sqrt(double(n)))));
            cols = rows;
        }
        n = int64_t(rows) * cols;
    }
    if (n < 1 || n > INT32_MAX) throw std::runtime_error("n out of range");

    std::mt19937_64 rng(s.seed);
    Edges edges(s.directed);
    std::vector<int> sources;

    const std::string& t = s.topology;
    if (t == "uniform") gen_uniform(edges, n, s.m, rng);
    else if (t == "rmat") gen_rmat(edges, n, s.m, rng);
    else if (t == "grid") gen_grid(edges, rows, cols);
    else if (t == "geometric") gen_geometric(edges, n, s.m, rng);
    else if (t == "chain") for (int64_t i = 0; i + 1 < n; ++i) edges.add(i, i + 1);
    else if (t == "star") for (int64_t i = 1; i < n; ++i) edges.add(0, i);
    else if (t == "layered") sources = gen_layered(edges, n, s.layer_width, s.layer_links, rng);
    else if (t == "hub") gen_hub(edges, n, s.m, s.hub_fraction, rng);
    else throw std::runtime_error("unknown topology " + t);
    edges.finish();

    // Weights come from their own stream, in sorted edge order.
    const uint64_t wseed = s.wseed ? s.wseed : s.seed ^ 0x9e3779b97f4a7c15ull;
    const uint64_t rseed = s.rseed ? s.rseed : s.seed ^ 0xc2b2ae3d27d4eb4full;
    WeightSampler sample(s);
    std::mt19937_64 wrng(wseed);
    std::vector<Weight> w(edges.keys.size());
    for (auto& x : w) x = sample(wrng);

    std::vector<int> label(n);
    for (int64_t i = 0; i < n; ++i) label[i] = int(i);
    if (s.relabel) {
        std::mt19937_64 rrng(rseed);
        shuffle(label, rrng);
    }
    for (int& v : sources) v = label[v];

    EdgeList el;
    el.n = int(n);
    for (size_t i = 0; i < edges.keys.size(); ++i) {
        int a = label[key_u(edges.keys[i])], b = label[key_v(edges.keys[i])];
        el.add(a, b, w[i]);
        if (!s.directed) el.add(b, a, w[i]);
    }
    Graph g = build_csr(el);

    Meta& meta = g.meta;
    meta.set("generator", "gen_graph/2");
    meta.set("gen_git_sha", build_git_sha());
    meta.set("gen_git_dirty", build_git_dirty() ? "1" : "0");
    meta.set("topology", t);
    meta.set("n", std::to_string(n));
    meta.set("m_requested", std::to_string(s.m));
    meta.set("m_undirected", std::to_string(edges.keys.size()));
    meta.set("directed", s.directed ? "1" : "0");
    meta.set("seed", std::to_string(s.seed));
    meta.set("wseed", std::to_string(wseed));
    meta.set("rseed", std::to_string(rseed));
    meta.set("relabel", s.relabel ? "1" : "0");
    meta.set("wdist", s.wdist);
    meta.set("wmin", std::to_string(sample.lo()));
    meta.set("wmax", std::to_string(sample.hi()));
    if (s.decades >= 0) meta.set("decades", num(s.decades));
    if (t == "grid") {
        meta.set("rows", std::to_string(rows));
        meta.set("cols", std::to_string(cols));
    }
    if (t == "layered") {
        meta.set("layer_width", std::to_string(s.layer_width));
        meta.set("layer_links", std::to_string(s.layer_links));
        meta.set("sources", join(sources));
    }
    if (t == "hub") meta.set("hub_fraction", num(s.hub_fraction));
    if (t == "rmat") {
        meta.set("rmat_a", "0.57");
        meta.set("rmat_b", "0.19");
        meta.set("rmat_c", "0.19");
    }
    return g;
}

}  // namespace sssp
