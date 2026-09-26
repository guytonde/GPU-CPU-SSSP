#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "sssp/graph.hpp"
#include "sssp/reference.hpp"
#include "sssp/registry.hpp"
#include "sssp/solvers.hpp"

using namespace sssp;

namespace {

int g_checks = 0;
int g_failures = 0;
std::string g_case;

void fail(const std::string& what) {
    ++g_failures;
    if (g_failures < 25) std::printf("  FAIL %s: %s\n", g_case.c_str(), what.c_str());
}

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) fail(what);
}

void check_eq(int64_t got, int64_t want, const std::string& what) {
    check(got == want,
          what + " got " + std::to_string(got) + " want " + std::to_string(want));
}

bool gpu_available() {
    static const bool yes = gpu_device_count() > 0;
    return yes;
}

// Compares every registered solver against `want`. `delta` of 0 leaves the
// bucket width on auto.
void compare_all(const Graph& g, int source, const std::vector<Weight>& want,
                 int delta = 0) {
    for (const auto& entry : solver_table()) {
        if (entry.device == "gpu" && !gpu_available()) continue;

        auto solver = entry.make();
        if (delta > 0) solver->set_delta(delta);
        Run r = solver->run(g, source);

        if (r.dist.size() != want.size()) {
            fail(entry.name + " returned " +
                 std::to_string(r.dist.size()) + " distances");
            continue;
        }
        int bad = 0;
        int first = -1;
        for (size_t i = 0; i < want.size(); ++i) {
            if (r.dist[i] != want[i]) {
                if (first < 0) first = int(i);
                ++bad;
            }
        }
        ++g_checks;
        if (bad) {
            ++g_failures;
            if (g_failures < 25) {
                std::printf("  FAIL %s: %s wrong on %d of %zu, first v=%d "
                            "got %d want %d\n",
                            g_case.c_str(), entry.name.c_str(), bad, want.size(), first,
                            r.dist[first], want[first]);
            }
        }
    }
}

void compare_all(const EdgeList& el, int source, int delta = 0) {
    compare_all(build_csr(el), source, reference_sssp(el, source), delta);
}

EdgeList random_graph(std::mt19937_64& rng, int n, int m, Weight lo, Weight hi,
                      bool undirected) {
    EdgeList el;
    el.n = n;
    std::uniform_int_distribution<int> vertex(0, n - 1);
    std::uniform_int_distribution<Weight> weight(lo, hi);
    for (int i = 0; i < m; ++i) {
        int u = vertex(rng), v = vertex(rng);
        Weight w = weight(rng);
        el.add(u, v, w);
        if (undirected) el.add(v, u, w);
    }
    return el;
}

EdgeList chain(int n, Weight w) {
    EdgeList el;
    el.n = n;
    for (int i = 0; i + 1 < n; ++i) {
        el.add(i, i + 1, w);
        el.add(i + 1, i, w);
    }
    return el;
}

EdgeList star(int n, Weight w) {
    EdgeList el;
    el.n = n;
    for (int i = 1; i < n; ++i) {
        el.add(0, i, w * i);
        el.add(i, 0, w * i);
    }
    return el;
}

EdgeList mesh(int side, Weight w) {
    EdgeList el;
    el.n = side * side;
    auto id = [side](int r, int c) { return r * side + c; };
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            if (c + 1 < side) {
                el.add(id(r, c), id(r, c + 1), w);
                el.add(id(r, c + 1), id(r, c), w);
            }
            if (r + 1 < side) {
                el.add(id(r, c), id(r + 1, c), w);
                el.add(id(r + 1, c), id(r, c), w);
            }
        }
    }
    return el;
}

void test_random_small() {
    g_case = "random_small";
    const Weight ranges[][2] = {{1, 1}, {0, 3}, {1, 10}, {1, 1000}, {1, 100000}};

    for (int seed = 0; seed < 300; ++seed) {
        std::mt19937_64 rng(seed);
        int n = 2 + int(rng() % 63);
        int m = int(rng() % (4 * n + 1));
        const Weight* r = ranges[rng() % 5];
        EdgeList el = random_graph(rng, n, m, r[0], r[1], rng() % 2 == 0);
        compare_all(el, int(rng() % n));
    }
}

void test_forced_delta() {
    g_case = "forced_delta";
    for (int seed = 0; seed < 40; ++seed) {
        std::mt19937_64 rng(seed + 9000);
        EdgeList el = random_graph(rng, 40, 120, 1, 5000, true);
        for (int delta : {1, 2, 3, 17, 4999, 5000, 100000}) {
            compare_all(el, 0, delta);
        }
    }
}

void test_shapes() {
    g_case = "chain";
    compare_all(chain(200, 1), 0);
    compare_all(chain(200, 7), 100);

    g_case = "star";
    compare_all(star(300, 3), 0);
    compare_all(star(300, 3), 150);

    g_case = "mesh";
    compare_all(mesh(20, 1), 0);
    compare_all(mesh(20, 5), 210);

    g_case = "single_vertex";
    EdgeList one;
    one.n = 1;
    compare_all(one, 0);

    g_case = "disconnected";
    EdgeList split;
    split.n = 10;
    split.add(0, 1, 5);
    split.add(1, 2, 5);
    compare_all(split, 0);
    compare_all(split, 9);

    g_case = "self_and_parallel_edges";
    EdgeList dup;
    dup.n = 5;
    dup.add(0, 0, 3);
    dup.add(0, 1, 9);
    dup.add(0, 1, 2);
    dup.add(1, 2, 4);
    dup.add(2, 2, 0);
    dup.add(2, 3, 1);
    compare_all(dup, 0);

    g_case = "zero_weights";
    EdgeList zero;
    zero.n = 50;
    for (int i = 0; i + 1 < 50; ++i) zero.add(i, i + 1, 0);
    zero.add(49, 0, 0);
    compare_all(zero, 25);
}

// Large enough that delta-omp takes its parallel path: every relaxation lands
// on one of a few vertices, so all threads CAS the same words.
void test_contention() {
    g_case = "contention_funnel";
    const int hubs = 8;
    const int fan = 30000;

    EdgeList el;
    el.n = 1 + fan + hubs;
    std::mt19937_64 rng(777);
    std::uniform_int_distribution<Weight> weight(1, 4);
    for (int i = 0; i < fan; ++i) {
        el.add(0, 1 + i, weight(rng));
        for (int h = 0; h < hubs; ++h) {
            el.add(1 + i, 1 + fan + h, weight(rng));
        }
    }
    for (int h = 0; h < hubs; ++h) el.add(1 + fan + h, 0, 1);

    Graph g = build_csr(el);
    // Too big for the O(V*E) oracle; dijkstra is the reference here, and the
    // small cases above are what establish dijkstra.
    std::vector<Weight> want = make_dijkstra_heap()->run(g, 0).dist;
    for (int i = 0; i < 5; ++i) compare_all(g, 0, want, 1);

    g_case = "contention_competing_paths";
    EdgeList race;
    const int n = 50000;
    race.n = n;
    std::mt19937_64 rng2(778);
    std::uniform_int_distribution<Weight> w2(1, 6);
    for (int i = 0; i < n; ++i) {
        for (int k : {1, 7, 13, 101, 997}) {
            race.add(i, (i * k + 1) % n, w2(rng2));
        }
    }
    Graph rg = build_csr(race);
    std::vector<Weight> ref = make_dijkstra_heap()->run(rg, 0).dist;
    for (int delta : {1, 2, 5}) {
        for (int i = 0; i < 3; ++i) compare_all(rg, 0, ref, delta);
    }
}

void test_csr() {
    g_case = "csr";
    EdgeList el;
    el.n = 4;
    el.add(2, 3, 5);
    el.add(0, 1, 7);
    el.add(0, 3, 9);

    Graph g = build_csr(el);
    check_eq(g.num_edges(), 3, "edges");
    check_eq(g.degree(0), 2, "degree(0)");
    check_eq(g.degree(1), 0, "degree(1)");
    check_eq(g.degree(2), 1, "degree(2)");
    check_eq(g.offsets[4], 3, "offsets tail");
    check_eq(g.max_weight(), 9, "max_weight");

    g_case = "csr_out_of_range";
    EdgeList bad;
    bad.n = 2;
    bad.add(0, 5, 1);
    bool threw = false;
    try {
        build_csr(bad);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "out of range endpoint should throw");

    g_case = "dedup";
    EdgeList dups;
    dups.n = 3;
    dups.add(0, 1, 9);
    dups.add(0, 1, 2);
    dups.add(1, 2, 4);
    dedup(dups);
    check_eq(int64_t(dups.size()), 2, "kept edges");
    Graph dg = build_csr(dups);
    check_eq(dg.weights[0], 2, "dedup keeps the lighter parallel edge");

    g_case = "negative_weight";
    EdgeList neg;
    neg.n = 2;
    neg.add(0, 1, -3);
    threw = false;
    try {
        check_weights(build_csr(neg));
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "negative weight should be rejected");
}

void test_stats() {
    g_case = "stats";
    GraphStats s = graph_stats(build_csr(star(100, 1)));
    check_eq(s.max_degree, 99, "star hub degree");
    check(s.degree_stddev > 5.0, "star degree spread");

    SourceStats chain_from_end = source_stats(build_csr(chain(50, 1)), 0);
    check_eq(chain_from_end.levels, 50, "chain levels");
    check_eq(chain_from_end.max_frontier, 1, "chain frontier width");
    check_eq(chain_from_end.reached, 50, "chain reached");

    SourceStats star_from_hub = source_stats(build_csr(star(100, 1)), 0);
    check_eq(star_from_hub.levels, 2, "star levels");
    check_eq(star_from_hub.max_frontier, 99, "star frontier width");

    SourceStats cut = source_stats(build_csr(chain(50, 1)), 25);
    check_eq(cut.levels, 26, "chain levels from the middle");

    EdgeList island;
    island.n = 10;
    island.add(0, 1, 1);
    check_eq(source_stats(build_csr(island), 0).reached, 2, "reached");
}

void test_io() {
    g_case = "io_text";
    const std::string path = "/tmp/sssp_test_io.txt";
    EdgeList el;
    el.n = 4;
    el.add(0, 1, 3);
    el.add(1, 2, 4);
    el.add(2, 3, 5);
    write_edge_list(path, el);
    Graph g = load_graph(path);
    check_eq(g.n, 4, "n");
    check_eq(g.num_edges(), 3, "m");
    check_eq(g.max_weight(), 5, "max weight");

    g_case = "io_unweighted";
    const std::string unw = "/tmp/sssp_test_unweighted.txt";
    FILE* f = std::fopen(unw.c_str(), "w");
    std::fputs("# a comment\n0 1\n1 2\n2 3\n", f);
    std::fclose(f);
    Graph u = load_graph(unw);
    check_eq(u.num_edges(), 3, "no edge swallowed by a phantom header");
    check_eq(u.max_weight(), 1, "default weight");

    g_case = "io_binary";
    const std::string bin = "/tmp/sssp_test_io.bin";
    save_graph(bin, el);
    Graph b = load_graph(bin);
    check_eq(b.n, g.n, "n");
    check_eq(b.num_edges(), g.num_edges(), "m");
    check(b.offsets == g.offsets && b.targets == g.targets &&
              b.weights == g.weights,
          "binary round trip");

    g_case = "io_bad_magic";
    const std::string junk = "/tmp/sssp_test_junk.bin";
    f = std::fopen(junk.c_str(), "wb");
    std::fputs("not a graph at all, really", f);
    std::fclose(f);
    bool threw = false;
    try {
        load_graph(junk);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bad magic should throw");

    std::remove(path.c_str());
    std::remove(unw.c_str());
    std::remove(bin.c_str());
    std::remove(junk.c_str());
}

void test_residency() {
    g_case = "residency";
    if (!gpu_available()) return;

    std::mt19937_64 rng(4);
    Graph g = build_csr(random_graph(rng, 5000, 20000, 1, 100, true));
    std::vector<Weight> want = make_dijkstra_heap()->run(g, 0).dist;

    for (const auto& entry : solver_table()) {
        if (entry.device != "gpu") continue;
        auto solver = entry.make();

        Run first = solver->run(g, 0);
        Run second = solver->run(g, 1);
        Run third = solver->run(g, 0);

        check(first.timing.h2d_ms > 0.0,
              entry.name + " should pay the upload once");
        check(second.timing.h2d_ms == 0.0,
              entry.name + " should reuse the resident graph");
        check(second.timing.prep_ms == 0.0,
              entry.name + " should reuse derived arrays");
        check(third.dist == want,
              entry.name + " wrong after reuse");

        solver->release();
        check(solver->run(g, 0).timing.h2d_ms > 0.0,
              entry.name + " should re-upload after release");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) set_cpu_threads(std::atoi(argv[1]));

    std::printf("threads=%d gpu=%s\n", cpu_threads(),
                gpu_available() ? gpu_device_name().c_str() : "none");

    test_csr();
    test_io();
    test_stats();
    test_random_small();
    test_forced_delta();
    test_shapes();
    test_contention();
    test_residency();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
