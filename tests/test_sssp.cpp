#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "sssp/csv.hpp"
#include "sssp/generators.hpp"
#include "sssp/graph.hpp"
#include "sssp/reference.hpp"
#include "sssp/registry.hpp"
#include "sssp/solvers.hpp"
#include "sssp/stats.hpp"

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

// Checks every registered solver against `want`. delta 0 means the default.
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

// Large enough for delta-omp's parallel path, with all threads updating a few vertices.
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
    // Too big for the oracle.
    std::vector<Weight> want = reference_dijkstra(g, 0).dist;
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
    std::vector<Weight> ref = reference_dijkstra(rg, 0).dist;
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

SourceStats stats_from(const Graph& g, int source) {
    return source_stats(g, source, reference_dijkstra(g, source), component_ids(g));
}

void test_stats() {
    g_case = "stats";
    Graph st = build_csr(star(100, 1));
    GraphStats s = graph_stats(st);
    check_eq(s.deg_max, 99, "star hub degree");
    check_eq(s.components, 1, "star components");

    SourceStats chain_from_end = stats_from(build_csr(chain(50, 1)), 0);
    check_eq(chain_from_end.bfs_levels, 50, "chain levels");
    check_eq(chain_from_end.bfs_frontier_max, 1, "chain frontier width");
    check_eq(chain_from_end.reached, 50, "chain reached");
    check_eq(chain_from_end.sp_depth_max, 49, "chain shortest-path depth");
    check_eq(chain_from_end.m_reached, 98, "chain useful arcs");

    SourceStats star_from_hub = stats_from(st, 0);
    check_eq(star_from_hub.bfs_levels, 2, "star levels");
    check_eq(star_from_hub.bfs_frontier_max, 99, "star frontier width");

    SourceStats cut = stats_from(build_csr(chain(50, 1)), 25);
    check_eq(cut.bfs_levels, 26, "chain levels from the middle");

    EdgeList island;
    island.n = 10;
    island.add(0, 1, 1);
    Graph ig = build_csr(island);
    check_eq(stats_from(ig, 0).reached, 2, "reached");
    check_eq(graph_stats(ig).components, 9, "components");

    // A heavy direct edge and a light two-hop path: the shortest path takes
    // two hops even though BFS reaches the target in one.
    g_case = "sp_depth";
    EdgeList tri;
    tri.n = 3;
    tri.add(0, 2, 10);
    tri.add(0, 1, 1);
    tri.add(1, 2, 1);
    SourceStats t = stats_from(build_csr(tri), 0);
    check_eq(t.sp_depth_max, 2, "weighted hop depth");
    check_eq(t.bfs_levels, 2, "bfs levels");
    check_eq(t.max_dist, 2, "max distance");
}

void test_reference() {
    g_case = "reference_dijkstra";
    for (int seed = 0; seed < 200; ++seed) {
        std::mt19937_64 rng(seed + 500);
        int n = 2 + int(rng() % 80);
        EdgeList el = random_graph(rng, n, int(rng() % (4 * n + 1)), 0, 1 + int(rng() % 50),
                                   rng() % 2 == 0);
        Graph g = build_csr(el);
        int src = int(rng() % n);
        check(reference_dijkstra(g, src).dist == reference_sssp(el, src),
              "reference dijkstra agrees with the oracle");
    }
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
    Graph out = build_csr(el);
    out.meta.set("topology", "test");
    out.meta.set("seed", "42");
    save_graph(bin, out);
    Graph b = load_graph(bin);
    check_eq(b.n, g.n, "n");
    check_eq(b.num_edges(), g.num_edges(), "m");
    check(b.offsets == g.offsets && b.targets == g.targets &&
              b.weights == g.weights,
          "binary round trip");
    check(b.meta.get("seed") == "42" && b.meta.get("topology") == "test",
          "provenance survives the round trip");
    check(graph_hash(b) == graph_hash(out), "hash stable across the round trip");

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

    g_case = "csv_columns";
    const std::string csv = "/tmp/sssp_test.csv";
    threw = false;
    try {
        CsvWriter w(csv);
        Row a;
        a.add("x", int64_t(1)).add("y", int64_t(2));
        w.write(a);
        Row b2;
        b2.add("y", int64_t(2)).add("x", int64_t(1));
        w.write(b2);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a row with shifted columns should throw");

    std::remove(path.c_str());
    std::remove(unw.c_str());
    std::remove(bin.c_str());
    std::remove(junk.c_str());
    std::remove(csv.c_str());
}

GenSpec spec(const std::string& topo, int64_t n, int64_t m) {
    GenSpec s;
    s.topology = topo;
    s.n = n;
    s.m = m;
    return s;
}

void test_generators() {
    g_case = "gen_topology_invariant_under_weights";
    for (const char* topo : {"uniform", "rmat", "grid", "geometric", "layered", "hub"}) {
        GenSpec s = spec(topo, 1 << 12, 4 << 12);
        s.layer_width = 64;
        s.hub_fraction = 0.05;
        s.wdist = "unit";
        Graph a = generate(s);
        s.wdist = "uniform";
        s.wmax = 100;
        Graph b = generate(s);
        s.wdist = "logunif";
        s.decades = 4;
        Graph c = generate(s);
        check(topo_hash(a) == topo_hash(b) && topo_hash(b) == topo_hash(c),
              std::string(topo) + " topology changed with the weights");
        check(weights_hash(b) != weights_hash(c), std::string(topo) + " weights differ");
        check(graph_hash(generate(s)) == graph_hash(c), std::string(topo) + " deterministic");
    }

    g_case = "gen_relabel";
    GenSpec r = spec("rmat", 1 << 12, 4 << 12);
    Graph natural_rmat = [&] { GenSpec x = r; x.relabel = false; return generate(x); }();
    Graph relabeled = generate(r);
    check_eq(relabeled.num_edges(), natural_rmat.num_edges(), "relabel keeps the arcs");
    check_eq(graph_stats(relabeled).deg_max, graph_stats(natural_rmat).deg_max,
             "relabel keeps the degrees");
    check(graph_stats(natural_rmat).deg_max == natural_rmat.degree(0) ||
              natural_rmat.degree(0) > 10 * natural_rmat.avg_degree(),
          "unpermuted rmat puts a hub at vertex 0");

    g_case = "gen_no_spanning_tree";
    Graph empty = generate(spec("uniform", 1000, 0));
    check_eq(empty.num_edges(), 0, "no edges were added");
    check_eq(graph_stats(empty).components, 1000, "every vertex is its own component");

    g_case = "gen_grid_depth";
    GenSpec gs = spec("grid", 0, 0);
    gs.rows = 30;
    gs.cols = 50;
    gs.relabel = false;
    Graph grid = generate(gs);
    check_eq(grid.n, 1500, "grid n");
    check_eq(stats_from(grid, 0).bfs_levels, 30 + 50 - 1, "corner eccentricity");

    g_case = "gen_layered_depth";
    GenSpec ls = spec("layered", 64 * 100, 0);
    ls.layer_width = 64;
    Graph lay = generate(ls);
    check(lay.meta.has("sources"), "layered records its layer-0 sources");
    int src = std::stoi(lay.meta.get("sources").substr(0, lay.meta.get("sources").find(',')));
    SourceStats lst = stats_from(lay, src);
    check(lst.bfs_levels >= 100 && lst.bfs_levels <= 100 + 6,
          "layered depth tracks the layer count, got " + std::to_string(lst.bfs_levels));
    // Only vertices of the last layer that nobody linked to are unreachable.
    check(lst.reached >= 6400 - 64 && lst.reached <= 6400,
          "layered reaches all but part of the last layer, got " + std::to_string(lst.reached));

    g_case = "gen_hub";
    GenSpec hs = spec("hub", 1 << 14, 4 << 14);
    hs.hub_fraction = 0.25;
    GraphStats hst = graph_stats(generate(hs));
    check(hst.deg_max >= (1 << 14) - 1 - 64 && hst.deg_max <= (1 << 14),
          "hub degree tracks the fraction, got " + std::to_string(hst.deg_max));
}

// On unit weights the counters are exact.
void test_counters() {
    g_case = "counters";
    GenSpec s = spec("uniform", 1 << 12, 4 << 12);
    s.wdist = "unit";
    Graph g = generate(s);
    std::vector<int> comp = component_ids(g);
    int src = 0;
    while (comp[src] != 0 || g.degree(src) == 0) ++src;
    SourceStats st = stats_from(g, src);

    Run d = make_dijkstra_heap()->run(g, src);
    check_eq(d.counters.vertices_expanded, st.reached, "dijkstra expansions");
    check_eq(d.counters.edges_touched, st.m_reached, "dijkstra edges");

    if (!gpu_available()) return;
    auto frontier = find_solver("gpu-frontier")->make();
    Run f = frontier->run(g, src);
    check_eq(f.counters.vertices_expanded, st.reached, "gpu-frontier expansions");
    check_eq(f.counters.edges_touched, st.m_reached, "gpu-frontier edges");
    check_eq(f.counters.iterations, st.bfs_levels, "gpu-frontier rounds");
    check_eq(f.counters.host_syncs, st.bfs_levels, "one blocking sync per round");

    frontier->release();
    frontier->set_counters(false);
    Run off = frontier->run(g, src);
    check_eq(off.counters.edges_touched, 0, "device counters off");
    check(off.dist == f.dist, "counters do not change the answer");

    auto topo = find_solver("gpu-topo")->make();
    Run t = topo->run(g, src);
    check(t.counters.edges_touched >= st.m_reached, "gpu-topo rescans at least the useful arcs");
}

void test_residency() {
    g_case = "residency";
    if (!gpu_available()) return;

    std::mt19937_64 rng(4);
    Graph g = build_csr(random_graph(rng, 5000, 20000, 1, 100, true));
    std::vector<Weight> want = reference_dijkstra(g, 0).dist;

    for (const auto& entry : solver_table()) {
        if (entry.device != "gpu") continue;
        auto solver = entry.make();

        Run first = solver->run(g, 0);
        Run second = solver->run(g, 1);
        Run third = solver->run(g, 0);

        check(first.timing.h2d_ms > 0.0 && first.timing.alloc_ms > 0.0,
              entry.name + " should pay allocation and upload once");
        check(second.timing.h2d_ms == 0.0 && second.timing.alloc_ms == 0.0,
              entry.name + " should reuse the resident graph and buffers");
        check(second.timing.prep_ms == 0.0,
              entry.name + " should reuse derived arrays");
        check(third.dist == want, entry.name + " wrong after reuse");

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
    test_reference();
    test_generators();
    test_counters();
    test_random_small();
    test_forced_delta();
    test_shapes();
    test_contention();
    test_residency();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
