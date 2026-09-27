#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sssp/csv.hpp"
#include "sssp/delta.hpp"
#include "sssp/env.hpp"
#include "sssp/graph.hpp"
#include "sssp/reference.hpp"
#include "sssp/registry.hpp"
#include "sssp/rng.hpp"
#include "sssp/solvers.hpp"
#include "sssp/stats.hpp"
#include "sssp/timer.hpp"

using namespace sssp;
namespace fs = std::filesystem;

namespace {

// Above n*m of this, only the reference Dijkstra checks the results.
constexpr int64_t kOracleWork = 200000000;

struct Options {
    std::string graph;
    std::string experiment = "adhoc";
    std::string out = "results/raw/adhoc";
    std::string mode = "oneshot";
    std::vector<int> explicit_sources;
    int sources = 8;
    int queries = 0;
    int reps = 5;
    int warmup = 1;
    int delta = 0;
    double delta_scale = 0.0;
    int threads = 0;
    std::string affinity_label = "default";
    uint64_t seed = 1;
    bool shuffle = true;
    std::string cache = "warm";
    bool counters = true;
    bool instrument = false;
    bool preheat = true;
    int oracle = -1;
    std::string only;
    bool quiet = false;
    bool list = false;
    std::string cmdline;
};

void usage() {
    std::cout << R"(usage: bench <graph.bin> [options]

Runs the chosen solvers on one graph and writes raw per-execution samples to
<out>/<run_id>/{runs,graphs,sources,samples}.csv (and batch.csv in batch mode).

  --experiment X     label written to runs.csv (default adhoc)
  --out DIR          parent directory for the run directory (default results/raw/adhoc)
  --mode M           oneshot  every sample pays allocation, upload and free
                     resident sessions of --queries sequential queries, one upload each
                     batch    CPU serial solvers answer --queries sources concurrently
  --sources N        sources to sample (default 8); from meta "sources" if the graph
                     has them, otherwise uniformly from the giant component
  --source V         use this source (repeatable); overrides sampling
  --queries K        queries per resident session / batch (default: --sources)
  --reps N           timed repetitions per source, or sessions (default 5)
  --warmup N         untimed warm-up executions per solver, recorded as warmup=1 (default 1)
  --delta D          pin the bucket width for delta, delta-omp, gpu-nearfar
  --delta-scale X    bucket width = X * the heuristic
  --threads N        OpenMP threads (default: physical cores in the affinity mask)
  --affinity-label L free-form label for the pinning in force (recorded)
  --seed S           seed for sources and solver order (default 1)
  --no-shuffle       run solvers in registry order
  --cache C          warm | flushed (evict CPU L3 and GPU L2 before every sample)
  --no-counters      disable device-side work counters (to measure their cost)
  --instrument       per-kernel events; rows are flagged and excluded from timing analysis
  --no-preheat       skip the untimed pre-sample GPU warm-up
  --oracle | --no-oracle   force the O(V*E) check on or off (default: when n*m <= 2e8)
  --only a,b,c       run only these solvers
  --quiet            no summary table
  --list             print the solver table and exit
)";
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 0; i < argc; ++i) o.cmdline += (i ? " " : "") + std::string(argv[i]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "-h" || a == "--help") return false;
        else if (a == "--list") o.list = true;
        else if (a == "--experiment") o.experiment = next();
        else if (a == "--out") o.out = next();
        else if (a == "--mode") o.mode = next();
        else if (a == "--sources") o.sources = std::stoi(next());
        else if (a == "--source") o.explicit_sources.push_back(std::stoi(next()));
        else if (a == "--queries") o.queries = std::stoi(next());
        else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--delta") o.delta = std::stoi(next());
        else if (a == "--delta-scale") o.delta_scale = std::stod(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--affinity-label") o.affinity_label = next();
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--no-shuffle") o.shuffle = false;
        else if (a == "--cache") o.cache = next();
        else if (a == "--no-counters") o.counters = false;
        else if (a == "--instrument") o.instrument = true;
        else if (a == "--no-preheat") o.preheat = false;
        else if (a == "--oracle") o.oracle = 1;
        else if (a == "--no-oracle") o.oracle = 0;
        else if (a == "--only") o.only = next();
        else if (a == "--quiet") o.quiet = true;
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else o.graph = a;
    }
    if (o.mode != "oneshot" && o.mode != "resident" && o.mode != "batch") {
        throw std::runtime_error("--mode must be oneshot, resident or batch");
    }
    if (o.cache != "warm" && o.cache != "flushed") {
        throw std::runtime_error("--cache must be warm or flushed");
    }
    if (o.reps < 1 || o.sources < 1) throw std::runtime_error("--reps and --sources must be >= 1");
    return true;
}

std::string make_run_id() {
    std::random_device rd;
    char tail[8];
    std::snprintf(tail, sizeof(tail), "%04x", unsigned(rd() & 0xffff));
    std::string t = utc_now();
    std::string compact;
    for (char c : t) {
        if (c != '-' && c != ':') compact += c;
    }
    return compact + "-" + tail;
}

// Sources come from --source, the graph's meta (layer 0 of a layered graph),
// or random vertices of the giant component.
std::vector<int> pick_sources(const Graph& g, const std::vector<int>& comp,
                              const Options& o, int count, std::string& policy) {
    std::mt19937_64 rng(o.seed);
    std::vector<int> pool;
    if (!o.explicit_sources.empty()) {
        policy = "list";
        for (int v : o.explicit_sources) {
            if (v < 0 || v >= g.n) throw std::runtime_error("source out of range");
        }
        pool = o.explicit_sources;
    } else if (g.meta.has("sources")) {
        policy = "meta";
        for (const auto& s : split(g.meta.get("sources"), ',')) pool.push_back(std::stoi(s));
        shuffle(pool, rng);
    } else {
        policy = "giant";
        std::vector<char> used(g.n, 0);
        int64_t attempts = 0;
        while (int(pool.size()) < count && attempts < int64_t(g.n) * 20 + 1000) {
            ++attempts;
            int v = int(below(rng, uint64_t(g.n)));
            if (used[v] || comp[v] != 0 || g.degree(v) == 0) continue;
            used[v] = 1;
            pool.push_back(v);
        }
    }
    if (pool.empty()) throw std::runtime_error("no eligible source vertices");
    std::vector<int> out;
    for (int i = 0; i < count; ++i) out.push_back(pool[i % pool.size()]);
    if (count > int(pool.size())) policy += "+repeat";
    return out;
}

struct Reference {
    std::vector<Weight> dist;
    uint64_t hash = 0;
    std::string kind;
};

Reference make_reference(const Graph& g, int source, bool oracle, SourceStats& stats,
                         const std::vector<int>& comp) {
    RefResult rd = reference_dijkstra(g, source);
    Reference ref;
    ref.kind = "refdijkstra";
    if (oracle) {
        if (reference_sssp(g, source) != rd.dist) {
            throw std::runtime_error("oracle and reference dijkstra disagree at source " +
                                     std::to_string(source));
        }
        ref.kind = "oracle";
    }
    stats = source_stats(g, source, rd, comp);
    ref.hash = dist_hash(rd.dist);
    ref.dist = std::move(rd.dist);
    return ref;
}

int64_t count_mismatches(const std::vector<Weight>& a, const std::vector<Weight>& b) {
    if (a.size() != b.size()) return int64_t(std::max(a.size(), b.size()));
    int64_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
    return bad;
}

// Streams through a buffer much larger than the L3.
void flush_host_caches() {
    static std::vector<unsigned char> junk(size_t(512) << 20);
    static unsigned char fill = 0;
    std::memset(junk.data(), ++fill, junk.size());
}

Row graph_row(const std::string& run_id, const std::string& path, const Graph& g,
              const GraphStats& s) {
    Row r;
    r.add("run_id", run_id)
        .add("graph_id", hex64(graph_hash(g)))
        .add("topo_hash", hex64(topo_hash(g)))
        .add("weights_hash", hex64(weights_hash(g)))
        .add("path", path)
        .add("n", int64_t(g.n))
        .add("m", g.num_edges())
        .add("deg_max", int64_t(s.deg_max))
        .add("components", int64_t(s.components))
        .add("giant_frac", s.giant_frac)
        .add("w_min", int64_t(s.w_min))
        .add("w_max", int64_t(s.w_max));
    for (const char* k : {"generator", "gen_git_sha", "gen_git_dirty", "topology",
                          "m_requested", "m_undirected", "directed", "seed", "wseed",
                          "rseed", "relabel", "wdist", "wmin", "wmax", "decades", "rows",
                          "cols", "layer_width", "layer_links", "hub_fraction"}) {
        r.add(k, g.meta.get(k));
    }
    return r;
}

Row source_row(const std::string& run_id, const std::string& graph_id,
               const SourceStats& s, const Reference& ref) {
    Row r;
    r.add("run_id", run_id)
        .add("graph_id", graph_id)
        .add("source", int64_t(s.source))
        .add("source_degree", int64_t(s.source_degree))
        .add("in_giant", s.in_giant)
        .add("reached", s.reached)
        .add("m_reached", s.m_reached)
        .add("bfs_levels", int64_t(s.bfs_levels))
        .add("bfs_frontier_max", int64_t(s.bfs_frontier_max))
        .add("bfs_frontier_wmean", s.bfs_frontier_wmean)
        .add("sp_depth_max", int64_t(s.sp_depth_max))
        .add("sp_depth_mean", s.sp_depth_mean)
        .add("max_dist", s.max_dist)
        .add("ref_kind", ref.kind)
        .add("ref_dist_hash", hex64(ref.hash));
    return r;
}

struct Sample {
    std::string solver, device, session;
    int source = 0, query_index = 0, rep = 0, order_pos = 0;
    bool warmup = false;
    Run run;
    double run_ms = 0.0, free_ms = 0.0;
    GpuSample gpu;
    double load = -1;
    double clock_mhz = std::nan("");
    int64_t mismatches = 0;
    uint64_t hash = 0;
};

class Bench {
public:
    Bench(const Options& o, const Graph& g, const std::string& run_id, const fs::path& dir)
        : o_(o), g_(g), run_id_(run_id), graph_id_(hex64(graph_hash(g))),
          samples_((dir / "samples.csv").string()) {}

    Sample execute(Solver& solver, int source, bool release_after) {
        Sample s;
        s.solver = solver.name();
        s.device = solver.device();
        const bool gpu = s.device == "gpu";
        if (o_.cache == "flushed") {
            flush_host_caches();
            gpu_flush_l2();
        }
        if (o_.preheat && gpu) s.clock_mhz = gpu_preheat();
        s.gpu = gpu_sample();
        s.load = load1();
        s.source = source;

        Timer wall;
        s.run = solver.run(g_, source);
        s.run_ms = wall.ms();
        if (release_after) {
            Timer f;
            solver.release();
            s.free_ms = f.ms();
        }
        s.hash = dist_hash(s.run.dist);
        return s;
    }

    void check(Sample& s, const Reference& ref) {
        s.mismatches = s.hash == ref.hash ? 0 : count_mismatches(s.run.dist, ref.dist);
        if (s.mismatches) fail(s);
    }

    void write(Sample& s) {
        const Timing& t = s.run.timing;
        const Counters& c = s.run.counters;
        Row r;
        r.add("run_id", run_id_)
            .add("graph_id", graph_id_)
            .add("source", int64_t(s.source))
            .add("solver", s.solver)
            .add("device", s.device)
            .add("mode", o_.mode)
            .add("session", s.session)
            .add("query_index", int64_t(s.query_index))
            .add("rep", int64_t(s.rep))
            .add("warmup", s.warmup)
            .add("instrumented", o_.instrument)
            .add("order_pos", int64_t(s.order_pos))
            .add("cache_state", o_.cache)
            .add("threads", int64_t(s.device == "cpu" ? cpu_threads() : 0))
            .add("counters_on", o_.counters)
            .add("delta_used", int64_t(s.run.delta_used))
            .add("wall_ms", s.run_ms + s.free_ms)
            .add("prep_ms", t.prep_ms)
            .add("alloc_ms", t.alloc_ms)
            .add("h2d_ms", t.h2d_ms)
            .add("solve_ms", t.solve_ms)
            .add("d2h_ms", t.d2h_ms)
            .add("free_ms", s.free_ms)
            .add("kernel_ms", t.kernel_ms >= 0 ? t.kernel_ms : std::nan(""))
            .add("iterations", c.iterations)
            .add("sync_rounds", c.sync_rounds)
            .add("host_syncs", c.host_syncs)
            .add("parallel_phases", c.parallel_phases)
            .add("serial_phases", c.serial_phases)
            .add("vertices_expanded", c.vertices_expanded)
            .add("edges_touched", c.edges_touched)
            .add("mismatches", s.mismatches)
            .add("dist_hash", hex64(s.hash))
            .add("gpu_clock_mhz_measured", s.clock_mhz)
            .add("gpu_foreign_procs", int64_t(s.gpu.foreign_procs))
            .add("load1", s.load);
        samples_.write(r);
        if (!s.warmup) {
            summary_[s.solver].push_back({s.run_ms + s.free_ms, t.solve_ms, t.h2d_ms,
                                          double(c.host_syncs), double(c.edges_touched)});
        }
    }

    void fail(const Sample& s) {
        std::cerr << "MISMATCH " << s.solver << " source " << s.source << ": "
                  << s.mismatches << " vertices\n";
        failures_++;
    }

    int failures() const { return failures_; }

    void print_summary(std::ostream& out) const {
        out << std::left << std::setw(14) << "solver" << std::right << std::setw(6) << "runs"
            << std::setw(11) << "wall ms" << std::setw(11) << "solve ms" << std::setw(10)
            << "h2d ms" << std::setw(9) << "syncs" << std::setw(13) << "edges"
            << "\n";
        for (const auto& kv : summary_) {
            auto med = [&](int field) {
                std::vector<double> v;
                for (const auto& s : kv.second) v.push_back(s[field]);
                std::sort(v.begin(), v.end());
                return v.empty() ? 0.0 : v[v.size() / 2];
            };
            out << std::left << std::setw(14) << kv.first << std::right << std::setw(6)
                << kv.second.size() << std::fixed << std::setprecision(3) << std::setw(11)
                << med(0) << std::setw(11) << med(1) << std::setw(10) << med(2)
                << std::setprecision(0) << std::setw(9) << med(3) << std::setw(13) << med(4)
                << "\n";
        }
    }

private:
    const Options& o_;
    const Graph& g_;
    std::string run_id_, graph_id_;
    CsvWriter samples_;
    int failures_ = 0;
    // wall, solve, h2d, syncs, edges per timed sample
    std::map<std::string, std::vector<std::array<double, 5>>> summary_;
};

Row run_row(const Options& o, const std::string& run_id, const std::string& graph_id,
            const std::string& started, const std::string& finished,
            const std::string& policy, int count, const KeyValues& host,
            const KeyValues& gpu, const GpuSample& at_start, double load_start) {
    Row r;
    r.add("run_id", run_id)
        .add("experiment", o.experiment)
        .add("started_utc", started)
        .add("finished_utc", finished)
        .add("cmdline", o.cmdline)
        .add("graph_path", o.graph)
        .add("graph_id", graph_id)
        .add("mode", o.mode)
        .add("reps", int64_t(o.reps))
        .add("warmup", int64_t(o.warmup))
        .add("sources", int64_t(count))
        .add("queries", int64_t(o.queries))
        .add("source_policy", policy)
        .add("bench_seed", int64_t(o.seed))
        .add("order_policy", o.shuffle ? "shuffled" : "registry")
        .add("cache_state", o.cache)
        .add("counters", o.counters)
        .add("instrument", o.instrument)
        .add("preheat", o.preheat)
        .add("threads", int64_t(cpu_threads()))
        .add("affinity_label", o.affinity_label)
        .add("delta_forced", int64_t(o.delta))
        .add("delta_scale", o.delta_scale)
        .add("oracle_limit", kOracleWork)
        .add("gpu_block", int64_t(256))
        .add("gpu_util_start_pct", int64_t(at_start.util_pct))
        .add("gpu_foreign_procs_start", int64_t(at_start.foreign_procs))
        .add("load1_start", load_start)
        .add("load1_end", finished.empty() ? std::nan("") : load1());
    for (const auto& kv : host) r.add(kv.first, kv.second);
    for (const auto& kv : gpu) r.add(kv.first, kv.second);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    // Otherwise the first launch of each kernel pays for loading it.
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);

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
    if (o.list) {
        for (const auto& e : solver_table()) std::cout << e.name << " " << e.device << "\n";
        return 0;
    }
    if (o.graph.empty()) {
        usage();
        return 1;
    }

    try {
        const std::string started = utc_now();
        Graph g = load_graph(o.graph);
        if (g.n == 0) throw std::runtime_error("graph has no vertices");
        check_weights(g);

        std::vector<std::string> wanted = split(o.only, ',');
        for (const auto& name : wanted) {
            if (!find_solver(name)) throw std::runtime_error("no solver named " + name);
        }

        set_cpu_threads(o.threads);
        const int gpus = gpu_device_count();
        if (gpus > 0) gpu_warmup();
        const GpuSample at_start = gpu_sample();
        const double load_start = load1();

        const std::string run_id = make_run_id();
        const fs::path dir = fs::path(o.out) / run_id;
        fs::create_directories(dir);
        const std::string graph_id = hex64(graph_hash(g));

        const GraphStats gs = graph_stats(g);
        const std::vector<int> comp = component_ids(g);
        const int count = o.mode == "oneshot" ? o.sources : (o.queries > 0 ? o.queries : o.sources);
        std::string policy;
        const std::vector<int> sources = pick_sources(g, comp, o, count, policy);
        const bool use_oracle =
            o.oracle == 1 || (o.oracle < 0 && int64_t(g.n) * g.num_edges() <= kOracleWork);

        KeyValues host = host_environment();
        KeyValues gpu = gpu_properties();
        for (const auto& kv : gpu_nvml_info()) gpu.push_back(kv);
        CsvWriter(dir / "runs.csv")
            .write(run_row(o, run_id, graph_id, started, "", policy, count, host, gpu,
                           at_start, load_start));
        CsvWriter((dir / "graphs.csv").string()).write(graph_row(run_id, o.graph, g, gs));

        std::vector<const SolverEntry*> entries;
        std::vector<std::unique_ptr<Solver>> solvers;
        for (const auto& entry : solver_table()) {
            if (!wanted.empty() &&
                std::find(wanted.begin(), wanted.end(), entry.name) == wanted.end()) {
                continue;
            }
            if (entry.device == "gpu" && gpus == 0) continue;
            if (o.mode == "batch" && (entry.device != "cpu" || entry.name == "delta-omp")) continue;
            auto s = entry.make();
            if (o.delta > 0) {
                s->set_delta(o.delta);
            } else if (o.delta_scale > 0) {
                s->set_delta(std::max(1, int(std::llround(pick_delta(g) * o.delta_scale))));
            }
            s->set_counters(o.counters);
            s->set_instrument(o.instrument);
            entries.push_back(&entry);
            solvers.push_back(std::move(s));
        }
        if (solvers.empty()) throw std::runtime_error("no solvers to run");

        if (!o.quiet) {
            std::cout << o.graph << ": n=" << g.n << " m=" << g.num_edges()
                      << " graph_id=" << graph_id << "\n"
                      << "run " << run_id << " mode=" << o.mode << " sources=" << count
                      << " (" << policy << ") reps=" << o.reps << " threads=" << cpu_threads()
                      << " gpu=" << gpu_device_name() << "\n";
        }

        CsvWriter source_csv((dir / "sources.csv").string());
        std::map<int, Reference> refs;
        auto reference_for = [&](int s) -> const Reference& {
            auto it = refs.find(s);
            if (it != refs.end()) return it->second;
            SourceStats st;
            Reference ref = make_reference(g, s, use_oracle, st, comp);
            source_csv.write(source_row(run_id, graph_id, st, ref));
            if (o.mode != "oneshot") ref.dist.clear();
            return refs.emplace(s, std::move(ref)).first->second;
        };
        auto recheck = [&](Sample& smp, const Reference& ref) {
            if (!ref.dist.empty() || smp.hash == ref.hash) return;
            Reference full;
            full.hash = ref.hash;
            full.dist = reference_dijkstra(g, smp.source).dist;
            smp.mismatches = count_mismatches(smp.run.dist, full.dist);
        };

        Bench bench(o, g, run_id, dir);
        std::mt19937_64 order_rng(o.seed + 1);
        std::vector<size_t> order(solvers.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;

        for (int w = 0; w < o.warmup && o.mode != "batch"; ++w) {
            const Reference& ref = reference_for(sources[0]);
            for (size_t i = 0; i < solvers.size(); ++i) {
                Sample s = bench.execute(*solvers[i], sources[0], true);
                s.warmup = true;
                s.rep = -1 - w;
                bench.check(s, ref);
                recheck(s, ref);
                bench.write(s);
            }
        }

        if (o.mode == "oneshot") {
            for (int src : sources) {
                const Reference& ref = reference_for(src);
                for (int rep = 0; rep < o.reps; ++rep) {
                    if (o.shuffle) shuffle(order, order_rng);
                    for (size_t pos = 0; pos < order.size(); ++pos) {
                        Sample s = bench.execute(*solvers[order[pos]], src, true);
                        s.rep = rep;
                        s.order_pos = int(pos);
                        bench.check(s, ref);
                        bench.write(s);
                    }
                }
                refs[src].dist = std::vector<Weight>();
            }
        } else if (o.mode == "resident") {
            for (int src : sources) reference_for(src);
            for (int rep = 0; rep < o.reps; ++rep) {
                if (o.shuffle) shuffle(order, order_rng);
                for (size_t pos = 0; pos < order.size(); ++pos) {
                    Solver& solver = *solvers[order[pos]];
                    solver.release();
                    for (int q = 0; q < int(sources.size()); ++q) {
                        const bool last = q + 1 == int(sources.size());
                        Sample s = bench.execute(solver, sources[q], last);
                        s.session = "s" + std::to_string(rep);
                        s.query_index = q;
                        s.rep = rep;
                        s.order_pos = int(pos);
                        const Reference& ref = refs.at(sources[q]);
                        s.mismatches = s.hash == ref.hash ? 0 : -1;
                        recheck(s, ref);
                        if (s.mismatches) bench.fail(s);
                        bench.write(s);
                    }
                }
            }
        } else {
            // One independent serial solve per thread.
            for (int src : sources) reference_for(src);
            CsvWriter batch_csv((dir / "batch.csv").string());
            for (int rep = 0; rep < o.reps; ++rep) {
                for (size_t i = 0; i < solvers.size(); ++i) {
                    const SolverEntry& entry = *entries[i];
                    std::atomic<int64_t> bad{0};
                    Timer wall;
#pragma omp parallel for schedule(dynamic, 1) num_threads(cpu_threads())
                    for (int q = 0; q < int(sources.size()); ++q) {
                        auto solver = entry.make();
                        Run r = solver->run(g, sources[q]);
                        if (dist_hash(r.dist) != refs.at(sources[q]).hash) bad++;
                    }
                    double ms = wall.ms();
                    Row r;
                    r.add("run_id", run_id)
                        .add("graph_id", graph_id)
                        .add("solver", entry.name)
                        .add("batch_k", int64_t(sources.size()))
                        .add("threads", int64_t(cpu_threads()))
                        .add("rep", int64_t(rep))
                        .add("wall_ms", ms)
                        .add("mismatched_queries", int64_t(bad.load()));
                    batch_csv.write(r);
                    if (!o.quiet) {
                        std::cout << "batch " << entry.name << " K=" << sources.size()
                                  << " wall=" << ms << " ms\n";
                    }
                }
            }
        }

        CsvWriter(dir / "runs.csv")
            .write(run_row(o, run_id, graph_id, started, utc_now(), policy, count, host, gpu,
                           at_start, load_start));
        if (!o.quiet && o.mode != "batch") bench.print_summary(std::cout);
        if (!o.quiet) std::cout << "wrote " << dir.string() << "\n";
        if (bench.failures() > 0) {
            std::cerr << bench.failures() << " incorrect results\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
