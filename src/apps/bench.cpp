#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sssp/graph.hpp"
#include "sssp/reference.hpp"
#include "sssp/registry.hpp"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

using namespace sssp;

namespace {

// Above this the O(V*E) oracle costs more than the benchmark it is checking.
constexpr int64_t kOracleWork = 200000000;

struct Options {
    std::string graph;
    int source = 0;
    int sources = 1;
    int reps = 5;
    int delta = 0;
    int threads = 0;
    uint64_t seed = 1;
    bool resident = false;
    bool shuffle = true;
    bool stats = false;
    bool list = false;
    int oracle = -1;  // -1 auto, 0 off, 1 on
    std::string only;
    std::string csv;
    std::string tag;
};

// Every timing sample a solver produced, so the report can show a distribution
// instead of a single run that happened to hit a quiet moment.
struct Samples {
    std::vector<Timing> reps;

    void add(const Timing& t) { reps.push_back(t); }

    std::vector<double> field(double (*get)(const Timing&)) const {
        std::vector<double> v;
        v.reserve(reps.size());
        for (const auto& t : reps) v.push_back(get(t));
        std::sort(v.begin(), v.end());
        return v;
    }
};

double prep_of(const Timing& t) { return t.prep_ms; }
double h2d_of(const Timing& t) { return t.h2d_ms; }
double solve_of(const Timing& t) { return t.solve_ms; }
double d2h_of(const Timing& t) { return t.d2h_ms; }
double total_of(const Timing& t) { return t.total_ms(); }
// What a further query costs once the graph is already on the device.
double query_of(const Timing& t) { return t.solve_ms + t.d2h_ms; }

double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    double pos = q * (sorted.size() - 1);
    size_t lo = size_t(pos);
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

double median_of(const Samples& s, double (*get)(const Timing&)) {
    return percentile(s.field(get), 0.5);
}

struct Result {
    std::string solver;
    std::string device;
    Samples all;                  // every source and rep
    std::vector<Samples> per_src; // one entry per source
    int64_t rounds = 0;
    int64_t mismatches = 0;
    bool checked = false;
};

void usage() {
    std::cout << R"(usage: bench <graph> [options]

  --source V       source vertex, and the first of a --sources list (default 0)
  --sources N      run N sources: --source plus N-1 sampled at random
  --reps N         repetitions per source, the median is reported (default 5)
  --resident       keep the graph on the device between sources, so the upload
                   is paid once instead of per query
  --delta D        bucket width for delta-stepping and near-far (default: auto)
  --threads N      OpenMP threads (default: physical cores, or $OMP_NUM_THREADS)
  --seed S         seed for source sampling and solver order (default 1)
  --no-shuffle     run solvers in registry order instead of a shuffled one
  --oracle         check against the O(V*E) reference regardless of size
  --no-oracle      never run the oracle
  --stats          print per-source search statistics
  --only a,b,c     run only these solvers
  --csv PATH       append a machine-readable row per solver per source
  --tag NAME       label written into the csv rows
  --list           print the solver table and exit

Distances are checked against an independent O(V*E) Bellman-Ford when the graph
is small enough for it, and otherwise against dijkstra, which then reports `ref`
rather than a result of its own.
)";
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };

        if (a == "-h" || a == "--help") return false;
        else if (a == "--list") o.list = true;
        else if (a == "--source") o.source = std::stoi(next());
        else if (a == "--sources") o.sources = std::stoi(next());
        else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--delta") o.delta = std::stoi(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--resident") o.resident = true;
        else if (a == "--no-shuffle") o.shuffle = false;
        else if (a == "--oracle") o.oracle = 1;
        else if (a == "--no-oracle") o.oracle = 0;
        else if (a == "--stats") o.stats = true;
        else if (a == "--only") o.only = next();
        else if (a == "--csv") o.csv = next();
        else if (a == "--tag") o.tag = next();
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else o.graph = a;
    }
    return true;
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

void print_table() {
    std::cout << "solvers:\n";
    for (const auto& e : solver_table()) {
        std::cout << "  " << std::left << std::setw(14) << e.name << e.device
                  << "\n";
    }
}

int64_t count_mismatches(const std::vector<Weight>& ref,
                         const std::vector<Weight>& got) {
    if (ref.size() != got.size()) return int64_t(ref.size());
    int64_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) ++bad;
    }
    return bad;
}

std::vector<int> pick_sources(const Options& o, int n) {
    std::vector<int> src{o.source};
    if (o.sources <= 1) return src;

    std::mt19937_64 rng(o.seed);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<char> used(n, 0);
    used[o.source] = 1;
    while (int(src.size()) < o.sources) {
        int v = pick(rng);
        if (used[v]) continue;
        used[v] = 1;
        src.push_back(v);
    }
    return src;
}

// A tag or a path holding a comma would otherwise shift every later column.
std::string csv_field(const std::string& v) {
    if (v.find_first_of(",\"\n") == std::string::npos) return v;
    std::string out = "\"";
    for (char c : v) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + '"';
}

double mteps(const Graph& g, double solve_ms) {
    return solve_ms > 0 ? double(g.num_edges()) / (solve_ms * 1e3) : 0.0;
}

void write_csv(const std::string& path, const Options& o, const Graph& g,
               const GraphStats& gs, const std::vector<int>& sources,
               const std::vector<SourceStats>& sstats,
               const std::vector<Result>& results) {
    bool fresh = !std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot append to " + path);

    if (fresh) {
        out << "tag,graph,n,m,avg_deg,max_deg,deg_stddev,min_w,max_w,"
               "source,source_deg,levels,reached,max_frontier,avg_frontier,"
               "solver,device,resident,reps,"
               "prep_ms,h2d_ms,solve_ms,d2h_ms,total_ms,"
               "solve_min,solve_p25,solve_p75,solve_max,"
               "rounds,mteps,threads,correct\n";
    }

    for (const auto& r : results) {
        for (size_t s = 0; s < sources.size(); ++s) {
            const Samples& smp = r.per_src[s];
            if (smp.reps.empty()) continue;
            std::vector<double> solve = smp.field(solve_of);
            double med = percentile(solve, 0.5);

            out << csv_field(o.tag) << ',' << csv_field(o.graph) << ','
                << g.n << ',' << g.num_edges()
                << ',' << gs.avg_degree << ',' << gs.max_degree << ','
                << gs.degree_stddev << ',' << gs.min_weight << ','
                << gs.max_weight << ',' << sources[s] << ','
                << g.degree(sources[s]) << ',' << sstats[s].levels << ','
                << sstats[s].reached << ',' << sstats[s].max_frontier << ','
                << sstats[s].avg_frontier << ',' << r.solver << ',' << r.device
                << ',' << (o.resident ? 1 : 0) << ',' << smp.reps.size() << ','
                << median_of(smp, prep_of) << ',' << median_of(smp, h2d_of)
                << ',' << med << ',' << median_of(smp, d2h_of) << ','
                << median_of(smp, total_of) << ',' << solve.front() << ','
                << percentile(solve, 0.25) << ',' << percentile(solve, 0.75)
                << ',' << solve.back() << ',' << r.rounds << ','
                << mteps(g, med) << ',' << cpu_threads() << ','
                << (r.mismatches == 0 ? 1 : 0) << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        if (!parse(argc, argv, o)) {
            usage();
            print_table();
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        usage();
        return 1;
    }

    if (o.list) {
        print_table();
        return 0;
    }
    if (o.graph.empty()) {
        usage();
        return 1;
    }

    try {
        Graph g = load_graph(o.graph);
        if (g.n == 0) throw std::runtime_error("graph has no vertices");
        check_weights(g);
        if (o.source < 0 || o.source >= g.n) {
            throw std::runtime_error("source out of range");
        }
        if (o.reps < 1) throw std::runtime_error("--reps must be at least 1");
        if (o.sources < 1 || o.sources > g.n) {
            throw std::runtime_error("--sources must be between 1 and n");
        }

        std::vector<std::string> wanted = split(o.only, ',');
        for (const auto& name : wanted) {
            if (!find_solver(name)) {
                throw std::runtime_error("no solver named " + name);
            }
        }

        set_cpu_threads(o.threads);
        const int gpus = gpu_device_count();
        if (gpus > 0) gpu_warmup();

        const GraphStats gs = graph_stats(g);
        const bool use_oracle =
            o.oracle == 1 ||
            (o.oracle < 0 && int64_t(g.n) * g.num_edges() <= kOracleWork);

        std::vector<int> sources = pick_sources(o, g.n);
        std::vector<SourceStats> sstats;
        for (int s : sources) sstats.push_back(source_stats(g, s));

        std::vector<const SolverEntry*> entries;
        std::vector<std::unique_ptr<Solver>> solvers;
        int skipped_gpu = 0;
        for (const auto& entry : solver_table()) {
            if (!wanted.empty() &&
                std::find(wanted.begin(), wanted.end(), entry.name) ==
                    wanted.end()) {
                continue;
            }
            if (entry.device == "gpu" && gpus == 0) {
                ++skipped_gpu;
                continue;
            }
            entries.push_back(&entry);
            solvers.push_back(entry.make());
            if (o.delta > 0) solvers.back()->set_delta(o.delta);
        }
        if (solvers.empty()) {
            std::cerr << "no solvers ran\n";
            return 1;
        }

        std::vector<Result> results(solvers.size());
        for (size_t i = 0; i < solvers.size(); ++i) {
            results[i].solver = entries[i]->name;
            results[i].device = entries[i]->device;
            results[i].per_src.resize(sources.size());
        }

        // dijkstra is both the speedup baseline and, when the oracle is too
        // expensive, the fallback reference. Pinning it here keeps the shuffled
        // execution order from deciding which solver goes unchecked.
        size_t base_row = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].solver == "dijkstra") base_row = i;
        }

        std::cout << o.graph << ": n=" << g.n << " m=" << g.num_edges()
                  << std::fixed << std::setprecision(1)
                  << " avg_deg=" << gs.avg_degree
                  << " max_deg=" << gs.max_degree
                  << " deg_sd=" << gs.degree_stddev
                  << " w=[" << gs.min_weight << "," << gs.max_weight << "]\n"
                  << "gpu: " << gpu_device_name() << "\n"
                  << "cpu: " << cpu_threads() << " omp threads\n"
                  << "sources=" << sources.size() << " reps=" << o.reps
                  << " resident=" << (o.resident ? "yes" : "no")
                  << " order=" << (o.shuffle ? "shuffled" : "registry")
                  << " check="
                  << (use_oracle ? std::string("oracle")
                                 : results[base_row].solver)
                  << "\n\n";

        std::vector<size_t> order(solvers.size());
        std::iota(order.begin(), order.end(), size_t(0));
        std::mt19937_64 rng(o.seed);

        for (size_t s = 0; s < sources.size(); ++s) {
            std::vector<Weight> reference;
            if (use_oracle) {
                reference = reference_sssp(g, sources[s]);
            } else {
                solvers[base_row]->release();
                reference = solvers[base_row]->run(g, sources[s]).dist;
            }

            for (int rep = 0; rep < o.reps; ++rep) {
                // A fixed order lets whichever solver runs first absorb the
                // cache and clock effects of the ones before it.
                if (o.shuffle) std::shuffle(order.begin(), order.end(), rng);

                for (size_t i : order) {
                    if (!o.resident) solvers[i]->release();
                    Run run = solvers[i]->run(g, sources[s]);

                    results[i].all.add(run.timing);
                    results[i].per_src[s].add(run.timing);
                    results[i].rounds = run.rounds;

                    if (use_oracle || i != base_row) {
                        results[i].mismatches +=
                            count_mismatches(reference, run.dist);
                        results[i].checked = true;
                    }
                }
            }
        }

        const double base = median_of(results[base_row].all, solve_of);

        std::cout << std::left << std::setw(14) << "solver" << std::setw(5)
                  << "dev" << std::right << std::setw(9) << "prep"
                  << std::setw(9) << "h2d" << std::setw(10) << "solve"
                  << std::setw(9) << "d2h" << std::setw(10) << "total"
                  << std::setw(9) << "rounds" << std::setw(9) << "MTEPS"
                  << std::setw(9) << "speedup" << "  ok\n";
        std::cout << std::string(94, '-') << "\n";

        for (const auto& r : results) {
            double solve = median_of(r.all, solve_of);
            std::cout << std::left << std::setw(14) << r.solver << std::setw(5)
                      << r.device << std::right << std::fixed
                      << std::setprecision(3) << std::setw(9)
                      << median_of(r.all, prep_of) << std::setw(9)
                      << median_of(r.all, h2d_of) << std::setw(10) << solve
                      << std::setw(9) << median_of(r.all, d2h_of)
                      << std::setw(10) << median_of(r.all, total_of)
                      << std::setw(9) << r.rounds << std::setprecision(1)
                      << std::setw(9) << mteps(g, solve) << std::setprecision(2)
                      << std::setw(8) << (solve > 0 ? base / solve : 0.0) << "x"
                      << "  "
                      << (!r.checked ? "ref" : (r.mismatches == 0 ? "yes" : "NO"))
                      << "\n";
        }
        std::cout << "\nmedians over " << o.reps << " reps x " << sources.size()
                  << (sources.size() == 1 ? " source" : " sources")
                  << ", speedup against " << results[base_row].solver << "\n";

        if (results[0].all.reps.size() > 1) {
            std::cout << "\nsolve ms spread\n"
                      << std::left << std::setw(14) << "solver" << std::right
                      << std::setw(11) << "min" << std::setw(11) << "p25"
                      << std::setw(11) << "median" << std::setw(11) << "p75"
                      << std::setw(11) << "max\n";
            for (const auto& r : results) {
                std::vector<double> v = r.all.field(solve_of);
                std::cout << std::left << std::setw(14) << r.solver
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(11) << v.front() << std::setw(11)
                          << percentile(v, 0.25) << std::setw(11)
                          << percentile(v, 0.5) << std::setw(11)
                          << percentile(v, 0.75) << std::setw(11) << v.back()
                          << "\n";
            }
        }

        if (o.resident) {
            // The upload happens once, on the first run; every later query
            // pays only solve + download. Break-even is where the difference
            // per query has covered it.
            double best_cpu = 0.0;
            std::string best_cpu_name;
            for (const auto& r : results) {
                if (r.device != "cpu") continue;
                double q = median_of(r.all, query_of);
                if (best_cpu_name.empty() || q < best_cpu) {
                    best_cpu = q;
                    best_cpu_name = r.solver;
                }
            }

            std::cout << "\namortised over one upload\n"
                      << std::left << std::setw(14) << "solver" << std::right
                      << std::setw(12) << "upload ms" << std::setw(15)
                      << "per query ms" << std::setw(26)
                      << "queries to beat " + best_cpu_name << "\n";
            for (const auto& r : results) {
                double upload = 0.0;
                for (const auto& t : r.all.reps) {
                    upload = std::max(upload, t.prep_ms + t.h2d_ms);
                }
                double q = median_of(r.all, query_of);

                std::string breakeven = "-";
                if (r.device == "gpu") {
                    breakeven = q < best_cpu
                                    ? std::to_string(int64_t(
                                          upload / (best_cpu - q)) + 1)
                                    : "never";
                }
                std::cout << std::left << std::setw(14) << r.solver
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(12) << upload << std::setw(15) << q
                          << std::setw(26) << breakeven << "\n";
            }
        }

        if (o.stats || sources.size() > 1) {
            std::cout << "\nsources\n"
                      << std::right << std::setw(12) << "vertex"
                      << std::setw(9) << "degree" << std::setw(9) << "levels"
                      << std::setw(12) << "reached" << std::setw(14)
                      << "max frontier" << std::setw(14) << "avg frontier\n";
            for (size_t s = 0; s < sources.size(); ++s) {
                std::cout << std::right << std::setw(12) << sources[s]
                          << std::setw(9) << g.degree(sources[s])
                          << std::setw(9) << sstats[s].levels << std::setw(12)
                          << sstats[s].reached << std::setw(14)
                          << sstats[s].max_frontier << std::fixed
                          << std::setprecision(1) << std::setw(14)
                          << sstats[s].avg_frontier << "\n";
            }
            std::cout << "levels is the hop eccentricity of the source, a lower "
                         "bound on the diameter\n";
        }

        for (const auto& r : results) {
            if (r.checked && r.mismatches > 0) {
                std::cout << "\n" << r.solver << " disagrees on "
                          << r.mismatches << " vertex-distances across all runs"
                          << "\n";
            }
        }
        if (sstats[0].reached < g.n) {
            std::cout << "\n" << g.n - sstats[0].reached << " of " << g.n
                      << " vertices unreachable from source " << sources[0]
                      << "\n";
        }
        if (skipped_gpu > 0) {
            std::cout << "\nskipped " << skipped_gpu
                      << " gpu solvers: no cuda device visible\n";
        }

        if (!o.csv.empty()) {
            write_csv(o.csv, o, g, gs, sources, sstats, results);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
