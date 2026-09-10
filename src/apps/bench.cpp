#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sssp/graph.hpp"
#include "sssp/registry.hpp"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

using namespace sssp;

namespace {

struct Options {
    std::string graph;
    int source = 0;
    int reps = 3;
    int delta = 0;
    int threads = 0;
    std::string only;
    std::string csv;
    std::string tag;
    bool list = false;
};

struct Record {
    std::string solver;
    std::string device;
    Timing timing;
    int64_t rounds = 0;
    bool correct = true;
    int mismatches = 0;
};

void usage() {
    std::cout << R"(usage: bench <graph> [options]

  --source V       source vertex (default 0)
  --reps N         repetitions, best run is reported (default 3)
  --delta D        bucket width for delta-stepping and near-far (default: auto)
  --threads N      OpenMP threads (default: physical cores, or $OMP_NUM_THREADS)
  --only a,b,c     run only these solvers
  --csv PATH       append a machine-readable row per solver
  --tag NAME       label written into the csv rows
  --list           print the solver table and exit

The first CPU solver in the table is the reference; every other solver's
distance array is compared against it.
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
        else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--delta") o.delta = std::stoi(next());
        else if (a == "--threads") o.threads = std::stoi(next());
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
        std::cout << "  " << std::left << std::setw(14) << e.name
                  << std::setw(5) << e.device << e.blurb << "\n";
    }
}

int count_mismatches(const std::vector<Weight>& ref,
                     const std::vector<Weight>& got) {
    if (ref.size() != got.size()) return int(std::max(ref.size(), got.size()));
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) ++bad;
    }
    return bad;
}

void write_csv(const std::string& path, const Options& o, const Graph& g,
               const std::vector<Record>& records) {
    bool fresh = !std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot append to " + path);

    if (fresh) {
        out << "tag,graph,n,m,source,solver,device,solve_ms,transfer_ms,"
               "total_ms,rounds,mteps,threads,correct\n";
    }
    for (const auto& r : records) {
        double mteps = r.timing.solve_ms > 0
                           ? double(g.num_edges()) / (r.timing.solve_ms * 1e3)
                           : 0.0;
        out << o.tag << ',' << o.graph << ',' << g.n << ',' << g.num_edges()
            << ',' << o.source << ',' << r.solver << ',' << r.device << ','
            << r.timing.solve_ms << ',' << r.timing.transfer_ms << ','
            << r.timing.total_ms() << ',' << r.rounds << ',' << mteps << ','
            << cpu_threads() << ',' << (r.correct ? 1 : 0) << '\n';
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
        if (o.source < 0 || o.source >= g.n) {
            throw std::runtime_error("source out of range");
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
        std::cout << o.graph << ": n=" << g.n << " m=" << g.num_edges()
                  << " avg_deg=" << std::fixed << std::setprecision(1)
                  << g.avg_degree() << " maxw=" << g.max_weight() << "\n"
                  << "gpu: " << gpu_device_name() << "\n"
                  << "cpu: " << cpu_threads() << " omp threads\n"
                  << "source=" << o.source << " reps=" << o.reps << "\n\n";

        std::vector<Weight> reference;
        std::vector<Record> records;
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

            auto solver = entry.make();
            if (o.delta > 0) solver->set_delta(o.delta);

            Run best;
            for (int rep = 0; rep < o.reps; ++rep) {
                Run r = solver->run(g, o.source);
                if (rep == 0 || r.timing.total_ms() < best.timing.total_ms()) {
                    best = std::move(r);
                }
            }

            Record rec;
            rec.solver = entry.name;
            rec.device = entry.device;
            rec.timing = best.timing;
            rec.rounds = best.rounds;

            if (reference.empty()) {
                reference = best.dist;
            } else {
                rec.mismatches = count_mismatches(reference, best.dist);
                rec.correct = rec.mismatches == 0;
            }
            records.push_back(rec);
        }

        if (records.empty()) {
            std::cerr << "no solvers ran\n";
            return 1;
        }

        const double base = records.front().timing.solve_ms;
        std::cout << std::left << std::setw(14) << "solver" << std::setw(5)
                  << "dev" << std::right << std::setw(11) << "solve ms"
                  << std::setw(11) << "xfer ms" << std::setw(11) << "total ms"
                  << std::setw(9) << "rounds" << std::setw(10) << "MTEPS"
                  << std::setw(9) << "speedup" << "  ok\n";
        std::cout << std::string(88, '-') << "\n";

        for (const auto& r : records) {
            double mteps = r.timing.solve_ms > 0
                               ? double(g.num_edges()) / (r.timing.solve_ms * 1e3)
                               : 0.0;
            double speedup =
                r.timing.solve_ms > 0 ? base / r.timing.solve_ms : 0.0;

            std::cout << std::left << std::setw(14) << r.solver << std::setw(5)
                      << r.device << std::right << std::fixed
                      << std::setprecision(3) << std::setw(11)
                      << r.timing.solve_ms << std::setw(11)
                      << r.timing.transfer_ms << std::setw(11)
                      << r.timing.total_ms() << std::setw(9) << r.rounds
                      << std::setprecision(1) << std::setw(10) << mteps
                      << std::setprecision(2) << std::setw(8) << speedup << "x"
                      << "  " << (r.correct ? "yes" : "NO") << "\n";
        }

        for (const auto& r : records) {
            if (!r.correct) {
                std::cout << "\n" << r.solver << " disagrees with "
                          << records.front().solver << " on " << r.mismatches
                          << " of " << g.n << " vertices\n";
            }
        }

        int unreached = 0;
        for (Weight d : reference) {
            if (d == kInf) ++unreached;
        }
        if (unreached > 0) {
            std::cout << "\n" << unreached << " of " << g.n
                      << " vertices unreachable from source " << o.source
                      << "\n";
        }
        if (skipped_gpu > 0) {
            std::cout << "\nskipped " << skipped_gpu
                      << " gpu solvers: no cuda device visible\n";
        }

        if (!o.csv.empty()) write_csv(o.csv, o, g, records);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
