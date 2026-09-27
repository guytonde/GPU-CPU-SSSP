#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sssp/csv.hpp"
#include "sssp/env.hpp"
#include "sssp/generators.hpp"
#include "sssp/registry.hpp"
#include "sssp/rng.hpp"
#include "sssp/solvers.hpp"
#include "sssp/timer.hpp"

using namespace sssp;
namespace fs = std::filesystem;

// C0: the cost of an empty round, a frontier round by size, one thread walking
// a hub, copies to and from the GPU, and an empty OpenMP parallel region.

namespace {

struct Out {
    CsvWriter csv;
    std::string run_id;
    void put(const std::string& bench, const std::string& param, double param_value,
             int rep, double value, const std::string& unit) {
        Row r;
        r.add("run_id", run_id)
            .add("bench", bench)
            .add("param", param)
            .add("param_value", param_value)
            .add("rep", int64_t(rep))
            .add("value", value)
            .add("unit", unit);
        csv.write(r);
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

void round_overhead(Out& out) {
    struct Pattern {
        const char* name;
        int launches, syncs;
        bool memset;
    };
    // The rounds of gpu-frontier and of the inner loop of gpu-nearfar.
    const Pattern patterns[] = {{"sync_only", 0, 1, false},
                                {"frontier_round", 2, 1, true},
                                {"nearfar_inner", 2, 2, false}};
    for (const auto& p : patterns) {
        for (int rep = 0; rep < 5; ++rep) {
            gpu_preheat();
            double us = gpu_round_overhead_us(p.launches, p.syncs, p.memset, 20000);
            out.put("round_overhead", p.name, p.syncs, rep, us, "us");
        }
        std::cout << "  round " << p.name << " done\n";
    }
}

void copies(Out& out) {
    for (int shift = 20; shift <= 30; shift += 2) {
        size_t bytes = size_t(1) << shift;
        for (bool pinned : {false, true}) {
            for (bool h2d : {true, false}) {
                double ms = gpu_copy_ms(bytes, pinned, h2d, 5);
                std::string name = std::string(h2d ? "h2d" : "d2h") + (pinned ? "_pinned" : "_pageable");
                out.put("copy", name, double(bytes), 0, ms, "ms");
            }
        }
    }
    std::cout << "  copies done\n";
}

void round_sweep(Out& out, const std::string& label, const Graph& g) {
    std::mt19937_64 rng(7);
    std::vector<int> all(g.n);
    for (int i = 0; i < g.n; ++i) all[i] = i;
    shuffle(all, rng);
    for (int shift = 0; shift <= 20; shift += 1) {
        int size = std::min(g.n, 1 << shift);
        std::vector<int> queue(all.begin(), all.begin() + size);
        gpu_preheat();
        RoundTiming t = gpu_frontier_round(g, queue, 15);
        out.put("frontier_round_" + label, "edges", double(t.edges), 0, t.round_us, "round_us");
        out.put("frontier_round_" + label, "edges", double(t.edges), 0, t.kernel_us, "kernel_us");
        out.put("frontier_round_" + label, "vertices", double(size), 0, double(t.edges), "edges");
    }
    std::cout << "  round sweep " << label << " done\n";
}

void hub_latency(Out& out) {
    GenSpec s;
    s.topology = "star";
    s.n = 1 << 20;
    s.wdist = "uniform";
    s.relabel = false;  // so vertex 0 is the hub
    Graph g = generate(s);
    const int hub = 0;
    for (const char* name : {"gpu-topo", "gpu-frontier"}) {
        const SolverEntry* e = find_solver(name);
        if (!e) continue;
        auto solver = e->make();
        solver->run(g, hub);
        std::vector<double> per_edge;
        for (int rep = 0; rep < 5; ++rep) {
            gpu_preheat();
            Run r = solver->run(g, hub);
            // gpu-topo walks the hub once per round, gpu-frontier once.
            double walks = std::string(name) == "gpu-topo" ? double(r.counters.iterations) : 1.0;
            per_edge.push_back(r.timing.solve_ms * 1e6 / (walks * g.degree(hub)));
        }
        out.put("hub_latency", name, double(g.degree(hub)), 0, median(per_edge), "ns_per_edge");
    }
    std::cout << "  hub latency done\n";
}

void omp_regions(Out& out) {
    int max_threads = omp_get_num_procs();
    for (int t : {1, 2, 4, 8, 16, 24, 32}) {
        if (t > max_threads) continue;
        for (int rep = 0; rep < 5; ++rep) {
            const int iters = 20000;
            volatile int sink = 0;
            for (int i = 0; i < 200; ++i) {
#pragma omp parallel num_threads(t)
                { sink = sink + 1; }
            }
            Timer timer;
            for (int i = 0; i < iters; ++i) {
#pragma omp parallel num_threads(t)
                { sink = sink + 1; }
            }
            out.put("omp_region", "threads", t, rep, timer.ms() * 1e3 / iters, "us");
        }
    }
    std::cout << "  omp regions done\n";
}

}  // namespace

int main(int argc, char** argv) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    std::string out_dir = "results/raw/C0";
    std::string cmdline;
    for (int i = 0; i < argc; ++i) cmdline += (i ? " " : "") + std::string(argv[i]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else {
            std::cout << "usage: calibrate [--out DIR]\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }

    try {
        const std::string started = utc_now();
        std::string run_id = started;
        run_id.erase(std::remove(run_id.begin(), run_id.end(), '-'), run_id.end());
        run_id.erase(std::remove(run_id.begin(), run_id.end(), ':'), run_id.end());
        run_id += "-c0";
        fs::path dir = fs::path(out_dir) / run_id;
        fs::create_directories(dir);

        const bool gpu = gpu_device_count() > 0;
        if (gpu) gpu_warmup();
        auto env_row = [&](const std::string& finished) {
            Row r;
            r.add("run_id", run_id)
                .add("experiment", "C0")
                .add("started_utc", started)
                .add("finished_utc", finished)
                .add("cmdline", cmdline);
            for (const auto& kv : host_environment()) r.add(kv.first, kv.second);
            for (const auto& kv : gpu_properties()) r.add(kv.first, kv.second);
            for (const auto& kv : gpu_nvml_info()) r.add(kv.first, kv.second);
            CsvWriter((dir / "runs.csv").string()).write(r);
        };
        env_row("");

        Out out{CsvWriter((dir / "calibration.csv").string()), run_id};
        std::cout << "calibration run " << run_id << "\n";
        omp_regions(out);
        if (gpu) {
            round_overhead(out);
            copies(out);
            GenSpec u;
            u.topology = "uniform";
            u.n = 1 << 22;
            u.m = int64_t(4) << 22;
            round_sweep(out, "uniform_deg8", generate(u));
            GenSpec m;
            m.topology = "grid";
            m.rows = m.cols = 2048;
            round_sweep(out, "grid_deg4", generate(m));
            hub_latency(out);
        }
        env_row(utc_now());
        std::cout << "wrote " << dir.string() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
