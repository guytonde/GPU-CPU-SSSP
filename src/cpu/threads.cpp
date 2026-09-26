#include <omp.h>

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// Distinct physical cores, by grouping logical cpus that share a sibling list.
// SSSP is memory bound so the second hyperthread buys nothing, and on a shared
// machine oversubscribing leaves every OpenMP barrier waiting on a descheduled
// thread, turning a 2 us barrier into a multi-millisecond one.
int physical_cores() {
    std::set<std::string> cores;
    for (int i = 0; i < omp_get_num_procs(); ++i) {
        std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(i) +
                         "/topology/thread_siblings_list");
        std::string siblings;
        if (in && std::getline(in, siblings)) cores.insert(siblings);
    }
    return cores.empty() ? omp_get_num_procs() : int(cores.size());
}

int g_threads = 0;

}  // namespace

void set_cpu_threads(int n) {
    if (n <= 0) {
        n = std::getenv("OMP_NUM_THREADS") ? omp_get_max_threads()
                                           : physical_cores();
    }
    g_threads = n;
    omp_set_num_threads(n);
}

int cpu_threads() {
    if (g_threads == 0) set_cpu_threads(0);
    return g_threads;
}

}  // namespace sssp
