#include <omp.h>
#include <sched.h>

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include "sssp/solvers.hpp"

namespace sssp {

namespace {

// Physical cores among the CPUs in the affinity mask, so taskset is respected.
int physical_cores() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return omp_get_num_procs();

    std::set<std::string> cores;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &mask)) continue;
        std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(i) +
                         "/topology/thread_siblings_list");
        std::string siblings;
        cores.insert(in && std::getline(in, siblings) ? siblings : std::to_string(i));
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
