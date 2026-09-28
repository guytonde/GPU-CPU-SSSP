#pragma once

#include <string>
#include <utility>
#include <vector>

namespace sssp {

using KeyValues = std::vector<std::pair<std::string, std::string>>;

// Machine, environment and build details for runs.csv.
KeyValues host_environment();

// One minute load average, or -1.
double load1();

// GPU state from NVML, which is loaded at run time. -1 when unavailable.
struct GpuSample {
    int util_pct = -1;
    int foreign_procs = -1;  // compute processes on the GPU other than this one
};
GpuSample gpu_sample();
KeyValues gpu_nvml_info();

std::string utc_now();

}  // namespace sssp
