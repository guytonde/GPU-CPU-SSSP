#include "sssp/env.hpp"

#include <dlfcn.h>
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cctype>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <type_traits>

#include "sssp/build_info.hpp"

#if defined(__has_include)
#if __has_include(<nvml.h>)
#include <nvml.h>
#define SSSP_HAVE_NVML_H 1
#endif
#endif

namespace sssp {

namespace {

std::string read_line(const std::string& path) {
    std::ifstream in(path);
    std::string s;
    if (in) std::getline(in, s);
    return s;
}

std::string cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t c = line.find(':');
            return c == std::string::npos ? line : line.substr(c + 2);
        }
    }
    return "";
}

std::string mem_total() {
    std::ifstream in("/proc/meminfo");
    std::string key, unit;
    long long kb = 0;
    while (in >> key >> kb >> unit) {
        if (key == "MemTotal:") return std::to_string(kb * 1024);
    }
    return "";
}

// As a list like "0-7,16".
std::string affinity_list() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return "";
    std::ostringstream out;
    int start = -1, prev = -2;
    auto flush = [&]() {
        if (start < 0) return;
        if (out.tellp() > 0) out << ",";
        out << start;
        if (prev > start) out << "-" << prev;
    };
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &mask)) continue;
        if (i != prev + 1) {
            flush();
            start = i;
        }
        prev = i;
    }
    flush();
    return out.str();
}

int physical_cores_all() {
    std::set<std::string> cores;
    long n = sysconf(_SC_NPROCESSORS_CONF);
    for (long i = 0; i < n; ++i) {
        std::string s = read_line("/sys/devices/system/cpu/cpu" + std::to_string(i) +
                                  "/topology/thread_siblings_list");
        if (!s.empty()) cores.insert(s);
    }
    return int(cores.size());
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

}  // namespace

KeyValues host_environment() {
    KeyValues kv;
    utsname u{};
    uname(&u);
    kv.emplace_back("git_sha", build_git_sha());
    kv.emplace_back("git_dirty", build_git_dirty() ? "1" : "0");
    kv.emplace_back("git_diff_hash", build_diff_hash());
    kv.emplace_back("build_flags", build_flags());
    kv.emplace_back("hostname", u.nodename);
    kv.emplace_back("os_kernel", std::string(u.sysname) + " " + u.release);
    kv.emplace_back("cpu_model", cpu_model());
    kv.emplace_back("cpu_logical", std::to_string(sysconf(_SC_NPROCESSORS_CONF)));
    kv.emplace_back("cpu_physical", std::to_string(physical_cores_all()));
    kv.emplace_back("cpu_p_cpus", read_line("/sys/devices/cpu_core/cpus"));
    kv.emplace_back("cpu_e_cpus", read_line("/sys/devices/cpu_atom/cpus"));
    kv.emplace_back("cpu_l3", read_line("/sys/devices/system/cpu/cpu0/cache/index3/size"));
    kv.emplace_back("cpu_governor",
                    read_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"));
    std::string no_turbo = read_line("/sys/devices/system/cpu/intel_pstate/no_turbo");
    kv.emplace_back("cpu_turbo", no_turbo.empty() ? "" : (no_turbo == "0" ? "1" : "0"));
    kv.emplace_back("smt_active", read_line("/sys/devices/system/cpu/smt/active"));
    kv.emplace_back("mem_total_bytes", mem_total());
    kv.emplace_back("affinity", affinity_list());
    for (const char* v : {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                          "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "CUDA_MODULE_LOADING",
                          "CUDA_VISIBLE_DEVICES"}) {
        kv.emplace_back(std::string("env_") + v, env(v));
    }
    return kv;
}

double load1() {
    std::ifstream in("/proc/loadavg");
    double x = -1;
    in >> x;
    return x;
}

std::string utc_now() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

#ifdef SSSP_HAVE_NVML_H

namespace {

// Loaded with dlopen, so machines without the NVIDIA driver can still build and run.
struct Nvml {
    bool ok = false;
    nvmlDevice_t dev{};
    decltype(&nvmlDeviceGetMaxClockInfo) max_clock = nullptr;
    decltype(&nvmlDeviceGetUtilizationRates) util = nullptr;
    decltype(&nvmlDeviceGetComputeRunningProcesses_v3) procs = nullptr;
    decltype(&nvmlDeviceGetMaxPcieLinkGeneration) pcie_gen_max = nullptr;
    decltype(&nvmlDeviceGetCurrPcieLinkWidth) pcie_width = nullptr;
    decltype(&nvmlDeviceGetPowerManagementLimit) power_limit = nullptr;
    decltype(&nvmlSystemGetDriverVersion) driver = nullptr;

    Nvml() {
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) return;
        auto load = [lib](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
        };
        decltype(&nvmlInit_v2) init = nullptr;
        decltype(&nvmlDeviceGetHandleByIndex_v2) handle = nullptr;
        load(init, "nvmlInit_v2");
        load(handle, "nvmlDeviceGetHandleByIndex_v2");
        load(max_clock, "nvmlDeviceGetMaxClockInfo");
        load(util, "nvmlDeviceGetUtilizationRates");
        load(procs, "nvmlDeviceGetComputeRunningProcesses_v3");
        load(pcie_gen_max, "nvmlDeviceGetMaxPcieLinkGeneration");
        load(pcie_width, "nvmlDeviceGetCurrPcieLinkWidth");
        load(power_limit, "nvmlDeviceGetPowerManagementLimit");
        load(driver, "nvmlSystemGetDriverVersion");
        if (!init || !handle || init() != NVML_SUCCESS) return;
        unsigned index = 0;
        std::string visible = env("CUDA_VISIBLE_DEVICES");
        if (!visible.empty() && std::isdigit(static_cast<unsigned char>(visible[0]))) {
            index = unsigned(std::atoi(visible.c_str()));
        }
        ok = handle(index, &dev) == NVML_SUCCESS;
    }
};

Nvml& nvml() {
    static Nvml n;
    return n;
}

}  // namespace

GpuSample gpu_sample() {
    GpuSample s;
    Nvml& n = nvml();
    if (!n.ok) return s;
    nvmlUtilization_t u{};
    if (n.util && n.util(n.dev, &u) == NVML_SUCCESS) s.util_pct = int(u.gpu);
    if (n.procs) {
        nvmlProcessInfo_t info[64];
        unsigned count = 64;
        if (n.procs(n.dev, &count, info) == NVML_SUCCESS) {
            int foreign = 0;
            unsigned self = unsigned(getpid());
            for (unsigned i = 0; i < count; ++i) foreign += info[i].pid != self;
            s.foreign_procs = foreign;
        }
    }
    return s;
}

KeyValues gpu_nvml_info() {
    KeyValues kv;
    Nvml& n = nvml();
    if (!n.ok) return kv;
    char buf[96] = {0};
    if (n.driver && n.driver(buf, sizeof(buf)) == NVML_SUCCESS) kv.emplace_back("nvidia_driver", buf);
    unsigned v = 0;
    if (n.max_clock && n.max_clock(n.dev, NVML_CLOCK_SM, &v) == NVML_SUCCESS)
        kv.emplace_back("gpu_sm_clock_max_mhz", std::to_string(v));
    if (n.max_clock && n.max_clock(n.dev, NVML_CLOCK_MEM, &v) == NVML_SUCCESS)
        kv.emplace_back("gpu_mem_clock_max_mhz", std::to_string(v));
    if (n.pcie_gen_max && n.pcie_gen_max(n.dev, &v) == NVML_SUCCESS)
        kv.emplace_back("pcie_gen_max", std::to_string(v));
    if (n.pcie_width && n.pcie_width(n.dev, &v) == NVML_SUCCESS)
        kv.emplace_back("pcie_width", std::to_string(v));
    if (n.power_limit && n.power_limit(n.dev, &v) == NVML_SUCCESS)
        kv.emplace_back("gpu_power_limit_w", std::to_string(v / 1000));
    return kv;
}

#else

GpuSample gpu_sample() { return {}; }
KeyValues gpu_nvml_info() { return {}; }

#endif

}  // namespace sssp
