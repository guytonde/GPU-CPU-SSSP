#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sssp/solver.hpp"

namespace sssp {

struct SolverEntry {
    std::string name;
    std::string device;
    std::unique_ptr<Solver> (*make)();
};

const std::vector<SolverEntry>& solver_table();
const SolverEntry* find_solver(const std::string& name);

}  // namespace sssp
