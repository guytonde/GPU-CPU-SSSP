#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace sssp {

class Row {
public:
    Row& add(const std::string& col, const std::string& v);
    Row& add(const std::string& col, const char* v) { return add(col, std::string(v)); }
    Row& add(const std::string& col, double v);
    Row& add(const std::string& col, int64_t v);
    Row& add(const std::string& col, int v) { return add(col, int64_t(v)); }
    Row& add(const std::string& col, bool v) { return add(col, int64_t(v ? 1 : 0)); }
    const std::vector<std::pair<std::string, std::string>>& cells() const { return cells_; }

private:
    std::vector<std::pair<std::string, std::string>> cells_;
};

// Takes the header from the first row and throws if a later row differs.
// Flushes every row, so an interrupted run keeps what it wrote.
class CsvWriter {
public:
    explicit CsvWriter(const std::string& path);
    void write(const Row& row);

private:
    std::ofstream out_;
    std::vector<std::string> columns_;
    std::string path_;
};

std::string csv_escape(const std::string& v);

}  // namespace sssp
