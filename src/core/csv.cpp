#include "sssp/csv.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace sssp {

std::string csv_escape(const std::string& v) {
    if (v.find_first_of(",\"\n") == std::string::npos) return v;
    std::string out = "\"";
    for (char c : v) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + '"';
}

Row& Row::add(const std::string& col, const std::string& v) {
    cells_.emplace_back(col, v);
    return *this;
}

Row& Row::add(const std::string& col, double v) {
    if (!std::isfinite(v)) return add(col, std::string());
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return add(col, std::string(buf));
}

Row& Row::add(const std::string& col, int64_t v) {
    return add(col, std::to_string(v));
}

CsvWriter::CsvWriter(const std::string& path) : out_(path), path_(path) {
    if (!out_) throw std::runtime_error("cannot write " + path);
}

void CsvWriter::write(const Row& row) {
    const auto& cells = row.cells();
    if (columns_.empty()) {
        for (size_t i = 0; i < cells.size(); ++i) {
            columns_.push_back(cells[i].first);
            out_ << (i ? "," : "") << csv_escape(cells[i].first);
        }
        out_ << "\n";
    }
    if (cells.size() != columns_.size()) {
        throw std::runtime_error(path_ + ": row has " + std::to_string(cells.size()) +
                                 " columns, header has " + std::to_string(columns_.size()));
    }
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].first != columns_[i]) {
            throw std::runtime_error(path_ + ": column " + cells[i].first +
                                     " where header has " + columns_[i]);
        }
        out_ << (i ? "," : "") << csv_escape(cells[i].second);
    }
    out_ << "\n";
    out_.flush();
}

}  // namespace sssp
