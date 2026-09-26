#pragma once

#include <chrono>

namespace sssp {

// basically a wrapper for std::chrono::steady_clock to make it easier to time things in milliseconds
class Timer {
public:
    Timer() { reset(); }

    void reset() { start_ = std::chrono::steady_clock::now(); }

    double ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace sssp
