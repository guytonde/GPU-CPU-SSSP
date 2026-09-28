#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace sssp {

// The standard distributions and std::shuffle differ between implementations,
// mt19937_64 does not, so these are used instead.
inline uint64_t below(std::mt19937_64& rng, uint64_t bound) {
    unsigned __int128 p = static_cast<unsigned __int128>(rng()) * bound;
    return static_cast<uint64_t>(p >> 64);
}

inline double unit(std::mt19937_64& rng) {
    return double(rng() >> 11) * (1.0 / 9007199254740992.0);
}

template <typename T>
void shuffle(std::vector<T>& v, std::mt19937_64& rng) {
    for (size_t i = v.size(); i > 1; --i) {
        std::swap(v[i - 1], v[below(rng, i)]);
    }
}

}  // namespace sssp
