#pragma once

#include <cstdint>

namespace engine {
struct Limits; // forward declaration from search_algo.hpp
}

namespace engine::time {

class TimeHandler {
public:
    virtual ~TimeHandler() = default;

    // Return the number of milliseconds to spend on the search given UCI go limits.
    virtual std::uint64_t computeTimeMs(const engine::Limits &limits) const = 0;
};

} // namespace engine::time


