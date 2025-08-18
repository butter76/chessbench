#include "fixed_time.hpp"
#include "../search/search_algo.hpp" // for engine::Limits

namespace engine::time {

std::uint64_t FixedTime::computeTimeMs(const engine::Limits &limits) const {
    // If nodes limit is set, time management is bypassed by caller; still return a value.
    if (limits.nodes > 0) return 0;
    return time_ms_;
}

} // namespace engine::time


