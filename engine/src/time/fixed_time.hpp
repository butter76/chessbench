#pragma once

#include "time_handler.hpp"

namespace engine::time {

class FixedTime : public TimeHandler {
public:
    explicit FixedTime(std::uint64_t time_ms) : time_ms_(time_ms) {}

    std::uint64_t computeTimeMs(const engine::Limits &limits) const override;

private:
    std::uint64_t time_ms_;
};

} // namespace engine::time


