#pragma once

#include <cstdint>

#include "chess.hpp"

namespace engine {

// Forward declaration to avoid circular include with search headers
struct Limits;

class TimeHandler {
public:
	virtual ~TimeHandler() = default;

	struct TimeBudget {
		unsigned long long soft_ms{0}; // after this, finish current iteration and return
		unsigned long long hard_ms{0}; // after this, force stop()
	};

	// Returns soft/hard time budgets in milliseconds for the current side to move.
	// If limits indicate no time control, return {0,0} to mean "unbounded".
	virtual TimeBudget selectTimeBudget(const Limits &limits, chess::Color side_to_move) const = 0;
};

} // namespace engine


