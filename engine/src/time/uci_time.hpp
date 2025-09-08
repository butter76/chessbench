#pragma once

#include "time_handler.hpp"

namespace engine {

class UciTimeHandler : public TimeHandler {
public:
	TimeBudget selectTimeBudget(const Limits &limits, chess::Color side_to_move) const override {
		// Infinite or depth-limited only: no time cap from handler
		if (limits.infinite) return TimeBudget{0ULL, 0ULL};

		// go movetime X: obey strictly
		if (limits.movetime_ms > 0ULL) {
			unsigned long long hard = limits.movetime_ms;
			unsigned long long soft = (hard >= 10ULL) ? (hard - 5ULL) : hard; // finish current iter near hard
			return TimeBudget{soft, hard};
		}

		// Time controls (wtime/btime with optional inc and movestogo)
		unsigned long long time_left = (side_to_move == chess::Color::WHITE) ? limits.wtime_ms : limits.btime_ms;
		unsigned long long inc_ms    = (side_to_move == chess::Color::WHITE) ? limits.winc_ms  : limits.binc_ms;
		if (time_left == 0ULL) return TimeBudget{0ULL, 0ULL};

		// Move overhead (communication delay)
		const unsigned long long overhead = 50ULL; // 50ms overhead
		
		// Default moves to go if not specified
		int moves_to_go = limits.movestogo > 0 ? limits.movestogo : 40;
		
		// Calculate total time including future increments
		unsigned long long total_time_with_increments = time_left + (moves_to_go * inc_ms);
		
		// Calculate time after overhead
		unsigned long long time_after_overhead = (time_left > overhead) ? (time_left - overhead) : 0ULL;
		unsigned long long total_time_after_overhead = (total_time_with_increments > overhead) ? 
			(total_time_with_increments - overhead) : 0ULL;
		
		// Hard limit: minimum of 80% of (time_left - overhead) and 25% of (total_time_with_increments - overhead)
		unsigned long long hard_option1 = (time_after_overhead * 80ULL) / 100ULL; // 80% of (time_left - overhead)
		unsigned long long hard_option2 = (total_time_after_overhead * 25ULL) / 100ULL; // 25% of (total_time_with_increments - overhead)
		unsigned long long hard = std::min(hard_option1, hard_option2);
		
		// Soft limit: minimum of 5% of (time_left - overhead) and 1% of (total_time_with_increments - overhead)
		unsigned long long soft_option1 = (time_after_overhead * 5ULL) / 100ULL; // 5% of (time_left - overhead)
		unsigned long long soft_option2 = (total_time_after_overhead * 1ULL) / 100ULL; // 1% of (total_time_with_increments - overhead)
		unsigned long long soft = std::min(soft_option1, soft_option2);
		
		// Ensure soft doesn't exceed hard
		soft = std::min(soft, hard);
		
		// Ensure minimum values
		hard = std::max(hard, 1ULL);
		soft = std::max(soft, 1ULL);
		
		return TimeBudget{soft, hard};
	}
};

} // namespace engine


