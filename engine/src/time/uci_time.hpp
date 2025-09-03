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

		int moves_to_go = limits.movestogo > 0 ? limits.movestogo : 30; // default horizon
		unsigned long long base_alloc = time_left / static_cast<unsigned long long>(moves_to_go);
		unsigned long long bonus = inc_ms / 2ULL;
		unsigned long long alloc = base_alloc + bonus;

		// Keep a reserve buffer
		unsigned long long reserve = std::max(50ULL, time_left / 20ULL); // at least 50ms or 5%
		if (alloc + reserve > time_left) alloc = (time_left > reserve) ? (time_left - reserve) : (time_left / 2ULL);

		unsigned long long hard = alloc;
		unsigned long long soft = (hard >= 10ULL) ? (hard - 5ULL) : hard;
		return TimeBudget{soft, hard};
	}
};

} // namespace engine


