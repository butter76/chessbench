#pragma once

#include "time_handler.hpp"

namespace engine {

class FixedTime : public TimeHandler {
public:
	explicit FixedTime(unsigned long long fixed_ms) : fixed_ms_(fixed_ms) {}

	TimeBudget selectTimeBudget(const Limits & /*limits*/, chess::Color /*stm*/) const override {
		return TimeBudget{fixed_ms_, fixed_ms_};
	}

private:
	unsigned long long fixed_ms_;
};

} // namespace engine


