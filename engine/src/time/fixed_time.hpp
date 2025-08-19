#pragma once

#include "time_handler.hpp"

namespace engine {

class FixedTime : public TimeHandler {
public:
	explicit FixedTime(unsigned long long fixed_ms) : fixed_ms_(fixed_ms) {}

	unsigned long long selectTimeMs(const Limits & /*limits*/) const override {
		return fixed_ms_;
	}

private:
	unsigned long long fixed_ms_;
};

} // namespace engine


