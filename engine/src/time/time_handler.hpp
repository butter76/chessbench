#pragma once

#include <cstdint>

namespace engine {

// Forward declaration to avoid circular include with search headers
struct Limits;

class TimeHandler {
public:
	virtual ~TimeHandler() = default;

	// Returns the time budget in milliseconds given the provided limits.
	virtual unsigned long long selectTimeMs(const Limits &limits) const = 0;
};

} // namespace engine


