#pragma once

#include "time_handler.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace engine {

class StockTimeHandler : public TimeHandler {
public:
    StockTimeHandler() = default;

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
        unsigned long long inc_ms = (side_to_move == chess::Color::WHITE) ? limits.winc_ms : limits.binc_ms;
        if (time_left == 0ULL) return TimeBudget{0ULL, 0ULL};

        // Calculate time budget using Stockfish-style logic
        return calculateStockfishTimeBudget(time_left, inc_ms, limits.movestogo, side_to_move);
    }

private:
    TimeBudget calculateStockfishTimeBudget(unsigned long long time_left, 
                                          unsigned long long inc_ms, 
                                          int movestogo, 
                                          chess::Color side_to_move) const {
        // Move overhead (can be made configurable later)
        const unsigned long long move_overhead = 50ULL; // 50ms overhead
        
        // Maximum move horizon
        int centi_mtg = movestogo > 0 ? std::min(movestogo * 100, 5000) : 5051;
        
        // If less than one second, gradually reduce mtg
        if (time_left < 1000ULL) {
            centi_mtg = static_cast<int>(time_left * 5.051);
        }
        
        // Make sure timeLeft is > 0 since we may use it as a divisor
        unsigned long long adjusted_time_left = std::max(1ULL, 
            time_left + (inc_ms * (centi_mtg - 100) - move_overhead * (200 + centi_mtg)) / 100);
        
        double opt_scale, max_scale;
        
        // Calculate time constants based on current time left
        if (movestogo == 0) {
            // x basetime (+ z increment)
            double log_time_in_sec = std::log10(static_cast<double>(time_left) / 1000.0);
            double opt_constant = std::min(0.0032116 + 0.000321123 * log_time_in_sec, 0.00508017);
            double max_constant = std::max(3.3977 + 3.03950 * log_time_in_sec, 2.94761);
            
            // Use a default ply value (can be made configurable)
            int ply = 20; // Default middle game ply
            
            opt_scale = std::min(0.0121431 + std::pow(ply + 2.94693, 0.461073) * opt_constant,
                                0.213035 * static_cast<double>(time_left) / adjusted_time_left);
            
            max_scale = std::min(6.67704, max_constant + ply / 11.9847);
        } else {
            // x moves in y seconds (+ z increment)
            opt_scale = std::min((0.88 + 20.0 / 116.4) / (centi_mtg / 100.0), 
                                0.88 * static_cast<double>(time_left) / adjusted_time_left);
            max_scale = 1.3 + 0.11 * (centi_mtg / 100.0);
        }
        
        // Calculate optimum and maximum times
        unsigned long long optimum_time = static_cast<unsigned long long>(opt_scale * adjusted_time_left);
        unsigned long long maximum_time = static_cast<unsigned long long>(
            std::min(0.825179 * static_cast<double>(time_left) - static_cast<double>(move_overhead), 
                     max_scale * static_cast<double>(optimum_time))) - 10ULL;
        
        // Ensure we don't exceed available time
        maximum_time = std::min(maximum_time, time_left);
        optimum_time = std::min(optimum_time, maximum_time);
        
        // Ensure minimum values
        optimum_time = std::max(optimum_time, 1ULL);
        maximum_time = std::max(maximum_time, optimum_time);
        
        return TimeBudget{optimum_time, maximum_time};
    }
};

} // namespace engine
