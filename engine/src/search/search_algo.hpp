#pragma once

#include "chess.hpp"
#include "../options.hpp"
#include "../time/time_handler.hpp"

#include <string>
#include <vector>

namespace engine {

class SyzygyHandler; // forward declaration

struct Limits {
    bool infinite = false;
    unsigned long long nodes = 0;
    unsigned long long movetime_ms = 0;
    unsigned long long wtime_ms = 0;
    unsigned long long btime_ms = 0;
    unsigned long long winc_ms = 0;
    unsigned long long binc_ms = 0;
    int movestogo = 0;
    std::vector<std::string> searchmoves;
};

class SearchAlgo {
public:
    virtual ~SearchAlgo() = default;
    explicit SearchAlgo(engine::Options &options, const SyzygyHandler *tb = nullptr,
                        const engine::time::TimeHandler *timeHandler = nullptr)
        : options_(options), tb_(tb), timeHandler_(timeHandler) {}

    // Engine state management
    virtual void reset() = 0;                                   // set to startpos
    virtual void makemove(const std::string &uci) = 0;           // apply UCI move to internal board
    virtual chess::Board &getBoard() = 0;                        // access internal board
    virtual void stop() = 0;                                     // request to stop search if running

    // Search API operating on internal board
    virtual std::string searchBestMove(const Limits &limits) = 0;

protected:
    engine::Options &options_;
    const SyzygyHandler *tb_ = nullptr;
    const engine::time::TimeHandler *timeHandler_ = nullptr;
};

} // namespace engine


