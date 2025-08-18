#pragma once

#include "chess.hpp"

#include <string>
#include <vector>

namespace engine {

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
    virtual std::string searchBestMove(chess::Board &board, const Limits &limits) = 0;
};

} // namespace engine


