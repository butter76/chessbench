#pragma once

#include "search_algo.hpp"

#include <random>

namespace engine {

class RandomSearch : public SearchAlgo {
public:
    explicit RandomSearch(unsigned long long seed = std::random_device{}()) : rng_(seed) {}

    std::string searchBestMove(chess::Board &board, const Limits & /*limits*/) override {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        if (legal.empty()) return "0000";
        std::uniform_int_distribution<std::size_t> dist(0, legal.size() - 1);
        const chess::Move mv = legal[dist(rng_)];
        const std::string uci = chess::uci::moveToUci(mv);
        return uci;
    }

private:
    std::mt19937_64 rng_;
};

} // namespace engine


