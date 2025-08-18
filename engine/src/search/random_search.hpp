#pragma once

#include "search_algo.hpp"

#include <random>

namespace engine {

class RandomSearch : public SearchAlgo {
public:
    explicit RandomSearch(engine::Options &options,
                          unsigned long long seed = std::random_device{}())
        : SearchAlgo(options), rng_(seed), board_() {}

    void reset() override {
        board_ = chess::Board();
    }

    void makemove(const std::string &uci) override {
        const chess::Move move = chess::uci::uciToMove(board_, uci);
        if (move != chess::Move::NO_MOVE) {
            board_.makeMove(move);
        } else {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board_);
            for (const auto &m : legal) {
                if (chess::uci::moveToUci(m) == uci) {
                    board_.makeMove(m);
                    break;
                }
            }
        }
    }

    chess::Board &getBoard() override { return board_; }

    void stop() override { /* no-op for random */ }

    std::string searchBestMove(const Limits & /*limits*/) override {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        if (legal.empty()) return "0000";
        std::uniform_int_distribution<std::size_t> dist(0, legal.size() - 1);
        const chess::Move mv = legal[dist(rng_)];
        const std::string uci = chess::uci::moveToUci(mv);
        return uci;
    }

private:
    std::mt19937_64 rng_;
    chess::Board board_;
};

} // namespace engine


