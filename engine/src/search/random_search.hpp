#pragma once

#include "search_algo.hpp"

#include <random>

namespace engine {

class RandomSearch : public SearchAlgo {
public:
    explicit RandomSearch(engine::Options &options,
                          const SyzygyHandler *tb = nullptr,
                          const engine::time::TimeHandler *timeHandler = nullptr,
                          unsigned long long seed = std::random_device{}())
        : SearchAlgo(options, tb, timeHandler), rng_(seed), board_() {}

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

    std::string searchBestMove(const Limits &limits) override {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        if (legal.empty()) return "0000";
        // Respect nodes: caller may bypass time management entirely if nodes > 0
        if (limits.nodes == 0 && timeHandler_) {
            (void)timeHandler_->computeTimeMs(limits); // placeholder for future thinking time usage
        }
        // If tablebases are available and the position is in TB, prefer any move
        // that keeps the maximum WDL (win > draw > loss).
        if (tb_) {
            auto wdlHere = tb_->probeWDL(board_);
            if (wdlHere.has_value()) {
                int bestWdl = -3;
                chess::Move bestMove = legal[0];
                for (const auto &m : legal) {
                    board_.makeMove(m);
                    auto w = tb_->probeWDL(board_);
                    // If child not in TB, treat as neutral 0 to avoid preferring unknowns
                    int val = w.has_value() ? *w : 0;
                    if (val > bestWdl) {
                        bestWdl = val;
                        bestMove = m;
                    }
                    board_.unmakeMove(m);
                }
                return chess::uci::moveToUci(bestMove);
            }
        }
        std::uniform_int_distribution<std::size_t> dist(0, legal.size() - 1);
        return chess::uci::moveToUci(legal[dist(rng_)]);
    }

private:
    std::mt19937_64 rng_;
    chess::Board board_;
};

} // namespace engine


