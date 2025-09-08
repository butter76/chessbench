#pragma once

#include "search_algo.hpp"
#include "../parallel/nn_evaluator.hpp"
#include "../tokenizer.hpp"

#include <atomic>
#include <coroutine>
#include <future>
#include <limits>
#include <optional>
#include <utility>
#include <thread>
#include <iostream>

// cppcoro headers must be included at global scope
#include <cppcoro/task.hpp>
#include <cppcoro/sync_wait.hpp>

namespace engine {

class FixedDepthSearch : public SearchAlgo {
public:
    explicit FixedDepthSearch(engine::Options &options, const engine::TimeHandler *time_handler)
        : SearchAlgo(options, time_handler), board_() {
        evaluator_.start();
    }

    ~FixedDepthSearch() override {
        evaluator_.stop_and_join();
    }

    void reset() override { board_ = chess::Board(); }

    void makemove(const std::string &uci) override {
        const chess::Move move = chess::uci::uciToMove(board_, uci);
        if (move != chess::Move::NO_MOVE) {
            board_.makeMove(move);
        } else {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board_);
            for (const auto &m : legal) {
                if (chess::uci::moveToUci(m) == uci) { board_.makeMove(m); break; }
            }
        }
    }

    chess::Board &getBoard() override { return board_; }

    void stop() override { stop_requested_.store(true, std::memory_order_release); }

    std::string searchBestMove(const Limits &limits) override {
        stop_requested_.store(false, std::memory_order_release);
        const int maxDepth = limits.depth > 0 ? limits.depth : 2;

        chess::Movelist rootMoves;
        chess::movegen::legalmoves(rootMoves, board_);
        if (rootMoves.empty()) return "0000";

        chess::Move bestMove = chess::Move::NO_MOVE;
        float bestScore = -std::numeric_limits<float>::infinity();
        for (int depth = 1; depth <= maxDepth; ++depth) {
            if (stop_requested_.load(std::memory_order_acquire)) break;
            auto [score, move, aborted] = negamax_root(board_, depth);
            if (aborted) break;
            bestScore = score;
            if (move != chess::Move::NO_MOVE) bestMove = move;
        }

        if (bestMove == chess::Move::NO_MOVE) {
            // Fallback: choose first legal move if available
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board_);
            if (!legal.empty()) return chess::uci::moveToUci(legal[0]);
            return "0000";
        }
        return chess::uci::moveToUci(bestMove);
    }
    struct RootResult { float score; chess::Move pvMove; bool aborted; };

    RootResult negamax_root(const chess::Board& root, int depth) {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, root);
        if (moves.empty()) return {0.0f, chess::Move::NO_MOVE, false};

        struct SharedState {
            std::atomic<bool> aborted{false};
            std::mutex mtx;
            float best{-std::numeric_limits<float>::infinity()};
            chess::Move bestMove{chess::Move::NO_MOVE};
            std::atomic<int> remaining{0};
            std::promise<void> done;
        } shared;

        for (const auto& mv : moves) {
            chess::Board child = root;
            child.makeMove(mv);
            bool aborted_local = false;
            float score = -negamax(child, depth - 1, aborted_local);
            if (aborted_local || stop_requested_.load(std::memory_order_acquire)) {
                shared.aborted.store(true, std::memory_order_release);
                break;
            }
            {
                std::lock_guard<std::mutex> lock(shared.mtx);
                if (score > shared.best) { shared.best = score; shared.bestMove = mv; }
            }
        }
        if (shared.aborted.load(std::memory_order_acquire)) return {0.0f, shared.bestMove, true};
        return {shared.best, shared.bestMove, false};
    }

    float negamax(const chess::Board& node, int depth, bool& aborted) {
        if (stop_requested_.load(std::memory_order_acquire)) { aborted = true; return 0.0f; }
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, node);
        if (depth == 0 || moves.empty()) {
            auto val = evaluateBlocking(node);
            if (!val.has_value()) { aborted = true; return 0.0f; }
            return *val;
        }
        float best = -std::numeric_limits<float>::infinity();
        for (const auto& mv : moves) {
            if (stop_requested_.load(std::memory_order_acquire)) { aborted = true; return 0.0f; }
            chess::Board child = node;
            child.makeMove(mv);
            float score = -negamax(child, depth - 1, aborted);
            if (aborted) return 0.0f;
            if (score > best) best = score;
        }
        return best;
    }

private:
    chess::Board board_;
    std::atomic<bool> stop_requested_{false};
    engine_parallel::NNEvaluator evaluator_;
    

    std::optional<float> evaluateBlocking(const chess::Board &b) {
        if (stop_requested_.load(std::memory_order_acquire)) return std::nullopt;
        auto tokens = engine_tokenizer::tokenizeBoard(b);
        engine_parallel::EvalResult res;
        {
            // Simple synchronous bridge using a condition_variable on top of callback API
            std::mutex m; std::condition_variable cv; bool ready = false;
            evaluator_.enqueue(tokens, [&](engine_parallel::EvalResult r){
                std::lock_guard<std::mutex> lk(m); res = std::move(r); ready = true; cv.notify_one();
            });
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&]{ return ready; });
        }
        if (res.canceled || stop_requested_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return res.value;
    }
};

} // namespace engine


