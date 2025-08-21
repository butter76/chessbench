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
#include <iostream>

namespace engine {

// Minimal coroutine Task used to bridge awaitable to blocking waiting
struct TaskVoid {
    struct promise_type {
        TaskVoid get_return_object() noexcept { return TaskVoid{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }
        void return_void() noexcept {}
    };
    std::coroutine_handle<promise_type> coro;
    explicit TaskVoid(std::coroutine_handle<promise_type> h) : coro(h) {}
    TaskVoid(TaskVoid&& other) noexcept : coro(std::exchange(other.coro, {})) {}
    TaskVoid(const TaskVoid&) = delete;
    TaskVoid& operator=(TaskVoid&& other) noexcept {
        if (this != &other) {
            if (coro) coro.destroy();
            coro = std::exchange(other.coro, {});
        }
        return *this;
    }
    ~TaskVoid() { if (coro) coro.destroy(); }
};

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
        (void)limits; // fixed-depth search ignores time for this PoC
        stop_requested_.store(false, std::memory_order_release);

        chess::Movelist rootMoves;
        chess::movegen::legalmoves(rootMoves, board_);
        if (rootMoves.empty()) return "0000";

        const chess::Color rootColor = board_.sideToMove();
        float bestScore = -std::numeric_limits<float>::infinity();
        chess::Move bestMove = chess::Move::NO_MOVE;
        bool aborted = false;

        for (const auto &mv : rootMoves) {
            if (stop_requested_.load(std::memory_order_acquire)) break;
            chess::Board child = board_;
            child.makeMove(mv);

            float worstReply = std::numeric_limits<float>::infinity();
            chess::Movelist replies;
            chess::movegen::legalmoves(replies, child);
            if (replies.empty()) {
                // No reply: terminal position; evaluate child directly
                auto val = evaluateBlocking(child);
                if (!val.has_value()) { aborted = true; break; }
                worstReply = -(*val); // opponent stuck -> good for us
            } else {
                for (const auto &rv : replies) {
                    if (stop_requested_.load(std::memory_order_acquire)) break;
                    chess::Board leaf = child;
                    leaf.makeMove(rv);
                    auto val = evaluateBlocking(leaf);
                    if (!val.has_value()) { aborted = true; break; }
                    float eval = *val;
                    // Opponent chooses the move minimizing our outcome
                    if (eval < worstReply) worstReply = eval;
                }
            }

            // We want to maximize our outcome from root's perspective.
            float ourScore = (rootColor == chess::Color::WHITE) ? -worstReply : -worstReply;
            if (ourScore > bestScore) { bestScore = ourScore; bestMove = mv; }
            if (aborted) break;
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

private:
    chess::Board board_;
    std::atomic<bool> stop_requested_{false};
    engine_parallel::NNEvaluator evaluator_;

    // coroutine to await eval then set promise
    TaskVoid evalTask(const std::array<std::uint8_t, 68> tokens, std::promise<engine_parallel::EvalResult> *p) {
        engine_parallel::EvalAwaitable awaitable{&evaluator_, nullptr, tokens};
        engine_parallel::EvalResult res = co_await awaitable;
        p->set_value(res);
        co_return;
    }

    std::optional<float> evaluateBlocking(const chess::Board &b) {
        if (stop_requested_.load(std::memory_order_acquire)) return std::nullopt;
        auto tokens = engine_tokenizer::tokenizeBoard(b);
        std::promise<engine_parallel::EvalResult> prom;
        std::future<engine_parallel::EvalResult> fut = prom.get_future();
        TaskVoid t = evalTask(tokens, &prom);
        auto h = t.coro;
        t.coro = {};
        h.resume(); // will suspend into evaluator and resume when ready; evaluator destroys handle when done
        engine_parallel::EvalResult res = fut.get();
        if (res.canceled || stop_requested_.load(std::memory_order_acquire)) {
            std::cout << "[eval] canceled fen=" << b.getFen(false) << '\n';
            return std::nullopt;
        }
        std::cout << "[eval] value=" << res.value << " fen=" << b.getFen(false) << '\n';
        return res.value;
    }
};

} // namespace engine


