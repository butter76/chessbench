#pragma once

#include "search_algo.hpp"
#include "../parallel/nn_evaluator.hpp"
#include "../parallel/thread_pool.hpp"
#include "../tokenizer.hpp"

#include <atomic>
#include <coroutine>
#include <future>
#include <limits>
#include <optional>
#include <utility>
#include <thread>
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

        if (!pool_) {
            unsigned int threads = std::max(2u, std::thread::hardware_concurrency());
            pool_ = std::make_unique<engine_parallel::ThreadPool>(threads - 1);
        }

        struct SharedState {
            std::atomic<bool> aborted{false};
            std::mutex mtx;
            float best{-std::numeric_limits<float>::infinity()};
            chess::Move bestMove{chess::Move::NO_MOVE};
            std::atomic<int> remaining{0};
            std::promise<void> done;
        } shared;

        shared.remaining.store(static_cast<int>(moves.size()));

        for (const auto& mv : moves) {
            chess::Board child = root;
            child.makeMove(mv);
            pool_->submit([this, child, depth, &shared, mv]() mutable {
                if (shared.aborted.load(std::memory_order_acquire)) {
                    if (shared.remaining.fetch_sub(1) == 1) shared.done.set_value();
                    return;
                }
                bool aborted_local = false;
                float score = -negamax(child, depth - 1, aborted_local);
                if (aborted_local || stop_requested_.load(std::memory_order_acquire)) {
                    shared.aborted.store(true, std::memory_order_release);
                    if (shared.remaining.fetch_sub(1) == 1) shared.done.set_value();
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(shared.mtx);
                    if (score > shared.best) { shared.best = score; shared.bestMove = mv; }
                }
                if (shared.remaining.fetch_sub(1) == 1) shared.done.set_value();
            });
        }

        shared.done.get_future().wait();
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
    std::unique_ptr<engine_parallel::ThreadPool> pool_;

    // coroutine to await eval then set promise
    TaskVoid evalTask(const std::array<std::uint8_t, 68> tokens, std::promise<engine_parallel::EvalResult> *p) {
        engine_parallel::EvalAwaitable awaitable{&evaluator_, pool_.get(), tokens};
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
        h.resume(); // will suspend into evaluator and resume on pool when ready
        engine_parallel::EvalResult res = fut.get();
        if (res.canceled || stop_requested_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return res.value;
    }
};

} // namespace engine


