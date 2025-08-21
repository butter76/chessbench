#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstdio>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tokenizer.hpp"
#include "parallel/thread_pool.hpp"
#include "parallel/nn_evaluator.hpp"

// Minimal coroutine Task type
struct Task {
    struct promise_type {
        Task get_return_object() noexcept {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }
        void return_void() noexcept {}
    };

    std::coroutine_handle<promise_type> coro;

    explicit Task(std::coroutine_handle<promise_type> h) : coro(h) {}
    Task(Task&& other) noexcept : coro(std::exchange(other.coro, {})) {}
    Task(const Task&) = delete;
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (coro) coro.destroy();
            coro = std::exchange(other.coro, {});
        }
        return *this;
    }
    ~Task() {
        if (coro) coro.destroy();
    }
};

// removed local ThreadPool; using engine_parallel::ThreadPool

using engine_parallel::NNEvaluator;
using engine_parallel::EvalAwaitable;
using engine_parallel::ThreadPool;

// PoC SearchAlgo that spawns multiple coroutines exploring a fake tree
class SearchAlgoPoC {
public:
    SearchAlgoPoC(ThreadPool& pool, NNEvaluator& eval)
        : pool(pool), evaluator(eval) {}

    void start() {
        stopped.store(false);
        thread = std::jthread([this](std::stop_token) { run(); });
    }

    void stop() {
        stopped.store(true);
    }

    void join() {
        if (thread.joinable()) thread.join();
    }

private:
    ThreadPool& pool;
    NNEvaluator& evaluator;
    std::atomic<bool> stopped{false};
    std::jthread thread;

    Task workerTask(int taskId) {
        int depthBudget = 20;
        std::mt19937 rng(taskId + 1337);
        std::uniform_int_distribution<int> dist(1, 5);
        for (int depth = 0; depth < depthBudget && !stopped.load(); ++depth) {
            int feature = dist(rng);
            // Occasionally request an evaluation and suspend
            if (depth % 3 == 0) {
                // Build a fake board and tokenize it
                chess::Board board;
                (void)feature; // in real code, mutate board per feature
                auto tokens = engine_tokenizer::tokenizeBoard(board);
                EvalAwaitable awaiter{&evaluator, &pool, tokens};
                auto res = co_await awaiter;
                if (res.canceled || stopped.load()) { std::cout << "task " << taskId << " canceled at depth " << depth << '\n'; break; }
                std::cout << "task " << taskId
                          << " resumed at depth " << depth
                          << " eval=" << res.value
                          << " thread=" << std::this_thread::get_id() << '\n';
            }
            // Do some local work
            for (volatile int spin = 0; spin < 1000; ++spin) {}
        }
        std::cout << "task " << taskId << " finished" << '\n';
        co_return;
    }

    void run() {
        // Spawn multiple coroutines onto the pool
        const int numTasks = 8;
        for (int i = 0; i < numTasks; ++i) {
            pool.start(workerTask(i));
        }
        // Keep the search thread alive while tasks run
        while (!stopped.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
};

int main() {
    std::cout << "PoC: ThreadPool + Coroutines + NNEvaluator" << std::endl;
    unsigned int n = std::max(2u, std::thread::hardware_concurrency());
    ThreadPool pool(n - 1); // leave one for evaluator
    NNEvaluator evaluator;
    evaluator.start();

    SearchAlgoPoC search(pool, evaluator);
    search.start();

    // Let it run briefly, then signal stop
    std::this_thread::sleep_for(std::chrono::seconds(1));
    search.stop();
    search.join();

    evaluator.stop_and_join();
    pool.shutdown();

    std::cout << "PoC complete" << std::endl;
    return 0;
}


