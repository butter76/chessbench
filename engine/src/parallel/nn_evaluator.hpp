#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "tokenizer.hpp"
#include "parallel/thread_pool.hpp"

namespace engine_parallel {

struct EvalResult {
    float value = 0.0f;
    bool canceled = false;
};

// Awaitable used by coroutines to request an evaluation
struct EvalAwaitable {
    struct Request;

    class PromiseLatch {
    public:
        void set(EvalResult r) {
            result.store(r.value, std::memory_order_release);
            canceled.store(r.canceled, std::memory_order_release);
        }
        EvalResult get() const {
            return EvalResult{result.load(std::memory_order_acquire), canceled.load(std::memory_order_acquire)};
        }
    private:
        std::atomic<float> result{0.0f};
        std::atomic<bool> canceled{false};
    };

    struct Request {
        std::array<std::uint8_t, 68> tokens;
        PromiseLatch* latch;
        std::coroutine_handle<> handle;
        ThreadPool* pool{nullptr};
    };

    class NNEvaluator* evaluator;
    ThreadPool* pool;
    std::array<std::uint8_t, 68> tokens;
    PromiseLatch latch;

    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h);
    EvalResult await_resume() const noexcept { return latch.get(); }
};

class NNEvaluator {
public:
    static constexpr std::size_t kBatchSize = 16;

    NNEvaluator() = default;

    void start() {
        stop.store(false, std::memory_order_release);
        worker = std::jthread([this](std::stop_token st) { run(st); });
    }

    void stop_and_join() {
        stop.store(true, std::memory_order_release);
        cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    void enqueue(EvalAwaitable::Request r) {
        if (stop.load(std::memory_order_acquire)) {
            r.latch->set(EvalResult{0.0f, true});
            if (r.pool) {
                r.pool->resume(r.handle);
            } else {
                r.handle.resume();
                if (r.handle.done()) r.handle.destroy();
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(std::move(r));
        }
        cv.notify_one();
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<EvalAwaitable::Request> queue;
    std::atomic<bool> stop{false};
    std::jthread worker;

    void run(std::stop_token) {
        for (;;) {
            std::vector<EvalAwaitable::Request> batch;
            batch.reserve(kBatchSize);
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]{ return stop.load() || !queue.empty(); });
                if (stop.load() && queue.empty()) break;
                while (!queue.empty() && batch.size() < kBatchSize) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop();
                }
            }

            if (stop.load(std::memory_order_acquire)) {
                for (auto& r : batch) {
                    r.latch->set(EvalResult{0.0f, true});
                    if (r.pool) {
                        r.pool->resume(r.handle);
                    } else {
                        r.handle.resume();
                        if (r.handle.done()) r.handle.destroy();
                    }
                }
                continue;
            }

            // Dummy batched evaluation: compute a simple function of tokens
            // e.g., count material-ish heuristic quickly
            for (auto& r : batch) {
                float score = 0.0f;
                for (std::uint8_t t : r.tokens) score += static_cast<float>(t) * 0.001f;
                r.latch->set(EvalResult{score, false});
            }
            for (auto& r : batch) {
                if (r.pool) {
                    r.pool->resume(r.handle);
                } else {
                    r.handle.resume();
                    if (r.handle.done()) r.handle.destroy();
                }
            }
        }
    }
};

inline void EvalAwaitable::await_suspend(std::coroutine_handle<> h) {
    evaluator->enqueue(Request{tokens, &latch, h, pool});
}

} // namespace engine_parallel


