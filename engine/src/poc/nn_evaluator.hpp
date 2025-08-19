#pragma once

#include <coroutine>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poc {

class WorkStealingPool;

// Simple mock evaluator that batches integer "keys" and returns key*key as the value
class NNEvaluatorMock {
public:
    using Handle = std::coroutine_handle<>;

    explicit NNEvaluatorMock(WorkStealingPool* pool, std::size_t maxBatch = 16, std::uint32_t usSleep = 2000);
    ~NNEvaluatorMock();

    NNEvaluatorMock(const NNEvaluatorMock&) = delete;
    NNEvaluatorMock& operator=(const NNEvaluatorMock&) = delete;

    void start();
    void stop();
    void join();

    // Awaitable for requesting an eval; result is placed into the awaiting coroutine's promise.last_eval_value
    struct EvalAwaitable {
        NNEvaluatorMock* self;
        int key;
        Handle saved{};
        bool await_ready() const noexcept { return false; }
        void await_suspend(Handle h);
        std::optional<int> await_resume() const noexcept; // nullopt means canceled
    };

    EvalAwaitable await(int key) { return EvalAwaitable{this, key}; }

private:
    struct Waiters {
        std::vector<Handle> coros;
        int result = 0;
        bool has_result = false;
    };

    WorkStealingPool* pool_;
    std::jthread thread_;
    std::atomic<bool> stopping_{false};
    std::mutex mu_;
    std::condition_variable_any cv_;
    std::vector<int> batch_;
    std::unordered_map<int, Waiters> inflight_;
    std::size_t maxBatch_;
    std::uint32_t usSleep_;

    void run_loop() noexcept;
    void cancel_all_inflight();
};

} // namespace poc


