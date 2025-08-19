// A small coroutine-friendly work-stealing pool used for proof-of-concept.
// It schedules std::coroutine_handle<> tasks across N worker threads.
// Tasks are expected to be coroutines with promise_type that can be resumed and destroyed when done.

#pragma once

#include <coroutine>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <vector>

namespace poc {

class WorkStealingPool {
public:
    explicit WorkStealingPool(std::size_t numThreads);
    ~WorkStealingPool();

    WorkStealingPool(const WorkStealingPool&) = delete;
    WorkStealingPool& operator=(const WorkStealingPool&) = delete;

    void start();
    void stop();
    void join();

    // Enqueue a coroutine to be run by the pool (round-robin distribution).
    void post(std::coroutine_handle<> handle);

    // Resume a coroutine with an externally prepared state (e.g., an NN result).
    // This is used by the evaluator thread: it must NOT call resume() directly.
    void resume_later(std::coroutine_handle<> handle);

    std::size_t num_workers() const noexcept { return workers_.size(); }

private:
    struct Worker {
        std::deque<std::coroutine_handle<>> deque;
        std::mutex dequeMutex;
        std::jthread thread;
        std::size_t index = 0;
    };

    std::vector<Worker> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<std::size_t> rrIndex_{0};

    // Global wake mechanism for idle workers
    std::mutex wakeMutex_;
    std::condition_variable_any wakeCv_;

    // Internal helpers
    static void worker_loop(WorkStealingPool* pool, std::size_t workerIndex) noexcept;
    bool try_pop_local(std::size_t workerIndex, std::coroutine_handle<>& out);
    bool try_steal(std::size_t thiefIndex, std::coroutine_handle<>& out);
    void push_local(std::size_t workerIndex, std::coroutine_handle<> handle);
};

// A simple detached Task<void> type for fire-and-forget coroutines.
struct Task {
    struct promise_type {
        Task get_return_object() noexcept { return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        void return_void() const noexcept {}
        void unhandled_exception() { std::terminate(); }
        // Storage for last async result (used by awaitables in the POC)
        // For the POC, we just keep an integer; real code would use a variant/struct
        std::atomic<int> last_eval_value{0};
        std::atomic<bool> last_eval_canceled{false};
    };

    std::coroutine_handle<promise_type> handle;
    explicit Task(std::coroutine_handle<promise_type> h) : handle(h) {}
    Task(Task&& other) noexcept : handle(other.handle) { other.handle = {}; }
    Task& operator=(Task&& other) noexcept { if (this != &other) { handle = other.handle; other.handle = {}; } return *this; }
    Task(const Task&) = delete; Task& operator=(const Task&) = delete;
    ~Task() { /* detached; pool destroys on completion */ }
};

// Helper to schedule a Task on the pool (detached).
inline void co_spawn(WorkStealingPool& pool, Task&& t) {
    if (t.handle) {
        pool.post(t.handle);
        t.handle = {};
    }
}

} // namespace poc


