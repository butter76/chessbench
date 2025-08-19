#include "async_pool.hpp"

#include <cassert>

namespace poc {

WorkStealingPool::WorkStealingPool(std::size_t numThreads) {
    if (numThreads == 0) numThreads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    workers_.resize(numThreads);
    for (std::size_t i = 0; i < numThreads; ++i) workers_[i].index = i;
}

WorkStealingPool::~WorkStealingPool() { stop(); join(); }

void WorkStealingPool::start() {
    stopping_.store(false, std::memory_order_relaxed);
    for (std::size_t i = 0; i < workers_.size(); ++i) {
        workers_[i].thread = std::jthread(&WorkStealingPool::worker_loop, this, i);
    }
}

void WorkStealingPool::stop() {
    bool expected = false;
    if (stopping_.compare_exchange_strong(expected, true)) {
        wakeCv_.notify_all();
    }
}

void WorkStealingPool::join() {
    for (auto &w : workers_) {
        if (w.thread.joinable()) w.thread.join();
    }
}

void WorkStealingPool::post(std::coroutine_handle<> handle) {
    const std::size_t idx = rrIndex_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    push_local(idx, handle);
}

void WorkStealingPool::resume_later(std::coroutine_handle<> handle) {
    // Use round-robin to place resumed continuations
    post(handle);
}

void WorkStealingPool::push_local(std::size_t workerIndex, std::coroutine_handle<> handle) {
    auto &w = workers_[workerIndex];
    {
        std::scoped_lock lk(w.dequeMutex);
        w.deque.push_back(handle);
    }
    wakeCv_.notify_one();
}

bool WorkStealingPool::try_pop_local(std::size_t workerIndex, std::coroutine_handle<>& out) {
    auto &w = workers_[workerIndex];
    std::scoped_lock lk(w.dequeMutex);
    if (w.deque.empty()) return false;
    out = w.deque.back();
    w.deque.pop_back();
    return true;
}

bool WorkStealingPool::try_steal(std::size_t thiefIndex, std::coroutine_handle<>& out) {
    for (std::size_t offset = 1; offset <= workers_.size(); ++offset) {
        std::size_t victim = (thiefIndex + offset) % workers_.size();
        auto &w = workers_[victim];
        std::scoped_lock lk(w.dequeMutex);
        if (!w.deque.empty()) {
            out = w.deque.front();
            w.deque.pop_front();
            return true;
        }
    }
    return false;
}

void WorkStealingPool::worker_loop(WorkStealingPool* pool, std::size_t workerIndex) noexcept {
    std::coroutine_handle<> task;
    while (!pool->stopping_.load(std::memory_order_relaxed)) {
        if (pool->try_pop_local(workerIndex, task)) {
            task.resume();
            if (task.done()) {
                task.destroy();
            } else {
                // If the coroutine suspended, it is responsible for re-enqueueing itself via awaitable or external source.
            }
            continue;
        }
        if (pool->try_steal(workerIndex, task)) {
            task.resume();
            if (task.done()) {
                task.destroy();
            }
            continue;
        }
        // Park until woken by new work
        std::unique_lock lk(pool->wakeMutex_);
        pool->wakeCv_.wait(lk);
    }
}

} // namespace poc


