#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace engine_parallel {

class ThreadPool {
public:
    explicit ThreadPool(unsigned int numThreads) : stopRequested(false) {
        if (numThreads == 0) numThreads = 1;
        workers.reserve(numThreads);
        for (unsigned int i = 0; i < numThreads; ++i) {
            workers.emplace_back(std::make_unique<Worker>());
        }
        for (unsigned int i = 0; i < numThreads; ++i) {
            threads.emplace_back([this, i] {
                workerLoop(i);
            });
        }
    }

    ~ThreadPool() { shutdown(); }

    void shutdown() {
        bool expected = false;
        if (stopRequested.compare_exchange_strong(expected, true)) {
            globalCv.notify_all();
            for (auto& w : workers) w->cv.notify_all();
            for (auto& t : threads) if (t.joinable()) t.join();
        }
    }

    void submit(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(globalMutex);
            globalQueue.push_back(std::move(fn));
        }
        globalCv.notify_one();
    }

    void resume(std::coroutine_handle<> h) {
        submit([h] {
            h.resume();
            if (h.done()) h.destroy();
        });
    }

    template <class TaskLike>
    void start(TaskLike&& t) {
        auto h = t.coro;
        t.coro = {};
        submit([h] {
            h.resume();
            if (h.done()) h.destroy();
        });
    }

private:
    struct Worker {
        std::deque<std::function<void()>> local;
        std::mutex mutex;
        std::condition_variable cv;
    };

    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::thread> threads;
    std::deque<std::function<void()>> globalQueue;
    std::mutex globalMutex;
    std::condition_variable globalCv;
    std::atomic<bool> stopRequested;

    bool tryPopLocal(unsigned int idx, std::function<void()>& out) {
        auto& w = *workers[idx];
        std::lock_guard<std::mutex> lock(w.mutex);
        if (!w.local.empty()) {
            out = std::move(w.local.front());
            w.local.pop_front();
            return true;
        }
        return false;
    }

    bool tryPopGlobal(std::function<void()>& out) {
        std::lock_guard<std::mutex> lock(globalMutex);
        if (!globalQueue.empty()) {
            out = std::move(globalQueue.front());
            globalQueue.pop_front();
            return true;
        }
        return false;
    }

    bool trySteal(unsigned int thiefIdx, std::function<void()>& out) {
        const unsigned int n = static_cast<unsigned int>(workers.size());
        for (unsigned int attempt = 0; attempt < n; ++attempt) {
            unsigned int victim = (thiefIdx + attempt + 1) % n;
            if (victim == thiefIdx) continue;
            auto& w = *workers[victim];
            std::lock_guard<std::mutex> lock(w.mutex);
            if (!w.local.empty()) {
                out = std::move(w.local.back());
                w.local.pop_back();
                return true;
            }
        }
        return false;
    }

    void workerLoop(unsigned int idx) {
        while (!stopRequested.load(std::memory_order_relaxed)) {
            std::function<void()> task;
            if (tryPopLocal(idx, task) || tryPopGlobal(task) || trySteal(idx, task)) {
                task();
                continue;
            }
            std::unique_lock<std::mutex> lock(globalMutex);
            globalCv.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
};

} // namespace engine_parallel


