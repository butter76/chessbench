#include "nn_evaluator.hpp"
#include "async_pool.hpp"

#include <chrono>

namespace poc {

NNEvaluatorMock::NNEvaluatorMock(WorkStealingPool* pool, std::size_t maxBatch, std::uint32_t usSleep)
    : pool_(pool), maxBatch_(maxBatch), usSleep_(usSleep) {}

NNEvaluatorMock::~NNEvaluatorMock() { stop(); join(); }

void NNEvaluatorMock::start() {
    stopping_.store(false, std::memory_order_relaxed);
    thread_ = std::jthread(&NNEvaluatorMock::run_loop, this);
}

void NNEvaluatorMock::stop() {
    bool expected = false;
    if (stopping_.compare_exchange_strong(expected, true)) {
        cv_.notify_all();
    }
}

void NNEvaluatorMock::join() {
    if (thread_.joinable()) thread_.join();
}

void NNEvaluatorMock::EvalAwaitable::await_suspend(Handle h) {
    // Record waiter and possibly enqueue new request
    std::scoped_lock lk(self->mu_);
    saved = h;
    auto [it, inserted] = self->inflight_.try_emplace(key);
    auto &w = it->second;
    const bool first_waiter = w.coros.empty();
    w.coros.push_back(h);
    if (first_waiter && !w.has_result) {
        self->batch_.push_back(key);
        self->cv_.notify_one();
    }
}

std::optional<int> NNEvaluatorMock::EvalAwaitable::await_resume() const noexcept {
    using Promise = poc::Task::promise_type;
    if (saved.promise().last_eval_canceled.load(std::memory_order_relaxed)) return std::nullopt;
    return saved.promise().last_eval_value.load(std::memory_order_relaxed);
}

void NNEvaluatorMock::run_loop() noexcept {
    using namespace std::chrono_literals;
    while (!stopping_.load(std::memory_order_relaxed)) {
        std::vector<int> todo;
        {
            std::unique_lock lk(mu_);
            cv_.wait_for(lk, std::chrono::microseconds(usSleep_), [&]{ return stopping_.load(std::memory_order_relaxed) || !batch_.empty(); });
            if (stopping_.load(std::memory_order_relaxed)) break;
            if (batch_.empty()) continue;
            // form batch up to maxBatch_
            const std::size_t n = std::min(batch_.size(), maxBatch_);
            todo.insert(todo.end(), batch_.begin(), batch_.begin() + n);
            batch_.erase(batch_.begin(), batch_.begin() + n);
        }

        // "Run inference": square the key as a fake result
        for (int key : todo) {
            int value = key * key;
            std::vector<Handle> waiters;
            {
                std::scoped_lock lk(mu_);
                auto it = inflight_.find(key);
                if (it == inflight_.end()) continue;
                it->second.result = value;
                it->second.has_result = true;
                waiters.swap(it->second.coros);
                inflight_.erase(it);
            }
            for (Handle h : waiters) {
                // Store result into promise if it matches our POC Task
                using Promise = poc::Task::promise_type;
                h.promise().last_eval_value.store(value, std::memory_order_relaxed);
                pool_->resume_later(h);
            }
        }
    }
    cancel_all_inflight();
}

void NNEvaluatorMock::cancel_all_inflight() {
    // Called after run loop exits to resume all waiters as canceled
    std::vector<std::pair<int, std::vector<Handle>>> all;
    {
        std::scoped_lock lk(mu_);
        for (auto &kv : inflight_) {
            all.emplace_back(kv.first, std::move(kv.second.coros));
        }
        inflight_.clear();
        batch_.clear();
    }
    for (auto &p : all) {
        for (Handle h : p.second) {
            h.promise().last_eval_canceled.store(true, std::memory_order_relaxed);
            pool_->resume_later(h);
        }
    }
}

} // namespace poc


