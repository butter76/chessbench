#include "async_pool.hpp"
#include "nn_evaluator.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace poc {

// A trivial coroutine-based task that requests NN evals, simulating a tree search.
Task worker_task(WorkStealingPool& pool, NNEvaluatorMock& nn, int id, int requests) {
    std::mt19937 rng(id * 7919u);
    std::uniform_int_distribution<int> dist(1, 100);
    int accum = 0;
    for (int i = 0; i < requests; ++i) {
        int key = dist(rng);
        // Suspend here until evaluator provides the result; thread isn't blocked.
        auto res = co_await nn.await(key);
        if (!res.has_value()) {
            std::cout << "task " << id << " canceled while waiting on key=" << key << "\n";
            co_return;
        }
        int value = *res;
        accum += value % 13;
        if (i % 10 == 0) {
            std::cout << "task " << id << ": i=" << i << ", key=" << key << ", val=" << value << "\n";
        }
    }
    std::cout << "task " << id << " done, accum=" << accum << "\n";
}

} // namespace poc

int main() {
    poc::WorkStealingPool pool(std::max(2u, std::thread::hardware_concurrency()));
    pool.start();

    poc::NNEvaluatorMock nn(&pool, 16, 3000); // batch up to 16, ~3ms latency
    nn.start();

    const int num_tasks = 8;
    const int reqs = 50;
    for (int i = 0; i < num_tasks; ++i) {
        poc::co_spawn(pool, poc::worker_task(pool, nn, i, reqs));
    }

    // Let tasks run for a while
    std::this_thread::sleep_for(2s);

    nn.stop();
    nn.join();
    pool.stop();
    pool.join();
}


