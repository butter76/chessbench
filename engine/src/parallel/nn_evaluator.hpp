#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <queue>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>

#include "tokenizer.hpp"
#include <cppcoro/task.hpp>
#include "options.hpp"

namespace engine_parallel {

struct EvalResult {
    float value = 0.0f;
    // Optional tensors returned by the network for one sample
    // Flat vectors in row-major order per sample
    std::vector<float> hl;       // e.g., logits over buckets
    std::vector<float> policy;   // e.g., move policy logits or probs
    std::vector<float> U;        // uncertainty head
    std::vector<float> Q;        // value head per-move or auxiliary
    bool canceled = false;
};

// Awaitable used by coroutines to request an evaluation
struct EvalAwaitable {
    struct Request;

    class PromiseLatch {
    public:
        void set(EvalResult r) {
            std::lock_guard<std::mutex> lock(mtx);
            result = std::move(r);
        }
        EvalResult get() const {
            std::lock_guard<std::mutex> lock(mtx);
            return result; // return by value
        }
    private:
        mutable std::mutex mtx;
        EvalResult result{};
    };

    struct Request {
        std::array<std::uint8_t, 68> tokens;
        PromiseLatch* latch;
        std::coroutine_handle<> handle;
    };

    class NNEvaluator* evaluator;
    std::array<std::uint8_t, 68> tokens;
    PromiseLatch latch;

    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h);
    EvalResult await_resume() const noexcept { return latch.get(); }
};

class NNEvaluator {
public:
    static constexpr std::size_t kBatchSize = 64;

    NNEvaluator() = default;

    explicit NNEvaluator(const engine::Options& options) : options_(&options) {}

    void start() {
        stop.store(false, std::memory_order_release);
        initialize_trt("./p2.plan");
        worker = std::jthread([this](std::stop_token st) { run(st); });
    }

    void stop_and_join() {
        stop.store(true, std::memory_order_release);
        cv.notify_all();
        if (worker.joinable()) worker.join();
        release_trt();
    }

    void enqueue(EvalAwaitable::Request r) {
        if (stop.load(std::memory_order_acquire)) {
            EvalResult er; er.value = 0.0f; er.canceled = true; r.latch->set(std::move(er));
            r.handle.resume();
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
    const engine::Options* options_{nullptr};

    // TensorRT objects
    nvinfer1::IRuntime* trt_runtime{nullptr};
    nvinfer1::ICudaEngine* trt_engine{nullptr};
    nvinfer1::IExecutionContext* trt_context{nullptr};
    cudaStream_t trt_stream{};

    // Simple TensorRT logger
    class TrtLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TRT] " << msg << '\n';
            }
        }
    };
    TrtLogger trt_logger;

    int binding_idx_tokens{-1};
    int binding_idx_value{-1};
    int binding_idx_hl{-1};
    int binding_idx_hardest{-1};
    int binding_idx_U{-1};
    int binding_idx_Q{-1};

    nvinfer1::DataType tokens_dtype{nvinfer1::DataType::kINT32};

    static inline size_t elementSize(nvinfer1::DataType t) {
        switch (t) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            case nvinfer1::DataType::kINT8: return 1;
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kINT64: return 8;
            case nvinfer1::DataType::kBOOL: return 1;
            default: return 0;
        }
    }

    static inline size_t volume(const nvinfer1::Dims& d) {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i] < 0 ? 1 : d.d[i]);
        return v;
    }

    void release_trt() {
        if (trt_context) { delete trt_context; trt_context = nullptr; }
        if (trt_engine) { delete trt_engine; trt_engine = nullptr; }
        if (trt_runtime) { delete trt_runtime; trt_runtime = nullptr; }
        if (trt_stream) { cudaStreamDestroy(trt_stream); trt_stream = nullptr; }
    }

    void initialize_trt(const char* plan_path) {
        release_trt();
        // Read plan file
        FILE* f = fopen(plan_path, "rb");
        if (!f) {
            std::cerr << "[NNEvaluator] Could not open plan file: " << plan_path << "\n";
            return;
        }
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<unsigned char> blob(static_cast<size_t>(len));
        if (fread(blob.data(), 1, blob.size(), f) != blob.size()) {
            std::cerr << "[NNEvaluator] Failed to read plan file: " << plan_path << "\n";
            fclose(f);
            return;
        }
        fclose(f);

        trt_runtime = nvinfer1::createInferRuntime(trt_logger);
        if (!trt_runtime) {
            std::cerr << "[NNEvaluator] Failed to create TensorRT runtime\n";
            return;
        }
        trt_engine = trt_runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!trt_engine) {
            std::cerr << "[NNEvaluator] Failed to deserialize engine\n";
            release_trt();
            return;
        }
        trt_context = trt_engine->createExecutionContext();
        if (!trt_context) {
            std::cerr << "[NNEvaluator] Failed to create execution context\n";
            release_trt();
            return;
        }
        if (cudaStreamCreate(&trt_stream) != cudaSuccess) {
            std::cerr << "[NNEvaluator] Failed to create CUDA stream\n";
            release_trt();
            return;
        }

        // Resolve bindings by name
        int nb = trt_engine->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = trt_engine->getIOTensorName(i);
            auto mode = trt_engine->getTensorIOMode(name);
            std::string n{name ? name : ""};
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (n == "tokens") {
                    binding_idx_tokens = i;
                    tokens_dtype = trt_engine->getTensorDataType(name);
                }
            } else {
                if (n == "value") binding_idx_value = i;
                if (n == "hl") binding_idx_hl = i;
                if (n == "hardest_policy") binding_idx_hardest = i;
                if (n == "U") binding_idx_U = i;
                if (n == "Q") binding_idx_Q = i;
            }
        }
        if (binding_idx_tokens < 0 || binding_idx_value < 0) {
            std::cerr << "[NNEvaluator] Required tensors not found (tokens/value)\n";
            release_trt();
            return;
        }
        std::cout << "[NNEvaluator] TensorRT engine loaded from " << plan_path << "\n";
    }

    static inline char tokenIndexToChar(std::uint8_t idx) {
        static constexpr char LUT[26] = {
            '0','1','2','3','4','5','6','7','8','9',
            'p','b','n','r','c','k','q',
            'P','B','N','R','C','Q','K',
            'x','.'
        };
        return idx < 26 ? LUT[idx] : '.';
    }

    static inline std::string tokensToString(const std::array<std::uint8_t, 68>& tokens) {
        std::string s;
        s.reserve(68);
        for (std::uint8_t t : tokens) {
            s.push_back(tokenIndexToChar(t));
        }
        return s;
    }

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

            if (!batch.empty()) {
                if (is_batch_verbose()) {
                    std::cout << "BATCHED " << batch.size() << " EVALS" << std::endl;
                }
            }

            if (stop.load(std::memory_order_acquire)) {
                for (auto& r : batch) {
                    EvalResult er; er.value = 0.0f; er.canceled = true; r.latch->set(std::move(er));
                    r.handle.resume();
                }
                continue;
            }

            // If TensorRT is initialized, run inference; otherwise fallback
            if (trt_context && binding_idx_tokens >= 0 && binding_idx_value >= 0) {
                const int B = static_cast<int>(batch.size());
                // Set input shape [B, 68]
                nvinfer1::Dims inputDims; inputDims.nbDims = 2; inputDims.d[0] = B; inputDims.d[1] = 68;
                if (!trt_context->setInputShape("tokens", inputDims)) {
                    std::cerr << "[NNEvaluator] Failed to set input shape" << std::endl;
                }

                // Host staging for tokens cast
                size_t tokensCount = static_cast<size_t>(B) * 68;
                std::vector<int32_t> tokens_i32;
                std::vector<int64_t> tokens_i64;
                void* d_tokens = nullptr;
                size_t tokensBytes = 0;
                if (tokens_dtype == nvinfer1::DataType::kINT32) {
                    tokens_i32.resize(tokensCount);
                    for (size_t i = 0; i < batch.size(); ++i) {
                        const auto& arr = batch[i].tokens;
                        for (int j = 0; j < 68; ++j) tokens_i32[i * 68 + j] = static_cast<int32_t>(arr[j]);
                    }
                    tokensBytes = tokens_i32.size() * sizeof(int32_t);
                    cudaMalloc(&d_tokens, tokensBytes);
                    cudaMemcpyAsync(d_tokens, tokens_i32.data(), tokensBytes, cudaMemcpyHostToDevice, trt_stream);
                } else {
                    tokens_i64.resize(tokensCount);
                    for (size_t i = 0; i < batch.size(); ++i) {
                        const auto& arr = batch[i].tokens;
                        for (int j = 0; j < 68; ++j) tokens_i64[i * 68 + j] = static_cast<int64_t>(arr[j]);
                    }
                    tokensBytes = tokens_i64.size() * sizeof(int64_t);
                    cudaMalloc(&d_tokens, tokensBytes);
                    cudaMemcpyAsync(d_tokens, tokens_i64.data(), tokensBytes, cudaMemcpyHostToDevice, trt_stream);
                }
                trt_context->setTensorAddress("tokens", d_tokens);

                // Prepare outputs; we need at least 'value', but provide all if present
                auto prepare_output = [&](const char* name, int bindingIndex, void** d_ptr, nvinfer1::DataType& dtype_out, size_t& elems) {
                    if (bindingIndex < 0) return;
                    nvinfer1::Dims d = trt_context->getTensorShape(name);
                    elems = volume(d);
                    dtype_out = trt_engine->getTensorDataType(name);
                    size_t bytes = elems * elementSize(dtype_out);
                    cudaMalloc(d_ptr, bytes);
                    trt_context->setTensorAddress(name, *d_ptr);
                };

                void* d_value = nullptr; size_t n_value = 0; nvinfer1::DataType dt_value = nvinfer1::DataType::kFLOAT;
                void* d_hl = nullptr; size_t n_hl = 0; nvinfer1::DataType dt_hl = nvinfer1::DataType::kFLOAT;
                void* d_hard = nullptr; size_t n_hard = 0; nvinfer1::DataType dt_hard = nvinfer1::DataType::kFLOAT;
                void* d_U = nullptr; size_t n_U = 0; nvinfer1::DataType dt_U = nvinfer1::DataType::kFLOAT;
                void* d_Q = nullptr; size_t n_Q = 0; nvinfer1::DataType dt_Q = nvinfer1::DataType::kFLOAT;

                prepare_output("value", binding_idx_value, &d_value, dt_value, n_value);
                if (binding_idx_hl >= 0) prepare_output("hl", binding_idx_hl, &d_hl, dt_hl, n_hl);
                if (binding_idx_hardest >= 0) prepare_output("hardest_policy", binding_idx_hardest, &d_hard, dt_hard, n_hard);
                if (binding_idx_U >= 0) prepare_output("U", binding_idx_U, &d_U, dt_U, n_U);
                if (binding_idx_Q >= 0) prepare_output("Q", binding_idx_Q, &d_Q, dt_Q, n_Q);

                // Execute
                if (!trt_context->enqueueV3(trt_stream)) {
                    std::cerr << "[NNEvaluator] enqueueV3 failed" << std::endl;
                    // Fail the batch safely
                    for (auto& r : batch) {
                        EvalResult er; er.value = 0.0f; er.canceled = true; r.latch->set(std::move(er));
                    }
                    // Skip device copies and resume continuations
                    for (auto& r : batch) {
                        r.handle.resume();
                    }
                    continue;
                }

                // Copy back outputs and distribute results
                std::vector<float> host_values(n_value);
                if (dt_value == nvinfer1::DataType::kHALF) {
                    std::vector<__half> tmp(n_value);
                    cudaMemcpyAsync(tmp.data(), d_value, n_value * sizeof(__half), cudaMemcpyDeviceToHost, trt_stream);
                    cudaStreamSynchronize(trt_stream);
                    for (size_t i = 0; i < n_value; ++i) host_values[i] = __half2float(tmp[i]);
                } else {
                    cudaMemcpyAsync(host_values.data(), d_value, n_value * sizeof(float), cudaMemcpyDeviceToHost, trt_stream);
                }
                cudaStreamSynchronize(trt_stream);

                auto copy_out_float = [&](void* d_ptr, size_t n_elem, nvinfer1::DataType dt) -> std::vector<float> {
                    std::vector<float> host;
                    if (!d_ptr || n_elem == 0) return host;
                    host.resize(n_elem);
                    if (dt == nvinfer1::DataType::kHALF) {
                        std::vector<__half> tmp(n_elem);
                        cudaMemcpyAsync(tmp.data(), d_ptr, n_elem * sizeof(__half), cudaMemcpyDeviceToHost, trt_stream);
                        cudaStreamSynchronize(trt_stream);
                        for (size_t i = 0; i < n_elem; ++i) host[i] = __half2float(tmp[i]);
                    } else {
                        cudaMemcpyAsync(host.data(), d_ptr, n_elem * sizeof(float), cudaMemcpyDeviceToHost, trt_stream);
                        cudaStreamSynchronize(trt_stream);
                    }
                    return host;
                };

                std::vector<float> host_hl = copy_out_float(d_hl, n_hl, dt_hl);
                std::vector<float> host_hard = copy_out_float(d_hard, n_hard, dt_hard);
                std::vector<float> host_U = copy_out_float(d_U, n_U, dt_U);
                std::vector<float> host_Q = copy_out_float(d_Q, n_Q, dt_Q);

                // Determine per-sample sizes (assume leading batch dim)
                auto per_or_zero = [&](size_t total) -> size_t { return (B > 0 && total % static_cast<size_t>(B) == 0) ? (total / static_cast<size_t>(B)) : 0; };
                const size_t per_hl = per_or_zero(n_hl);
                const size_t per_hard = per_or_zero(n_hard);
                const size_t per_U = per_or_zero(n_U);
                const size_t per_Q = per_or_zero(n_Q);

                for (int i = 0; i < B; ++i) {
                    float score = (n_value >= static_cast<size_t>((i + 1))) ? host_values[i] : 0.0f;
                    if (is_batch_verbose()) {
                        std::cout << tokensToString(batch[i].tokens) << " => " << score << std::endl;
                    }
                    EvalResult er;
                    er.value = score;
                    er.canceled = false;
                    if (per_hl > 0) er.hl = std::vector<float>(host_hl.begin() + static_cast<std::ptrdiff_t>(i * per_hl), host_hl.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hl));
                    if (per_hard > 0) er.policy = std::vector<float>(host_hard.begin() + static_cast<std::ptrdiff_t>(i * per_hard), host_hard.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hard));
                    if (per_U > 0) er.U = std::vector<float>(host_U.begin() + static_cast<std::ptrdiff_t>(i * per_U), host_U.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_U));
                    if (per_Q > 0) er.Q = std::vector<float>(host_Q.begin() + static_cast<std::ptrdiff_t>(i * per_Q), host_Q.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_Q));
                    batch[i].latch->set(std::move(er));
                }

                // Cleanup device buffers
                if (d_tokens) cudaFree(d_tokens);
                if (d_value) cudaFree(d_value);
                if (d_hl) cudaFree(d_hl);
                if (d_hard) cudaFree(d_hard);
                if (d_U) cudaFree(d_U);
                if (d_Q) cudaFree(d_Q);
            } else {
                // No TensorRT available: cancel all requests and continue without throwing
                for (auto& r : batch) {
                    EvalResult er; er.value = 0.0f; er.canceled = true; r.latch->set(std::move(er));
                    r.handle.resume();
                }
                continue;
            }
            for (auto& r : batch) {
                r.handle.resume();
            }
        }
    }

    bool is_batch_verbose() const {
        if (!options_) return false;
        const std::string v = options_->get("batchverbose", "");
        return !v.empty() && v != "0" && v != "false" && v != "False";
    }
};

inline void EvalAwaitable::await_suspend(std::coroutine_handle<> h) {
    evaluator->enqueue(Request{tokens, &latch, h});
}

} // namespace engine_parallel


