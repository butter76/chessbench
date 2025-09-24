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
#include <functional>
#include <algorithm>
#include <array>

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

// Simplified callback-based request interface; no coroutine handles involved

class NNEvaluator {
public:
    // Runtime-configurable maximum batch size, set during initialize_trt()
    // Defaults to 48 if no option provided.
    std::size_t max_batch_size_{48};

    NNEvaluator() = default;

    explicit NNEvaluator(const engine::Options& options) : options_(&options) {}

    void start() {
        stop.store(false, std::memory_order_release);
        worker = std::jthread([this](std::stop_token st) { run(st); });
    }

    void stop_and_join() {
        stop.store(true, std::memory_order_release);
        cv.notify_all();
        if (worker.joinable()) worker.join();
        release_trt();
    }

    // Explicitly initialize TensorRT using configured or default plan path.
    // Safe to call multiple times; it will release any previous engine.
    void initialize_trt() {
        const std::string plan_path = resolve_plan_path();
        initialize_trt(plan_path.c_str());
    }

    void enqueue(const std::array<std::uint8_t, 68>& tokens,
                 std::function<void(EvalResult)> on_ready) {
        if (stop.load(std::memory_order_acquire)) {
            EvalResult er; er.value = 0.0f; er.canceled = true; on_ready(std::move(er));
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(Request{tokens, std::move(on_ready)});
        }
        cv.notify_one();
    }

    // Cancel all pending requests in the queue. In-flight work is not interrupted.
    void cancelQueue() {
        std::queue<Request> pending;
        {
            std::lock_guard<std::mutex> lock(mutex);
            std::swap(pending, queue);
        }
        while (!pending.empty()) {
            Request r = std::move(pending.front());
            pending.pop();
            EvalResult er; er.value = 0.0f; er.canceled = true;
            if (r.on_ready) r.on_ready(std::move(er));
        }
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    struct Request {
        std::array<std::uint8_t, 68> tokens;
        std::function<void(EvalResult)> on_ready;
    };
    std::queue<Request> queue;
    std::atomic<bool> stop{false};
    std::jthread worker;
    const engine::Options* options_{nullptr};

    // TensorRT objects
    nvinfer1::IRuntime* trt_runtime{nullptr};
    nvinfer1::ICudaEngine* trt_engine{nullptr};
    nvinfer1::IExecutionContext* trt_context{nullptr};
    // One engine/context per allowed batch size (r0_{B}.plan)
    std::vector<nvinfer1::ICudaEngine*> engines_{}; // index by B
    std::vector<nvinfer1::IExecutionContext*> contexts_{}; // index by B
    cudaStream_t trt_stream{};

    // CUDA Graph caching per batch size 1..max_batch_size_
    struct BatchGraph {
        int batchSize{0};
        bool valid{false};
        int profileIndex{-1};
        nvinfer1::IExecutionContext* context{nullptr};
        nvinfer1::ICudaEngine* engine{nullptr};
        nvinfer1::DataType dt_tokens{nvinfer1::DataType::kINT32};
        // Graph handles
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t graphExec{nullptr};
        // Device buffers
        void* d_tokens{nullptr};
        void* d_value{nullptr};
        void* d_hl{nullptr};
        void* d_hard{nullptr};
        void* d_U{nullptr};
        void* d_Q{nullptr};
        // Pinned host buffers (fixed address for capture)
        void* h_tokens{nullptr}; // int32_t* or int64_t* depending on tokens_dtype
        void* h_value{nullptr};  // __half* or float*, matches dt_value
        void* h_hl{nullptr};     // __half* or float*
        void* h_hard{nullptr};   // __half* or float*
        void* h_U{nullptr};      // __half* or float*
        void* h_Q{nullptr};      // __half* or float*
        // Sizes and dtypes
        size_t n_value{0}; nvinfer1::DataType dt_value{nvinfer1::DataType::kFLOAT};
        size_t n_hl{0};    nvinfer1::DataType dt_hl{nvinfer1::DataType::kFLOAT};
        size_t n_hard{0};  nvinfer1::DataType dt_hard{nvinfer1::DataType::kFLOAT};
        size_t n_U{0};     nvinfer1::DataType dt_U{nvinfer1::DataType::kFLOAT};
        size_t n_Q{0};     nvinfer1::DataType dt_Q{nvinfer1::DataType::kFLOAT};
        size_t tokensCount{0}; // number of scalar token elements (B*68)
        size_t tokensBytes{0};
        size_t bytes_value{0};
        size_t bytes_hl{0};
        size_t bytes_hard{0};
        size_t bytes_U{0};
        size_t bytes_Q{0};
    };
    std::vector<BatchGraph> graphs_; // index by batch size

    void release_one_graph(BatchGraph& g) {
        if (g.graphExec) { cudaGraphExecDestroy(g.graphExec); g.graphExec = nullptr; }
        if (g.graph) { cudaGraphDestroy(g.graph); g.graph = nullptr; }
        if (g.d_tokens) { cudaFree(g.d_tokens); g.d_tokens = nullptr; }
        if (g.d_value) { cudaFree(g.d_value); g.d_value = nullptr; }
        if (g.d_hl) { cudaFree(g.d_hl); g.d_hl = nullptr; }
        if (g.d_hard) { cudaFree(g.d_hard); g.d_hard = nullptr; }
        if (g.d_U) { cudaFree(g.d_U); g.d_U = nullptr; }
        if (g.d_Q) { cudaFree(g.d_Q); g.d_Q = nullptr; }
        if (g.h_tokens) { cudaFreeHost(g.h_tokens); g.h_tokens = nullptr; }
        if (g.h_value) { cudaFreeHost(g.h_value); g.h_value = nullptr; }
        if (g.h_hl) { cudaFreeHost(g.h_hl); g.h_hl = nullptr; }
        if (g.h_hard) { cudaFreeHost(g.h_hard); g.h_hard = nullptr; }
        if (g.h_U) { cudaFreeHost(g.h_U); g.h_U = nullptr; }
        if (g.h_Q) { cudaFreeHost(g.h_Q); g.h_Q = nullptr; }
        g.valid = false; g.batchSize = 0;
        g.n_value = g.n_hl = g.n_hard = g.n_U = g.n_Q = 0;
        g.tokensBytes = g.tokensCount = 0;
        g.bytes_value = g.bytes_hl = g.bytes_hard = g.bytes_U = g.bytes_Q = 0;
    }

    void release_graphs() {
        for (auto& g : graphs_) release_one_graph(g);
    }

    static inline int profile_index_for_batch(int B) {
        if (B >= 1 && B <= 16) return B - 1; // profiles 0..15
        if (B == 24) return 16;
        if (B == 32) return 17;
        if (B == 48) return 18;
        return -1;
    }

    bool build_graph_for_batch(int B, int profileIndex, nvinfer1::IExecutionContext* ctx, nvinfer1::ICudaEngine* engine) {
        if (B < 0) return false;
        if (static_cast<size_t>(B) >= graphs_.size()) return false;
        BatchGraph& G = graphs_[static_cast<size_t>(B)];
        release_one_graph(G);
        if (!ctx) return false;

        // Configure input shape for this batch
        nvinfer1::Dims inputDims; inputDims.nbDims = 2; inputDims.d[0] = B; inputDims.d[1] = 68;
        // For multi-engine fixed-shape plans, no profile switch needed; profileIndex kept for completeness
        if (!ctx->setInputShape("tokens", inputDims)) {
            std::cerr << "[NNEvaluator] setInputShape failed for B=" << B << "\n";
            return false;
        }

        // Allocate tokens buffers
        G.batchSize = B;
        G.profileIndex = profileIndex;
        G.context = ctx;
        G.engine = engine;
        // Determine tokens dtype for packing
        G.dt_tokens = engine ? engine->getTensorDataType("tokens") : nvinfer1::DataType::kINT32;
        G.tokensCount = static_cast<size_t>(B) * 68;
        G.tokensBytes = G.tokensCount * (G.dt_tokens == nvinfer1::DataType::kINT32 ? sizeof(int32_t) : sizeof(int64_t));
        if (cudaHostAlloc(&G.h_tokens, G.tokensBytes, cudaHostAllocPortable) != cudaSuccess) {
            std::cerr << "[NNEvaluator] cudaHostAlloc h_tokens failed\n";
            return false;
        }
        if (cudaMalloc(&G.d_tokens, G.tokensBytes) != cudaSuccess) {
            std::cerr << "[NNEvaluator] cudaMalloc d_tokens failed\n";
            return false;
        }
        ctx->setTensorAddress("tokens", G.d_tokens);

        // Helper to prepare an output binding
        auto prep_out = [&](const char* name, int bindingIndex, void** d_ptr, void** h_ptr, size_t& n_elem, size_t& n_bytes, nvinfer1::DataType& dt) {
            if (bindingIndex < 0) return;
            nvinfer1::Dims d = ctx->getTensorShape(name);
            n_elem = volume(d);
            dt = engine->getTensorDataType(name);
            n_bytes = n_elem * elementSize(dt);
            if (n_bytes == 0) return;
            if (cudaMalloc(d_ptr, n_bytes) != cudaSuccess) {
                std::cerr << "[NNEvaluator] cudaMalloc output failed for " << name << "\n";
                return;
            }
            if (cudaHostAlloc(h_ptr, n_bytes, cudaHostAllocPortable) != cudaSuccess) {
                std::cerr << "[NNEvaluator] cudaHostAlloc output failed for " << name << "\n";
                return;
            }
            ctx->setTensorAddress(name, *d_ptr);
        };

        // Outputs: prepare if present in this engine
        auto has_tensor = [&](const char* tname) -> bool {
            int nb = engine->getNbIOTensors();
            for (int i = 0; i < nb; ++i) {
                const char* nm = engine->getIOTensorName(i);
                if (nm && std::string(nm) == std::string(tname)) return true;
            }
            return false;
        };
        if (has_tensor("value")) prep_out("value", 0, &G.d_value, &G.h_value, G.n_value, G.bytes_value, G.dt_value);
        if (has_tensor("hl"))    prep_out("hl", 0, &G.d_hl, &G.h_hl, G.n_hl, G.bytes_hl, G.dt_hl);
        if (has_tensor("hardest_policy")) prep_out("hardest_policy", 0, &G.d_hard, &G.h_hard, G.n_hard, G.bytes_hard, G.dt_hard);
        if (has_tensor("U"))     prep_out("U", 0, &G.d_U, &G.h_U, G.n_U, G.bytes_U, G.dt_U);
        if (has_tensor("Q"))     prep_out("Q", 0, &G.d_Q, &G.h_Q, G.n_Q, G.bytes_Q, G.dt_Q);

        // Begin capture
        if (cudaStreamBeginCapture(trt_stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
            std::cerr << "[NNEvaluator] cudaStreamBeginCapture failed\n";
            return false;
        }
        // H2D tokens copy from pinned host
        cudaMemcpyAsync(G.d_tokens, G.h_tokens, G.tokensBytes, cudaMemcpyHostToDevice, trt_stream);
        // Enqueue inference
        ctx->setTensorAddress("tokens", G.d_tokens);
        if (G.d_value) ctx->setTensorAddress("value", G.d_value);
        if (G.d_hl) ctx->setTensorAddress("hl", G.d_hl);
        if (G.d_hard) ctx->setTensorAddress("hardest_policy", G.d_hard);
        if (G.d_U) ctx->setTensorAddress("U", G.d_U);
        if (G.d_Q) ctx->setTensorAddress("Q", G.d_Q);
        ctx->enqueueV3(trt_stream);
        // D2H copies into pinned host buffers
        if (G.d_value && G.h_value && G.bytes_value) cudaMemcpyAsync(G.h_value, G.d_value, G.bytes_value, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_hl && G.h_hl && G.bytes_hl) cudaMemcpyAsync(G.h_hl, G.d_hl, G.bytes_hl, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_hard && G.h_hard && G.bytes_hard) cudaMemcpyAsync(G.h_hard, G.d_hard, G.bytes_hard, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_U && G.h_U && G.bytes_U) cudaMemcpyAsync(G.h_U, G.d_U, G.bytes_U, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_Q && G.h_Q && G.bytes_Q) cudaMemcpyAsync(G.h_Q, G.d_Q, G.bytes_Q, cudaMemcpyDeviceToHost, trt_stream);

        if (cudaStreamEndCapture(trt_stream, &G.graph) != cudaSuccess || !G.graph) {
            std::cerr << "[NNEvaluator] cudaStreamEndCapture failed\n";
            release_one_graph(G);
            return false;
        }
        if (cudaGraphInstantiate(&G.graphExec, G.graph, 0) != cudaSuccess || !G.graphExec) {
            std::cerr << "[NNEvaluator] cudaGraphInstantiate failed\n";
            release_one_graph(G);
            return false;
        }

        G.valid = true;
        return true;
    }

    void initialize_graphs() {
        // Ensure stream available
        if (!trt_stream) return;
        // Prepare graphs_ storage for indices [0..max_batch_size_]
        graphs_.assign(max_batch_size_ + 1, BatchGraph{});

        // Allowed batch sizes mapping
        static constexpr int allowed_sizes[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,48};
        for (int B : allowed_sizes) {
            if (B > static_cast<int>(max_batch_size_)) continue;
            nvinfer1::IExecutionContext* ctx = (static_cast<size_t>(B) < contexts_.size()) ? contexts_[static_cast<size_t>(B)] : nullptr;
            nvinfer1::ICudaEngine* engine = (static_cast<size_t>(B) < engines_.size()) ? engines_[static_cast<size_t>(B)] : nullptr;
            if (!ctx || !engine) {
                graphs_[static_cast<size_t>(B)].valid = false;
                continue;
            }
            bool ok = build_graph_for_batch(B, /*profileIndex*/0, ctx, engine);
            if (!ok) {
                graphs_[static_cast<size_t>(B)].valid = false;
            }
        }
    }

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
        release_graphs();
        if (trt_context) { delete trt_context; trt_context = nullptr; }
        for (auto*& ctx : contexts_) { if (ctx) { delete ctx; ctx = nullptr; } }
        contexts_.clear();
        if (trt_engine) { delete trt_engine; trt_engine = nullptr; }
        if (trt_runtime) { delete trt_runtime; trt_runtime = nullptr; }
        if (trt_stream) { cudaStreamDestroy(trt_stream); trt_stream = nullptr; }
    }

    std::string resolve_plan_path() const {
        std::string path;
        if (options_) {
            // Try a few sensible option keys; stored in lowercase by Options
            const char* keys[] = {"network", "plan", "model", "trtplan"};
            for (const char* k : keys) {
                path = options_->get(k, "");
                if (!path.empty()) break;
            }
        }
        if (path.empty()) path = "./r1.plan";
        return path;
    }

    void initialize_trt(const char* plan_path) {
        release_trt();
        trt_runtime = nvinfer1::createInferRuntime(trt_logger);
        if (!trt_runtime) {
            std::cerr << "[NNEvaluator] Failed to create TensorRT runtime\n";
            return;
        }
        // Load per-batch engines r0_{B}.plan and create contexts
        static constexpr int allowed_sizes[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,48};
        engines_.assign(max_batch_size_ + 1, nullptr);
        contexts_.assign(max_batch_size_ + 1, nullptr);

        // Derive directory and base filename from provided plan_path
        std::string provided_path = plan_path ? std::string(plan_path) : std::string();
        std::string dir;
        std::string base;
        {
            const std::string& path = provided_path;
            std::size_t last_slash = path.find_last_of("/\\");
            dir = (last_slash == std::string::npos) ? "." : path.substr(0, last_slash);
            std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
            if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".plan") {
                base = filename.substr(0, filename.size() - 5);
            } else {
                base = filename;
            }
            if (base.empty()) base = "r1"; // sensible fallback
        }

        for (int B : allowed_sizes) {
            if (B > static_cast<int>(max_batch_size_)) continue;
            std::string bpath = dir + "/" + base + "_" + std::to_string(B) + ".plan";
            FILE* fB = fopen(bpath.c_str(), "rb");
            if (!fB) {
                // Mark missing; continue
                continue;
            }
            fseek(fB, 0, SEEK_END);
            long lenB = ftell(fB);
            fseek(fB, 0, SEEK_SET);
            std::vector<unsigned char> blobB(static_cast<size_t>(lenB));
            if (fread(blobB.data(), 1, blobB.size(), fB) != blobB.size()) {
                fclose(fB);
                continue;
            }
            fclose(fB);
            nvinfer1::ICudaEngine* eng = trt_runtime->deserializeCudaEngine(blobB.data(), blobB.size());
            if (!eng) {
                continue;
            }
            nvinfer1::IExecutionContext* ctx = eng->createExecutionContext();
            if (!ctx) {
                delete eng;
                continue;
            }
            engines_[static_cast<size_t>(B)] = eng;
            contexts_[static_cast<size_t>(B)] = ctx;
        }
        if (cudaStreamCreate(&trt_stream) != cudaSuccess) {
            std::cerr << "[NNEvaluator] Failed to create CUDA stream\n";
            release_trt();
            return;
        }
        std::cerr << "[NNEvaluator] TensorRT engines loaded (per-batch)\n";

        // Determine max batch size from options (only once)
        if (options_) {
            std::string bs = options_->get("batchsize", "");
            if (!bs.empty()) {
                try {
                    unsigned long parsed = std::stoul(bs);
                    if (parsed >= 1ul && parsed <= 1024ul) {
                        max_batch_size_ = static_cast<std::size_t>(parsed);
                    }
                } catch (...) {
                    // ignore invalid input, keep default
                    std::cerr << "[NNEvaluator] Invalid batch size option: " << bs << "\n";
                }
            }
        }

        // Initialize CUDA Graphs for allowed batch sizes
        initialize_graphs();
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
            std::vector<Request> batch;
            batch.reserve(max_batch_size_);
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]{ return stop.load() || !queue.empty(); });
                if (stop.load() && queue.empty()) break;
                // Choose the largest allowed batch size in {1..16,24,32,48} that fits 'available' and max_batch_size_
                std::size_t available = queue.size();
                std::size_t to_take = 0;
                static constexpr int allowed_sizes[] = {48, 32, 24, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
                for (int s : allowed_sizes) {
                    if (static_cast<std::size_t>(s) <= max_batch_size_ && available >= static_cast<std::size_t>(s)) {
                        to_take = static_cast<std::size_t>(s);
                        break;
                    }
                }
                while (!queue.empty() && batch.size() < to_take) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop();
                }
            }

            if (!batch.empty()) {
                if (is_batch_verbose()) {
                    std::cerr << "BATCHED " << batch.size() << " EVALS" << std::endl;
                }
            }

            if (stop.load(std::memory_order_acquire)) {
                for (auto& r : batch) {
                    EvalResult er; er.value = 0.0f; er.canceled = true; r.on_ready(std::move(er));
                }
                continue;
            }

            // Ensure TensorRT is initialized lazily using configured plan path
            if (contexts_.empty()) {
                const std::string plan_path = resolve_plan_path();
                initialize_trt(plan_path.c_str());
            }

            // If TensorRT per-batch engines are initialized, run inference via CUDA Graphs
            if (!contexts_.empty()) {
                const int B = static_cast<int>(batch.size());
                // Use CUDA Graph if available for this batch size
                if (B > 0 && B <= static_cast<int>(max_batch_size_) && static_cast<size_t>(B) < graphs_.size() && graphs_[static_cast<size_t>(B)].valid) {
                    BatchGraph& G = graphs_[static_cast<size_t>(B)];
                    // Pack tokens into pinned host input buffer
                    if (G.dt_tokens == nvinfer1::DataType::kINT32) {
                        int32_t* ht = reinterpret_cast<int32_t*>(G.h_tokens);
                        for (int i = 0; i < B; ++i) {
                            const auto& arr = batch[static_cast<size_t>(i)].tokens;
                            for (int j = 0; j < 68; ++j) ht[i * 68 + j] = static_cast<int32_t>(arr[j]);
                        }
                    } else {
                        int64_t* ht = reinterpret_cast<int64_t*>(G.h_tokens);
                        for (int i = 0; i < B; ++i) {
                            const auto& arr = batch[static_cast<size_t>(i)].tokens;
                            for (int j = 0; j < 68; ++j) ht[i * 68 + j] = static_cast<int64_t>(arr[j]);
                        }
                    }

                    // Launch captured graph
                    cudaGraphLaunch(G.graphExec, trt_stream);
                    cudaStreamSynchronize(trt_stream);

                    // Read outputs from pinned host buffers and dispatch
                    auto pinned_to_float = [&](void* h_ptr, size_t n_elem, nvinfer1::DataType dt) -> std::vector<float> {
                        std::vector<float> out;
                        if (!h_ptr || n_elem == 0) return out;
                        out.resize(n_elem);
                        if (dt == nvinfer1::DataType::kHALF) {
                            const __half* p = reinterpret_cast<const __half*>(h_ptr);
                            for (size_t i = 0; i < n_elem; ++i) out[i] = __half2float(p[i]);
                        } else if (dt == nvinfer1::DataType::kFLOAT) {
                            const float* p = reinterpret_cast<const float*>(h_ptr);
                            std::copy(p, p + n_elem, out.begin());
                        } else {
                            // Unsupported dtype for outputs
                            out.assign(n_elem, 0.0f);
                        }
                        return out;
                    };

                    std::vector<float> host_values = pinned_to_float(G.h_value, G.n_value, G.dt_value);
                    std::vector<float> host_hl    = pinned_to_float(G.h_hl,    G.n_hl,    G.dt_hl);
                    std::vector<float> host_hard  = pinned_to_float(G.h_hard,  G.n_hard,  G.dt_hard);
                    std::vector<float> host_U     = pinned_to_float(G.h_U,     G.n_U,     G.dt_U);
                    std::vector<float> host_Q     = pinned_to_float(G.h_Q,     G.n_Q,     G.dt_Q);

                    auto per_or_zero = [&](size_t total) -> size_t { return (B > 0 && total % static_cast<size_t>(B) == 0) ? (total / static_cast<size_t>(B)) : 0; };
                    const size_t per_hl = per_or_zero(G.n_hl);
                    const size_t per_hard = per_or_zero(G.n_hard);
                    const size_t per_U = per_or_zero(G.n_U);
                    const size_t per_Q = per_or_zero(G.n_Q);

                    for (int i = 0; i < B; ++i) {
                        float score = (G.n_value >= static_cast<size_t>((i + 1))) ? host_values[static_cast<size_t>(i)] : 0.0f;
                        if (is_batch_verbose()) {
                            std::cerr << tokensToString(batch[static_cast<size_t>(i)].tokens) << " => " << score << std::endl;
                        }
                        EvalResult er;
                        er.value = score;
                        er.canceled = false;
                        if (per_hl > 0) er.hl = std::vector<float>(host_hl.begin() + static_cast<std::ptrdiff_t>(i * per_hl), host_hl.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hl));
                        if (per_hard > 0) er.policy = std::vector<float>(host_hard.begin() + static_cast<std::ptrdiff_t>(i * per_hard), host_hard.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hard));
                        if (per_U > 0) er.U = std::vector<float>(host_U.begin() + static_cast<std::ptrdiff_t>(i * per_U), host_U.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_U));
                        if (per_Q > 0) er.Q = std::vector<float>(host_Q.begin() + static_cast<std::ptrdiff_t>(i * per_Q), host_Q.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_Q));
                        batch[static_cast<size_t>(i)].on_ready(std::move(er));
                    }
                } else {
                    // No graph for this B; cancel tasks gracefully
                    for (auto& r : batch) {
                        EvalResult er; er.value = 0.0f; er.canceled = true; r.on_ready(std::move(er));
                    }
                    continue;
                }
            } else {
                // No TensorRT available: cancel all requests and continue without throwing
                for (auto& r : batch) {
                    EvalResult er; er.value = 0.0f; er.canceled = true; r.on_ready(std::move(er));
                }
                continue;
            }
            // No explicit resumption here; callbacks have already notified waiters
        }
    }

    bool is_batch_verbose() const {
        if (!options_) return false;
        const std::string v = options_->get("batchverbose", "");
        return !v.empty() && v != "0" && v != "false" && v != "False";
    }
};

} // namespace engine_parallel


