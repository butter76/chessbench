#pragma once

#include "search_algo.hpp"
#include "../parallel/nn_evaluator.hpp"
#include "../parallel/thread_pool.hpp"
#include "../tokenizer.hpp"
#include "lks_node.hpp"

#include <atomic>
#include <coroutine>
#include <future>
#include <limits>
#include <optional>
#include <utility>
#include <thread>
#include <iostream>
#include <cmath>

namespace engine {

// Minimal coroutine Task used to bridge awaitable to blocking waiting
struct LksTaskVoid {
    struct promise_type {
        LksTaskVoid get_return_object() noexcept { return LksTaskVoid{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }
        void return_void() noexcept {}
    };
    std::coroutine_handle<promise_type> coro;
    explicit LksTaskVoid(std::coroutine_handle<promise_type> h) : coro(h) {}
    LksTaskVoid(LksTaskVoid&& other) noexcept : coro(std::exchange(other.coro, {})) {}
    LksTaskVoid(const LksTaskVoid&) = delete;
    LksTaskVoid& operator=(LksTaskVoid&& other) noexcept {
        if (this != &other) {
            if (coro) coro.destroy();
            coro = std::exchange(other.coro, {});
        }
        return *this;
    }
    ~LksTaskVoid() { if (coro) coro.destroy(); }
};

class LksSearch : public SearchAlgo {
public:
    explicit LksSearch(engine::Options &options, const engine::TimeHandler *time_handler)
        : SearchAlgo(options, time_handler), board_() {
        evaluator_.start();
    }

    ~LksSearch() override {
        evaluator_.stop_and_join();
    }

    void reset() override { board_ = chess::Board(); }

    void makemove(const std::string &uci) override {
        const chess::Move move = chess::uci::uciToMove(board_, uci);
        if (move != chess::Move::NO_MOVE) {
            board_.makeMove(move);
        } else {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board_);
            for (const auto &m : legal) {
                if (chess::uci::moveToUci(m) == uci) { board_.makeMove(m); break; }
            }
        }
    }

    chess::Board &getBoard() override { return board_; }

    void stop() override { stop_requested_.store(true, std::memory_order_release); }

    std::string searchBestMove(const Limits &limits) override {
        stop_requested_.store(false, std::memory_order_release);
        float maxDepth = (limits.depth > 0) ? static_cast<float>(limits.depth) : 2.0f;
        float currentDepth = std::min(2.0f, maxDepth);

        chess::Movelist rootMoves;
        chess::movegen::legalmoves(rootMoves, board_);
        if (rootMoves.empty()) return "0000";

        chess::Move bestMove = chess::Move::NO_MOVE;
        float bestScore = -std::numeric_limits<float>::infinity();
        while (currentDepth <= maxDepth + 1e-6f) {
            if (stop_requested_.load(std::memory_order_acquire)) break;
            auto [score, move, aborted] = lks_root(board_, currentDepth, -1.0f, 1.0f);
            if (aborted) break;
            bestScore = score;
            if (move != chess::Move::NO_MOVE) bestMove = move;
            currentDepth += 0.2f;
        }

        if (bestMove == chess::Move::NO_MOVE) {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board_);
            if (!legal.empty()) return chess::uci::moveToUci(legal[0]);
            return "0000";
        }
        return chess::uci::moveToUci(bestMove);
    }

    struct RootResult { float score; chess::Move pvMove; bool aborted; };

    enum class NodeType { PV, CUT, ALL };

    struct SearchOutcome { float score; chess::Move bestMove; bool aborted; };

    SearchOutcome pvs_search(const chess::Board& board, float depth, float alpha, float beta, NodeType node_type, bool want_move) {
        if (stop_requested_.load(std::memory_order_acquire)) return {0.0f, chess::Move::NO_MOVE, true};

        LKSNode node = create_node(board);
        if (node.terminal || depth <= 0.0f || node.policy.empty()) {
            return {node.value, chess::Move::NO_MOVE, false};
        }

        const float NULL_EPS = 1e-4f;
        const float depth_reduction = -2.0f * std::log(node.U + 1e-6f);

        auto is_leaf_node = [&](const LKSNode &n) {
            for (const auto &pe : n.policy) if (pe.child != nullptr) return false;
            return true;
        };

        float weight_divisor = 1.0f;
        int unexpanded_count = 0;
        float total_weight = 0.0f;
        if (is_leaf_node(node)) {
            for (std::size_t i = 0; i < node.policy.size(); ++i) {
                const auto &pe = node.policy[i];
                if (pe.child == nullptr) {
                    if ((total_weight > 0.80f && i >= 2) || (total_weight > 0.95f && i >= 1)) {
                        weight_divisor -= pe.policy;
                    } else {
                        unexpanded_count += 1;
                    }
                }
                total_weight += pe.policy;
            }
            if (depth <= std::log(static_cast<float>(std::max(0, unexpanded_count)) + 1e-6f) + depth_reduction) {
                return {node.value, chess::Move::NO_MOVE, false};
            }
        }

        float best_move_depth = -std::numeric_limits<float>::infinity();
        float bestScore = -std::numeric_limits<float>::infinity();
        chess::Move bestMove = chess::Move::NO_MOVE;
        total_weight = 0.0f;

        for (std::size_t i = 0; i < node.policy.size(); ++i) {
            auto &pe = node.policy[i];
            const float move_weight = pe.policy;
            float new_depth = depth + std::log(move_weight + 1e-6f) - std::log(weight_divisor + 1e-6f) - 0.1f;
            if (best_move_depth < -1e-6f) best_move_depth = new_depth;

            if (new_depth <= depth_reduction && pe.child == nullptr) {
                if ((total_weight > 0.80f && i >= 2) || (total_weight > 0.95f && i >= 1)) {
                    total_weight += move_weight;
                    continue;
                }
            }

            if (pe.child == nullptr) {
                chess::Board child_board = board;
                child_board.makeMove(pe.move);
                LKSNode child = create_node(child_board);
                arena_.push_back(std::make_unique<LKSNode>(std::move(child)));
                pe.child = arena_.back().get();
            }

            int child_re_searches = 0;
            const float RE_SEARCH_DEPTH = 0.2f;
            float score = -std::numeric_limits<float>::infinity();

            for (;;) {
                float search_alpha = (i == 0) ? -beta : -alpha - NULL_EPS;
                float search_beta = -alpha;
                NodeType next_type;
                if (i == 0 && node_type == NodeType::PV) next_type = NodeType::PV;
                else if (node_type == NodeType::CUT) next_type = NodeType::ALL;
                else next_type = NodeType::CUT;

                auto child_out = pvs_search(pe.child->board, new_depth, search_alpha, search_beta, next_type, false);
                if (child_out.aborted) return {0.0f, bestMove, true};
                score = -child_out.score;

                if (i > 0 && score > alpha) {
                    if (new_depth < best_move_depth) {
                        new_depth += RE_SEARCH_DEPTH;
                        if (new_depth > best_move_depth) new_depth = best_move_depth + RE_SEARCH_DEPTH;
                        child_re_searches += 1;
                        continue;
                    } else {
                        // Full-window re-search
                        child_out = pvs_search(pe.child->board, new_depth, -beta, -alpha, (node_type == NodeType::PV) ? NodeType::PV : next_type, false);
                        if (child_out.aborted) return {0.0f, bestMove, true};
                        score = -child_out.score;
                    }
                }
                if (new_depth > best_move_depth) best_move_depth = new_depth;
                break;
            }

            if (score > bestScore) { bestScore = score; if (want_move) bestMove = pe.move; }
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
            total_weight += move_weight;
        }
        return {bestScore, bestMove, false};
    }

    RootResult lks_root(const chess::Board& root, float depth, float alpha, float beta) {
        auto out = pvs_search(root, depth, alpha, beta, NodeType::PV, true);
        return {out.score, out.bestMove, out.aborted};
    }

    float lks(const chess::Board& board, float depth, float alpha, float beta, bool& aborted, NodeType node_type) {
        auto out = pvs_search(board, depth, alpha, beta, node_type, false);
        aborted = out.aborted;
        return out.score;
    }

private:
    chess::Board board_;
    std::atomic<bool> stop_requested_{false};
    engine_parallel::NNEvaluator evaluator_;
    std::unique_ptr<engine_parallel::ThreadPool> pool_;
    std::vector<std::unique_ptr<LKSNode>> arena_;

    // coroutine to await eval then set promise
    LksTaskVoid evalTask(const std::array<std::uint8_t, 68> tokens, std::promise<engine_parallel::EvalResult> *p) {
        engine_parallel::EvalAwaitable awaitable{&evaluator_, pool_.get(), tokens};
        engine_parallel::EvalResult res = co_await awaitable;
        p->set_value(res);
        co_return;
    }

    std::optional<engine_parallel::EvalResult> evaluateFullBlocking(const chess::Board &b) {
        if (stop_requested_.load(std::memory_order_acquire)) return std::nullopt;
        auto tokens = engine_tokenizer::tokenizeBoard(b);
        std::promise<engine_parallel::EvalResult> prom;
        std::future<engine_parallel::EvalResult> fut = prom.get_future();
        LksTaskVoid t = evalTask(tokens, &prom);
        auto h = t.coro;
        t.coro = {};
        h.resume();
        engine_parallel::EvalResult res = fut.get();
        if (res.canceled || stop_requested_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return res;
    }

    std::optional<float> evaluateBlocking(const chess::Board &b) {
        if (stop_requested_.load(std::memory_order_acquire)) return std::nullopt;
        auto tokens = engine_tokenizer::tokenizeBoard(b);
        std::promise<engine_parallel::EvalResult> prom;
        std::future<engine_parallel::EvalResult> fut = prom.get_future();
        LksTaskVoid t = evalTask(tokens, &prom);
        auto h = t.coro;
        t.coro = {};
        h.resume();
        engine_parallel::EvalResult res = fut.get();
        if (res.canceled || stop_requested_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return res.value;
    }

    static inline int fileCharToIndex(char f) { return static_cast<int>(f - 'a'); }
    static inline int rankCharToIndex(char r) { return static_cast<int>(r - '1'); }
    static inline int squareIndexFromFileRank(char file_c, char rank_c) {
        int file = fileCharToIndex(file_c);
        int rank = rankCharToIndex(rank_c);
        return rank * 8 + file;
    }
    static inline int mirrorSquareIndex(int sq) {
        int file = sq % 8;
        int rank = sq / 8;
        int mirroredRank = 7 - rank;
        return mirroredRank * 8 + file;
    }
    static inline int parseSquareOriented(const std::string &sq, bool flip) {
        // sq like "e2". If flip == false, mirror across horizontal axis; else normal
        int idx = squareIndexFromFileRank(sq[0], sq[1]);
        return flip ? idx : mirrorSquareIndex(idx);
    }
    static inline std::pair<int,int> move_to_indices(const chess::Move &mv, bool flip) {
        std::string uci = chess::uci::moveToUci(mv);
        int s1 = parseSquareOriented(uci.substr(0,2), flip);
        // promotion handling for r, b, n
        int s2;
        if (uci.size() == 5 && (uci[4] == 'r' || uci[4] == 'b' || uci[4] == 'n')) {
            char prom = uci[4];
            char src_file = uci[0];
            char dst_file = uci[2];
            int left_idx = -1, fwd_idx = -1, right_idx = -1;
            if (prom == 'r') { left_idx = mirrorSquareIndex(0); fwd_idx = 64; right_idx = mirrorSquareIndex(5); }
            else if (prom == 'b') { left_idx = mirrorSquareIndex(1); fwd_idx = 65; right_idx = mirrorSquareIndex(6); }
            else { left_idx = mirrorSquareIndex(2); fwd_idx = 66; right_idx = mirrorSquareIndex(7); }
            if (src_file == dst_file) s2 = fwd_idx;
            else if (src_file > dst_file) s2 = left_idx;
            else s2 = right_idx;
        } else {
            s2 = parseSquareOriented(uci.substr(2,2), flip);
        }
        return {s1, s2};
    }

public:
    // Create an LKS node from a board by evaluating model outputs
    LKSNode create_node(const chess::Board &board) {
        // Terminal detection (ignore syzygy and TT)
        const auto game_over = board.isGameOver();
        if (game_over.first != chess::GameResultReason::NONE) {
            float terminal_value = 0.0f;
            if (game_over.first == chess::GameResultReason::CHECKMATE) {
                // Side to move has no moves and is in check => current player loses
                terminal_value = -1.0f;
            } else {
                terminal_value = 0.0f; // stalemate, repetition, fifty-move, insufficient material
            }
            return LKSNode(board, terminal_value, {}, 0.0f, true);
        }

        // Evaluate network fully
        auto eval_opt = evaluateFullBlocking(board);
        if (!eval_opt.has_value()) {
            return LKSNode(board, 0.0f, {}, 0.0f, false);
        }
        engine_parallel::EvalResult eval = *eval_opt;

        // Convert scalar value to [-1, 1]
        float value = 2.0f * eval.value - 1.0f;

        // Prepare policy over legal moves using logits from eval.policy (hardest_policy)
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        const bool flip = (board.sideToMove() == chess::Color::BLACK);

        struct Gathered { chess::Move mv; float logit; float U; float Q; int idx; int s1; int s2; };
        std::vector<Gathered> gathered;
        gathered.reserve(legal.size());

        const int stride = 68;
        const int policy_size = static_cast<int>(eval.policy.size());
        const int U_size = static_cast<int>(eval.U.size());
        const int Q_size = static_cast<int>(eval.Q.size());

        for (const auto &mv : legal) {
            auto [s1, s2] = move_to_indices(mv, flip);
            long long flat = static_cast<long long>(s1) * stride + static_cast<long long>(s2);
            if (flat < 0 || flat >= policy_size) continue;
            float logit = eval.policy[static_cast<std::size_t>(flat)];
            float Um = (flat >= 0 && flat < U_size) ? eval.U[static_cast<std::size_t>(flat)] : 0.0f;
            float Qm = (flat >= 0 && flat < Q_size) ? eval.Q[static_cast<std::size_t>(flat)] : 0.0f;
            gathered.push_back(Gathered{mv, logit, Um, Qm, static_cast<int>(flat), s1, s2});
        }

        // Softmax over gathered logits
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto &g : gathered) max_logit = std::max(max_logit, g.logit);
        double sum_exp = 0.0;
        for (auto &g : gathered) { g.logit = std::exp(static_cast<double>(g.logit - max_logit)); sum_exp += g.logit; }
        if (sum_exp <= 0.0) sum_exp = 1.0;

        // Build policy entries, transform U/Q with sigmoid and Q from parent's perspective
        std::vector<LKSPolicyEntry> entries;
        entries.reserve(gathered.size());
        auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        for (const auto &g : gathered) {
            float prob = static_cast<float>(g.logit / sum_exp);
            float U_val = sigmoid(g.U);
            float Q_val = sigmoid(g.Q);
            Q_val = (Q_val * 2.0f - 1.0f) * -1.0f;
            LKSPolicyEntry e;
            e.move = g.mv;
            e.policy = prob;
            e.U = U_val;
            e.Q = Q_val;
            e.child = nullptr;
            entries.push_back(e);
        }
        std::sort(entries.begin(), entries.end(), [](const LKSPolicyEntry &a, const LKSPolicyEntry &b){ return a.policy > b.policy; });

        // Compute node-level U from hl logits as wdl_variance
        float node_U = 0.0f;
        const int bins = static_cast<int>(eval.hl.size());
        if (bins > 0) {
            // softmax over hl
            double max_hl = -std::numeric_limits<double>::infinity();
            for (float v : eval.hl) if (v > max_hl) max_hl = v;
            std::vector<double> probs(eval.hl.size());
            double sump = 0.0;
            for (std::size_t i = 0; i < eval.hl.size(); ++i) { probs[i] = std::exp(static_cast<double>(eval.hl[i]) - max_hl); sump += probs[i]; }
            if (sump <= 0.0) sump = 1.0;
            for (double &p : probs) p /= sump;

            // bin centers in [0,1]
            const double N = static_cast<double>(bins);
            double mean = 0.0, mean_sq = 0.0;
            for (int i = 0; i < bins; ++i) {
                double center = (2.0 * static_cast<double>(i) + 1.0) / (2.0 * N);
                mean += probs[i] * center;
                mean_sq += probs[i] * center * center;
            }
            double variance = std::max(0.0, mean_sq - mean * mean);
            node_U = static_cast<float>(std::sqrt(variance * 4.0));
        }

        return LKSNode(board, value, std::move(entries), node_U, false);
    }
};

} // namespace engine




