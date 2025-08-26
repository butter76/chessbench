#pragma once

#include "search_algo.hpp"
#include "../parallel/nn_evaluator.hpp"
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
#include <memory>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <cppcoro/task.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/static_thread_pool.hpp>

namespace engine {

// Using cppcoro::task for async operations

class LksSearch : public SearchAlgo {
public:
    explicit LksSearch(engine::Options &options, const engine::TimeHandler *time_handler)
        : SearchAlgo(options, time_handler), board_(), evaluator_(options),
          pool_(static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()))) {
        evaluator_.start();
    }

    ~LksSearch() override {
        evaluator_.stop_and_join();
    }

    void reset() override { board_ = chess::Board(); root_.reset(); }

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
        root_.reset();
    }

    chess::Board &getBoard() override { return board_; }

    void stop() override { stop_requested_.store(true, std::memory_order_release); }

    std::string searchBestMove(const Limits &limits) override {
        stop_requested_.store(false, std::memory_order_release);
        // Reset per-search statistics
        stat_gpu_evaluations_.store(0, std::memory_order_relaxed);
        stat_nodes_created_.store(0, std::memory_order_relaxed);
        stat_tbhits_.store(0, std::memory_order_relaxed);
        stat_tthits_.store(0, std::memory_order_relaxed);
        stat_seldepth_.store(0, std::memory_order_relaxed);
        // Mark search start time
        {
            using namespace std::chrono;
            const std::int64_t now_ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
            stat_search_start_ns_.store(now_ns, std::memory_order_relaxed);
        }
        float maxDepth = (limits.depth > 0) ? static_cast<float>(limits.depth) : 30.0f;
        float currentDepth = std::min(2.0f, maxDepth);

        chess::Movelist rootMoves;
        chess::movegen::legalmoves(rootMoves, board_);
        if (rootMoves.empty()) return "0000";

        chess::Move bestMove = chess::Move::NO_MOVE;
        float bestScore = -std::numeric_limits<float>::infinity();
        if (!root_) {
            LKSNode rootNode = cppcoro::sync_wait(create_node(board_));
            root_ = std::make_unique<LKSNode>(std::move(rootNode));
        }
        while (currentDepth <= maxDepth + 1e-2f) {
            if (stop_requested_.load(std::memory_order_acquire)) break;
            if (!node_limit_check(limits)) break;
            auto [score, move, aborted] = cppcoro::sync_wait([&]() -> cppcoro::task<RootResult> {
                co_await pool_.schedule();
                co_return co_await lks_root(*root_, currentDepth, -1.0f, 1.0f);
            }());
            if (aborted) break;
            bestScore = score;
            if (move != chess::Move::NO_MOVE) bestMove = move;
            // Emit UCI info line for this iteration
            print_info_line(currentDepth, bestScore);
            currentDepth += 0.2f;
        }

        if (bestMove == chess::Move::NO_MOVE) {
            if (root_ && !root_->policy.empty()) {
                return chess::uci::moveToUci(root_->policy[0].move);
            }
            return "0000";
        }
        return chess::uci::moveToUci(bestMove);
    }

    // Statistics API (thread-safe)
    std::uint64_t getGpuEvaluationsCount() const {
        return stat_gpu_evaluations_.load(std::memory_order_relaxed);
    }

    std::uint64_t getNodesCreatedCount() const {
        return stat_nodes_created_.load(std::memory_order_relaxed);
    }

    int getSelDepth() const {
        return stat_seldepth_.load(std::memory_order_relaxed);
    }

    std::uint64_t getTBHitsCount() const {
        return stat_tbhits_.load(std::memory_order_relaxed);
    }

    std::uint64_t getTTHitsCount() const {
        return stat_tthits_.load(std::memory_order_relaxed);
    }

    std::uint64_t getElapsedTimeMs() const {
        using namespace std::chrono;
        const std::int64_t start_ns = stat_search_start_ns_.load(std::memory_order_relaxed);
        if (start_ns <= 0) return 0;
        const std::int64_t now_ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
        const std::int64_t delta_ns = now_ns - start_ns;
        if (delta_ns <= 0) return 0;
        return static_cast<std::uint64_t>(delta_ns / 1000000);
    }

    struct RootResult { float score; chess::Move pvMove; bool aborted; };

    enum class NodeType { PV, CUT, ALL };

    struct SearchOutcome { float score; chess::Move bestMove; bool aborted; };

    cppcoro::task<SearchOutcome> lks_search(LKSNode& node, float depth, float alpha, float beta, NodeType node_type, int rec_depth = 0) {
        if (stop_requested_.load(std::memory_order_acquire)) co_return SearchOutcome{0.0f, chess::Move::NO_MOVE, true};

        if (node.terminal || node.policy.empty()) {
            co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
        }

        if (rec_depth > 50) {
            co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
        }

        // Ensure policy is sorted and normalized before expansion
        sort_and_normalize(node);

        const float NULL_EPS = 1e-4f;
        const float node_depth_reduction = -2.0f * std::log(node.U + 1e-6f);

        auto is_leaf_node = [&](const LKSNode &n) {
            for (const auto &pe : n.policy) if (pe.child) return false;
            return true;
        };

        float weight_divisor = 1.0f;
        int unexpanded_count = 0;
        float total_weight_scan = 0.0f;
        for (std::size_t i = 0; i < node.policy.size(); ++i) {
            const auto &pe = node.policy[i];
            if (pe.child == nullptr) {
                if ((total_weight_scan > 0.80f && i >= 2) || (total_weight_scan > 0.95f && i >= 1)) {
                    weight_divisor -= pe.policy;
                } else {
                    unexpanded_count += 1;
                }
            }
            total_weight_scan += pe.policy;
        }
        if (is_leaf_node(node)) {
            if (depth <= std::log(static_cast<float>(std::max(0, unexpanded_count)) + 1e-6f) + node_depth_reduction) {
                co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
            }

            // First expansion bookkeeping: update maximum selective depth
            const int new_seldepth = rec_depth + 1;
            int observed = stat_seldepth_.load(std::memory_order_relaxed);
            if (new_seldepth > observed) {
                while (!stat_seldepth_.compare_exchange_weak(observed, new_seldepth, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    if (new_seldepth <= observed) break;
                }
            }
        }

        // Phase 1: Identify the set of children that are part of the search tree according to the depth
        const std::size_t policy_size = node.policy.size();
        std::vector<std::size_t> filtered_indices;
        filtered_indices.reserve(policy_size);
        std::vector<float> new_depths(policy_size, 0.0f);

        float best_move_depth = -std::numeric_limits<float>::infinity();
        float bestScore = -std::numeric_limits<float>::infinity();
        chess::Move bestMove = chess::Move::NO_MOVE;

        float total_weight = 0.0f; // running total for filtering logic
        for (std::size_t i = 0; i < policy_size; ++i) {
            auto &pe = node.policy[i];
            const float move_weight = pe.policy;
            float new_depth = depth + std::log(move_weight + 1e-6f) - std::log(weight_divisor + 1e-6f) - 0.1f;
            if (i == 0) best_move_depth = new_depth;

            new_depths[i] = new_depth;

            bool should_filter = false;
            if (!pe.child) {
                const float local_reduction = -2.0f * std::log(pe.U + 1e-6f);
                if (new_depth <= local_reduction) {
                    if ((total_weight > 0.80f && i >= 2) || (total_weight > 0.95f && i >= 1)) {
                        should_filter = true;
                    }
                }
            }
            if (!should_filter) filtered_indices.push_back(i);
            total_weight += move_weight;
        }

        // Phase 2: for each filtered child, ensure child exists, run initial search, collect improvers (i>0)
        const float RE_SEARCH_DEPTH = 0.2f;
        std::vector<std::size_t> improver_indices;
        improver_indices.reserve(filtered_indices.size());

        for (std::size_t idx_pos = 0; idx_pos < filtered_indices.size(); ++idx_pos) {
            std::size_t i = filtered_indices[idx_pos];
            auto &pe = node.policy[i];

            // Create child and backprop if needed
            if (!pe.child) {
                chess::Board child_board = node.board;
                child_board.makeMove(pe.move);
                LKSNode created = co_await create_node(child_board);
                pe.child = std::make_unique<LKSNode>(std::move(created));
                backpropagate_policy_updates(node, *pe.child, pe.move);
            }

            float new_depth = new_depths[i];
            float search_alpha = (i == 0) ? -beta : -alpha - NULL_EPS;
            float search_beta = -alpha;
            NodeType next_type;
            if (i == 0 && node_type == NodeType::PV) next_type = NodeType::PV;
            else if (node_type == NodeType::CUT) next_type = NodeType::ALL;
            else next_type = NodeType::CUT;

            auto child_out = co_await lks_search(*pe.child, new_depth, search_alpha, search_beta, next_type, rec_depth + 1);
            if (child_out.aborted) co_return SearchOutcome{0.0f, bestMove, true};
            float score = -child_out.score;

            // Use the first move to seed initial values
            if (i == 0) {
                if (score > alpha) alpha = score;
                bestScore = score;
                bestMove = pe.move;
                if (score >= beta) {
                    co_return SearchOutcome{bestScore, bestMove, false};
                }
            } else {
                if (score > alpha) {
                    improver_indices.push_back(i);
                }
            }
        }

        // Phase 3: re-search improvers (non-first moves)
        for (std::size_t k = 0; k < improver_indices.size(); ++k) {
            std::size_t i = improver_indices[k];
            auto &pe = node.policy[i];
            float new_depth = new_depths[i];
            float score = std::numeric_limits<float>::infinity();
            int re_search_count = 0;

            // Continue incremental null-window searches until reaching best_move_depth
            while (score > alpha && new_depth < best_move_depth) {
                // Increment depth
                new_depth += RE_SEARCH_DEPTH;
                if (new_depth >= best_move_depth) new_depth = best_move_depth + RE_SEARCH_DEPTH;
                re_search_count += 1;

                NodeType next_type = (node_type == NodeType::CUT) ? NodeType::ALL : NodeType::CUT;
                auto child_out = co_await lks_search(*pe.child, new_depth, -alpha - NULL_EPS, -alpha, next_type, rec_depth + 1);
                if (child_out.aborted) co_return SearchOutcome{0.0f, bestMove, true};
                score = -child_out.score;
            }

            // If still improving alpha, do full-window re-search
            if (score > alpha) {
                NodeType next_type = (node_type == NodeType::CUT) ? NodeType::ALL : NodeType::CUT;
                NodeType fw_type = (node_type == NodeType::PV) ? NodeType::PV : next_type;
                auto child_out = co_await lks_search(*pe.child, new_depth, -beta, -alpha, fw_type, rec_depth + 1);
                if (child_out.aborted) co_return SearchOutcome{0.0f, bestMove, true};
                score = -child_out.score;
            }

            // Update policy
            float new_policy = pe.policy;
            if (node_type == NodeType::CUT && score > alpha) {
                new_policy = pe.policy * std::exp(re_search_count * RE_SEARCH_DEPTH);
            } else {
                new_policy = pe.policy + 0.1f;
                if (!(score > alpha)) {
                    float clip = std::max(node.policy[0].policy * 0.98f, pe.policy);
                    new_policy = std::min(new_policy, clip);
                }
            }
            pe.policy = new_policy;
            if (new_depth > best_move_depth) best_move_depth = new_depth; // TODO: This line is suspicious

            // After finishing this child's re-searches, update global alpha
            if (score > alpha) alpha = score;
            if (score > bestScore) {
                bestScore = score;
                bestMove = pe.move;
            }
            if (score >= beta) {
                co_return SearchOutcome{bestScore, bestMove, false};
            }
        }

        co_return SearchOutcome{bestScore, bestMove, false};
    }

    cppcoro::task<RootResult> lks_root(LKSNode& root, float depth, float alpha, float beta) {
        auto out = co_await lks_search(root, depth, alpha, beta, NodeType::PV, 0);
        co_return RootResult{out.score, out.bestMove, out.aborted};
    }

private:
    chess::Board board_;
    std::atomic<bool> stop_requested_{false};
    engine_parallel::NNEvaluator evaluator_;
    cppcoro::static_thread_pool pool_;
    
    std::unique_ptr<LKSNode> root_;
    std::atomic<std::uint64_t> stat_gpu_evaluations_{0};
    std::atomic<std::uint64_t> stat_nodes_created_{0};
    std::atomic<int> stat_seldepth_{0};
    std::atomic<std::uint64_t> stat_tbhits_{0};
    std::atomic<std::uint64_t> stat_tthits_{0};
    std::atomic<std::int64_t> stat_search_start_ns_{0};

    // --- Helpers for UCI info output ---
    void print_info_line(float depth, float bestScore) {
        std::ostringstream oss;
        // depth with fractional display
        oss << "info depth " << std::fixed << std::setprecision(1) << depth;
        // seldepth
        oss << " seldepth " << getSelDepth();
        // multipv always 1
        oss << " multipv 1";
        // score in centipawns using expected reward -> cp mapping
        int score_cp = static_cast<int>(std::lround(std::tan(static_cast<double>(bestScore) * 1.563754) * 90.0));
        oss << " score cp " << score_cp;
        // nodes and nps
        std::uint64_t nodes = getNodesCreatedCount();
        oss << " nodes " << nodes;
        std::uint64_t ms = getElapsedTimeMs();
        std::uint64_t nps = (ms == 0) ? nodes : (nodes * 1000ULL) / ms;
        oss << " nps " << nps;
        // tbhits and time
        oss << " tbhits " << getTBHitsCount();
        oss << " time " << ms;
        // pv line
        std::string pv_line = build_pv_line();
        if (!pv_line.empty()) {
            oss << " pv " << pv_line;
        }
        std::cout << oss.str() << '\n';
    }

    std::string build_pv_line() const {
        if (!root_) return {};
        const LKSNode *node = root_.get();
        std::ostringstream pv;
        bool first = true;
        while (node && !node->policy.empty()) {
            const auto &pe = node->policy[0];
            if (pe.move == chess::Move::NO_MOVE) break;
            if (!first) pv << ' ';
            pv << chess::uci::moveToUci(pe.move);
            first = false;
            if (pe.child) node = pe.child.get(); else break;
        }
        return pv.str();
    }

    // Check node/evaluation limits for early termination
    bool node_limit_check(const Limits &limits) const {
        if (limits.nodes == 0ULL) return true; // no explicit node limit set
        const std::uint64_t evals = stat_gpu_evaluations_.load(std::memory_order_relaxed);
        // Continue while GPU evals remain under 95% of the node limit
        return evals * 100ULL < limits.nodes * 95ULL;
    }

    cppcoro::task<engine_parallel::EvalResult> evaluateAsync(const chess::Board &b) {
        auto tokens = engine_tokenizer::tokenizeBoard(b);
        engine_parallel::EvalAwaitable awaitable{&evaluator_, tokens};
        engine_parallel::EvalResult res = co_await awaitable;
        co_return res;
    }

    // Wrapper that ensures continuation resumes on our thread pool and tracks stats
    cppcoro::task<engine_parallel::EvalResult> evaluate_on_pool(const chess::Board &b) {
        // Count this evaluation request
        stat_gpu_evaluations_.fetch_add(1, std::memory_order_relaxed);
        // Await GPU evaluation (may resume on evaluator worker thread)
        engine_parallel::EvalResult res = co_await evaluateAsync(b);
        // Bounce back to the search thread pool to continue the coroutine on the desired executor
        co_await pool_.schedule();
        co_return res;
    }

    void sort_and_normalize(LKSNode &node) {
        std::sort(node.policy.begin(), node.policy.end(), [](const LKSPolicyEntry &a, const LKSPolicyEntry &b){ return a.policy > b.policy; });
        float sum = 0.0f;
        for (const auto &e : node.policy) sum += e.policy;
        if (sum > 0.0f) {
            for (auto &e : node.policy) e.policy = e.policy / sum;
        }
    }

    void backpropagate_policy_updates(LKSNode &parent, const LKSNode &child, const chess::Move &move) {
        // Find entry for move in parent
        for (auto &entry : parent.policy) {
            if (entry.move == move) {
                const float parent_to_node_policy = entry.policy;
                const float parent_Q_for_child = entry.Q;
                // Child value is from child's perspective; flip to parent's
                const float child_from_parent_perspective = -child.value;
                const float backup = (child_from_parent_perspective - parent_Q_for_child) / (parent.value + 1.01f);
                const float new_policy_prob = parent_to_node_policy * std::exp(backup);
                entry.policy = new_policy_prob;
                // TODO: sanity check this backprop
                return;
            }
        }
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
    cppcoro::task<LKSNode> create_node(const chess::Board &board) {
        // Track node creation
        stat_nodes_created_.fetch_add(1, std::memory_order_relaxed);
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
            co_return LKSNode(board, terminal_value, {}, 0.0f, true);
        }

        // Evaluate network fully (suspend until ready)
        engine_parallel::EvalResult eval = co_await evaluate_on_pool(board);
        if (eval.canceled || stop_requested_.load(std::memory_order_acquire)) {
            co_return LKSNode(board, 0.0f, {}, 0.0f, false);
        }

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
            entries.push_back(std::move(e));
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

        co_return LKSNode(board, value, std::move(entries), node_U, false);
    }
};

} // namespace engine




