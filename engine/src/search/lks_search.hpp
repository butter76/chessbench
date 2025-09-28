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
#include <unordered_map>
#include <mutex>
#include <array>
#include <string>
#include <algorithm>
#include <tbb/concurrent_hash_map.h>
#include <memory_resource>
#include <xxhash.h>

// Syzygy helpers
#include "../syzygy_helpers.hpp"

#include <cppcoro/task.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/async_manual_reset_event.hpp>
#include <cppcoro/when_all_ready.hpp>

namespace engine {
constexpr float IT_DEPTH_STEP = 0.2f;
constexpr float RE_SEARCH_DEPTH = IT_DEPTH_STEP;
constexpr float IMPROVER_POLICY_INCREASE = RE_SEARCH_DEPTH / 2;

constexpr float NULL_EPS = 1e-6f;

// Using cppcoro::task for async operations

class LksSearch : public SearchAlgo {
public:
    explicit LksSearch(engine::Options &options, const engine::TimeHandler *time_handler)
        : SearchAlgo(options, time_handler), board_(), evaluator_(options) {
        evaluator_.start();
        ensure_pool_built();
    }

    ~LksSearch() override {
        evaluator_.stop_and_join();
    }

    void reset() override {
        board_ = chess::Board();
        root_key_ = std::nullopt;
        // Release per-search arena memory then rebuild maps with fresh allocator state
        search_arena_.release();
        node_map_ = NodeMap(NodeAlloc(&search_arena_));
        tt_generation_.fetch_add(1, std::memory_order_relaxed);
    }

    void makemove(const std::string &uci) override {
        const chess::Move move = chess::uci::uciToMove(board_, uci);
        if (move == chess::Move::NO_MOVE) {
            root_key_ = std::nullopt;
            tt_generation_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Always update the board state
        board_.makeMove(move);

        // Clear the TT if the board has repeated as we have 2-fold repetition on
        if (board_.isRepetition(1)) {
            tt_generation_.fetch_add(1, std::memory_order_relaxed);
        }

        // With key-based map, simply advance root key; map retains reused children implicitly
        root_key_ = make_key128(board_);
    }

    chess::Board &getBoard() override { return board_; }

    // Initialize engine explicitly (e.g., after isready)
    void initialize() {
        evaluator_.initialize_trt();
        ensure_pool_built();
        // Read configurable PV depth threshold for forcing all children expansion
        {
            int parsed = 0;
            try {
                const std::string opt = options_.get("forceallchildrenonpvdepth", "0");
                parsed = std::stoi(opt);
            } catch (...) {
                parsed = 0;
            }
            if (parsed < 0) parsed = 0;
            if (parsed > 16) parsed = 16;
            force_all_children_on_pv_depth_ = parsed;
        }
        // Initialize Syzygy tablebases once using configured path
        static std::once_flag tb_once_flag;
        std::call_once(tb_once_flag, [this]() {
            const std::string tb_path_opt = options_.get("syzygypath", "../syzygy_tables/3-4-5/");
            const char *tb_path_cstr = tb_path_opt.empty() ? nullptr : tb_path_opt.c_str();
            (void)tb_init(tb_path_cstr);
        });
    }

    void stop() override {
        stop_requested_.store(true, std::memory_order_release);
        evaluator_.cancelQueue();
    }

    std::string searchBestMove(const Limits &limits) override {
        ensure_pool_built();
        stop_requested_.store(false, std::memory_order_release);
        // Reset per-search statistics
        stat_gpu_evaluations_.store(0, std::memory_order_relaxed);
        stat_nodes_created_.store(0, std::memory_order_relaxed);
        stat_tbhits_.store(0, std::memory_order_relaxed);
        stat_tthits_.store(0, std::memory_order_relaxed);
        stat_seldepth_.store(0, std::memory_order_relaxed);
        stat_parent_nodes_.store(0, std::memory_order_relaxed);
        // Mark search start time
        {
            using namespace std::chrono;
            const std::int64_t now_ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
            stat_search_start_ns_.store(now_ns, std::memory_order_relaxed);
        }

        // Syzygy TB early exit for 3-5 men
        if (auto tb_move = engine::syzygy::probe_best_move(board_)) {
            stat_tbhits_.fetch_add(1, std::memory_order_relaxed);
            return chess::uci::moveToUci(*tb_move);
        }
        
        // Establish time budgets (only if time controls are provided)
        engine::TimeHandler::TimeBudget budget{};
        const bool has_time_limits = (limits.movetime_ms > 0ULL) || (limits.wtime_ms > 0ULL) || (limits.btime_ms > 0ULL);
        if (time_handler_ != nullptr && has_time_limits && !limits.infinite) {
            budget = time_handler_->selectTimeBudget(limits, board_.sideToMove());
        }
        using clock = std::chrono::steady_clock;
        const clock::time_point start_tp = clock::now();
        const clock::time_point soft_deadline = (budget.soft_ms > 0ULL) ? (start_tp + std::chrono::milliseconds(budget.soft_ms)) : clock::time_point::max();
        const clock::time_point hard_deadline = (budget.hard_ms > 0ULL) ? (start_tp + std::chrono::milliseconds(budget.hard_ms)) : clock::time_point::max();
        // Watchdog to enforce hard deadline by triggering stop()
        std::jthread watchdog([this, hard_deadline](std::stop_token st){
            if (hard_deadline == std::chrono::steady_clock::time_point::max()) return;
            for (;;) {
                if (st.stop_requested()) break;
                if (std::chrono::steady_clock::now() >= hard_deadline) {
                    this->stop();
                    std::cout << "info string hard deadline reached\n" << std::flush;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        
        float maxDepth = (limits.depth > 0) ? static_cast<float>(limits.depth) : 35.0f;
        float currentDepth = std::min(2.0f, maxDepth);

        chess::Movelist rootMoves;
        chess::movegen::legalmoves(rootMoves, board_);
        if (rootMoves.empty()) return "0000";

        float bestScore = -std::numeric_limits<float>::infinity();
        if (!root_key_.has_value()) root_key_ = make_key128(board_);
        {
            auto rootMaybe = cppcoro::sync_wait(create_node(board_));
            if (!rootMaybe) {
                return chess::uci::moveToUci(rootMoves[0]);
            }
            // ensure root stored in map (create_node does) and get a local copy
            // local copy used for this iteration
            // Note: lks_root will persist updates back to map
        }
        
        while (currentDepth <= maxDepth + 1e-2f) {
            if (clock::now() >= hard_deadline) { stop(); break; }
            if (stop_requested_.load(std::memory_order_acquire)) break;
            if (!node_limit_check(limits)) break;
            // Load a fresh working copy of the root node from the map
            auto rootCopyOpt = cppcoro::sync_wait(load_or_create_node_copy(board_));
            if (!rootCopyOpt) break;
            LKSNode rootCopy = std::move(*rootCopyOpt);
            auto [score, move, aborted] = cppcoro::sync_wait([&]() -> cppcoro::task<RootResult> {
                co_await pool_->schedule();
                co_return co_await lks_root(rootCopy, board_, currentDepth, -1.0f, 1.0f);
            }());
            if (aborted) break;
            bestScore = score;
            // Emit UCI info line for this iteration
            print_info_line(currentDepth, bestScore);
            currentDepth += IT_DEPTH_STEP;
            // Respect soft deadline: finish current iteration then exit
            if (clock::now() >= soft_deadline) break;
        }

        // Return best move from the latest root record in the map
        {
            auto rootRec = try_load_node(make_key128(board_));
            if (rootRec && rootRec->bestMove != chess::Move::NO_MOVE) {
                return chess::uci::moveToUci(rootRec->bestMove);
            }
            // Fallback: pick top policy move if available
            if (rootRec && !rootRec->policy.empty()) return chess::uci::moveToUci(rootRec->policy[0].move);
            return "0000";
        }
    }

    // Statistics API (thread-safe)
    std::uint64_t getGpuEvaluationsCount() const {
        return stat_gpu_evaluations_.load(std::memory_order_relaxed);
    }

    std::uint64_t getNodesCreatedCount() const {
        return stat_nodes_created_.load(std::memory_order_relaxed);
    }

    std::uint64_t getParentNodesCount() const {
        return stat_parent_nodes_.load(std::memory_order_relaxed);
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

    struct Phase2ChildResult {
        bool aborted;
        bool cutoff;
        float alpha_out;
        float score;
        chess::Move move;
        bool is_improver;
    };

    cppcoro::task<SearchOutcome> lks_search(LKSNode& node, const chess::Board &board, float depth, float alpha, float beta, NodeType node_type, int rec_depth = 0, int pv_depth = 0, bool root = false) {
        if (stop_requested_.load(std::memory_order_acquire)) co_return SearchOutcome{0.0f, chess::Move::NO_MOVE, true};

        if (node.terminal || node.policy.empty()) {
            co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
        }

        // During search, use 2-fold repetition, but don't allow it for the root node
        if (board.isRepetition(1) && !root) {
            co_return SearchOutcome{0.0f, chess::Move::NO_MOVE, false};
        }

        if (rec_depth > 100) {
            co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
        }

        const float alpha0 = alpha;
        const float beta0 = beta;
        const int alpha_bin0 = alpha_to_bin(alpha0);

        // Transposition Table probe (after recursion depth guard)
        if (auto tt_score = query_tt(node, alpha0, beta0, depth, alpha_bin0)) {
            co_return SearchOutcome{*tt_score, chess::Move::NO_MOVE, false};
        }

        if (depth <= -2.0f) {
            // Avoid infinite loops with this
            co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
        }

        // Ensure policy is sorted and normalized before expansion
        sort_and_normalize(node);

        float node_depth_reduction = -2.0f * std::log(node.U + 1e-6f);

        if (node_type != NodeType::PV) {
            auto bin = alpha_bin0;
            float prob_greater_than_alpha = std::max(1e-6f, node.cdf[bin]);
            if (prob_greater_than_alpha < 0.1f && node.value < -1.0f + bin * 2.0f / 81.0f) {
                // This position is a moonshot, so alt formula
                float var_above_alpha = 1e-5f;
                float last_bin_cdf = 0.0f;
                for (int i = 80; i >= bin; --i) {
                    float pdf_i = node.cdf[i] - last_bin_cdf;
                    last_bin_cdf = node.cdf[i];
                    float bin_center = (i * 2.0f + 1.0f) / 81.0f - 1.0f;
                    var_above_alpha += pdf_i * (bin_center - node.value) * (bin_center - node.value);
                }
                node_depth_reduction = std::max(node_depth_reduction, -std::log(var_above_alpha * 2));
            }
        }

        auto child_exists = [&](const chess::Move &mv) -> bool {
            chess::Board cb = board;
            cb.makeMove(mv);
            NodeMap::const_accessor acc;
            return node_map_.find(acc, make_key128(cb));
        };
        auto is_leaf_node = [&](const LKSNode &n) {
            for (const auto &pe : n.policy) if (child_exists(pe.move)) return false;
            return true;
        };

        float weight_divisor = 1.0f;
        int unexpanded_count = 0;
        float total_weight_scan = 0.0f;
        for (std::size_t i = 0; i < node.policy.size(); ++i) {
            const auto &pe = node.policy[i];
            if (!child_exists(pe.move)) {
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
                // update_tt(board, alpha0, beta0, depth, node.value);
                co_return SearchOutcome{node.value, chess::Move::NO_MOVE, false};
            }

            // First expansion bookkeeping: update maximum selective depth
            stat_parent_nodes_.fetch_add(1, std::memory_order_relaxed);
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
            float new_depth = depth + std::log(move_weight + 1e-6f) - std::log(weight_divisor + 1e-6f);
            if (i == 0) best_move_depth = new_depth;

            new_depths[i] = new_depth;

            bool should_filter = false;
            if (!child_exists(pe.move)) {
                const float local_reduction = -2.0f * std::log(pe.U + 1e-6f);
                if (new_depth <= local_reduction) {
                    if ((total_weight > 0.80f && i >= 2) || (total_weight > 0.95f && i >= 1)) {
                        should_filter = !(pv_depth < force_all_children_on_pv_depth_) && !root;
                    }
                }
            }
            if (!should_filter) filtered_indices.push_back(i);
            total_weight += move_weight;
        }

        // Phase 2: for each filtered child, ensure child exists, run initial search
        // Jamboree-style: i==0 sequential, others in parallel without beta cutoffs
        std::vector<std::size_t> improver_indices;
        improver_indices.reserve(filtered_indices.size());

        if (!filtered_indices.empty()) {
            std::size_t i0 = filtered_indices[0];
            float depth0 = new_depths[i0];
            auto r0 = co_await process_phase2_child(node, board, i0, depth0, alpha, beta, node_type, rec_depth, pv_depth);
            if (r0.aborted) co_return SearchOutcome{0.0f, bestMove, true};
            alpha = r0.alpha_out;
            bestScore = r0.score;
            bestMove = r0.move;
            if (r0.cutoff) {
                update_tt(node, alpha0, beta0, depth, bestScore, alpha_bin0);
                node.bestMove = bestMove;
                persist_node_copy(board, node, depth);
                co_return SearchOutcome{bestScore, bestMove, false};
            }

            // Launch parallel searches for remaining children as coroutine tasks
            std::vector<std::size_t> non_first_indices;
            non_first_indices.reserve(filtered_indices.size() > 0 ? filtered_indices.size() - 1 : 0);
            std::vector<cppcoro::task<Phase2ChildResult>> tasks;
            tasks.reserve(filtered_indices.size() > 0 ? filtered_indices.size() - 1 : 0);

            const float alpha_after_first = alpha;
            for (std::size_t idx_pos = 1; idx_pos < filtered_indices.size(); ++idx_pos) {
                std::size_t i = filtered_indices[idx_pos];
                float nd = new_depths[i];
                non_first_indices.push_back(i);
                tasks.push_back(process_phase2_child(node, board, i, nd, alpha_after_first, beta, node_type, rec_depth, pv_depth));
            }

            // Await completion of all non-first children without blocking pool threads
            if (!tasks.empty()) {
                auto ready = co_await cppcoro::when_all_ready(std::move(tasks));
                for (std::size_t t = 0; t < ready.size(); ++t) {
                    Phase2ChildResult r = std::move(ready[t]).result();
                    if (r.aborted) co_return SearchOutcome{0.0f, bestMove, true};
                    if (r.is_improver) {
                        improver_indices.push_back(non_first_indices[t]);
                    }
                }
            }
        }

        // Reorder improvers: prioritize current bestMove if present and multiple improvers exist
        if (improver_indices.size() > 1) {
            chess::Move currentBest = node.bestMove;
            if (currentBest != chess::Move::NO_MOVE) {
                auto it = std::find_if(improver_indices.begin(), improver_indices.end(), [&](std::size_t idx) {
                    return node.policy[idx].move == currentBest;
                });
                if (it != improver_indices.end() && it != improver_indices.begin()) {
                    std::size_t idxVal = *it;
                    improver_indices.erase(it);
                    improver_indices.insert(improver_indices.begin(), idxVal);
                }
            }
        }

        // Phase 3: re-search improvers (non-first moves)
        bool improver = true;
        for (std::size_t k = 0; k < improver_indices.size(); ++k) {
            std::size_t i = improver_indices[k];
            auto &pe = node.policy[i];
            float new_depth = new_depths[i];
            float score = std::numeric_limits<float>::infinity();
            int re_search_count = 1;
            new_depth += RE_SEARCH_DEPTH;

            if (root) {
                std::ostringstream oss;
                oss << "info string re-searching improver " << chess::uci::moveToUci(pe.move) << " at nodes " << getGpuEvaluationsCount();
                std::cout << oss.str() << '\n' << std::flush;
            }

            // Continue incremental null-window searches until reaching best_move_depth
            while (score > alpha && new_depth < best_move_depth) {
                // Increment depth

                NodeType next_type = (node_type == NodeType::CUT) ? NodeType::ALL : NodeType::CUT;
                chess::Board child_board = board;
                child_board.makeMove(pe.move);
                int child_pv_depth = pv_depth + ((next_type != NodeType::PV) ? 1 : 0);
                LKSNode child_node = try_load_node(make_key128(child_board)).value();
                auto child_out = co_await lks_search(child_node, child_board, new_depth, -alpha - NULL_EPS, -alpha, next_type, rec_depth + 1, child_pv_depth);
                if (child_out.aborted) co_return SearchOutcome{0.0f, bestMove, true};
                score = -child_out.score;
                new_depth += RE_SEARCH_DEPTH;
                re_search_count += 1;

                if (score > alpha) {
                    improver = true;
                }
            }

            // If still improving alpha, do full-window re-search
            if (score > alpha) {
                NodeType next_type = (node_type == NodeType::CUT) ? NodeType::ALL : NodeType::CUT;
                NodeType fw_type = (node_type == NodeType::PV) ? NodeType::PV : next_type;
                int fw_pv_depth = pv_depth + ((fw_type != NodeType::PV) ? 1 : 0);
                chess::Board child_board = board;
                child_board.makeMove(pe.move);
                LKSNode child_node = try_load_node(make_key128(child_board)).value();
                auto child_out = co_await lks_search(child_node, child_board, new_depth, -beta, -alpha, fw_type, rec_depth + 1, fw_pv_depth);
                if (child_out.aborted) co_return SearchOutcome{0.0f, bestMove, true};
                score = -child_out.score;
            }

            if (score > alpha) {
                improver = true;
            }

            // Update policy
            if (improver) {
                if (root) {
                    std::ostringstream oss;
                    oss << "info string upweighting improver " << chess::uci::moveToUci(pe.move);
                    std::cout << oss.str() << '\n' << std::flush;
                }
                float new_policy = pe.policy;
                if (node_type == NodeType::CUT && score > alpha) {
                    new_policy = pe.policy * std::exp(re_search_count * RE_SEARCH_DEPTH);
                } else {
                    new_policy = pe.policy + IMPROVER_POLICY_INCREASE;
                    if (!(score > alpha)) {
                        float clip = std::max(node.policy[0].policy * 0.98f, pe.policy);
                        new_policy = std::min(new_policy, clip);
                    }
                }
                if (score > alpha) {
                    if (root) {
                        std::ostringstream oss;
                        oss << "info string successfully re-searched improver " << chess::uci::moveToUci(pe.move) << " at nodes " << getGpuEvaluationsCount();
                        std::cout << oss.str() << '\n' << std::flush;
                    }
                }
                pe.policy = new_policy;
            }

            // After finishing this child's re-searches, update global alpha
            if (score > alpha) alpha = score;
            if (score > bestScore) {
                bestScore = score;
                bestMove = pe.move;
                node.bestMove = bestMove; // update bestMove immediately in case of an abort
            }
            if (score >= beta) {
                update_tt(node, alpha0, beta0, depth, bestScore, alpha_bin0);
                node.bestMove = bestMove;
                persist_node_copy(board, node, depth);
                co_return SearchOutcome{bestScore, bestMove, false};
            }
            improver = false;
        }

        update_tt(node, alpha0, beta0, depth, bestScore, alpha_bin0);
        node.bestMove = bestMove;
        persist_node_copy(board, node, depth);
        co_return SearchOutcome{bestScore, bestMove, false};
    }

    cppcoro::task<RootResult> lks_root(LKSNode& root, const chess::Board &board, float depth, float alpha, float beta) {
        auto out = co_await lks_search(root, board, depth, alpha, beta, NodeType::PV, 0, 0, true);
        co_return RootResult{out.score, out.bestMove, out.aborted};
    }

private:
    // Resource must be constructed before (and destroyed after) PMR containers using it
    mutable std::pmr::synchronized_pool_resource search_arena_{};
    // --- Unified node storage in TBB map using LKSNode directly ---

    struct Key128 { std::uint64_t hi; std::uint64_t lo; };
    struct Key128HashCompare {
        std::size_t hash(const Key128 &k) const noexcept {
            std::uint64_t x = k.hi ^ (k.lo + 0x9e3779b97f4a7c15ULL + (k.hi << 6) + (k.hi >> 2));
            return static_cast<std::size_t>(x);
        }
        bool equal(const Key128 &a, const Key128 &b) const noexcept {
            return a.hi == b.hi && a.lo == b.lo;
        }
    };

    using NodeAlloc = std::pmr::polymorphic_allocator<std::pair<const Key128, LKSNode>>;
    using NodeMap = tbb::concurrent_hash_map<Key128, LKSNode, Key128HashCompare, NodeAlloc>;
    mutable NodeMap node_map_{NodeAlloc(&search_arena_)};

    struct CompactState {
        std::uint64_t bb[12];
        std::uint64_t castling_ep_stm; // pack: castling(0..15), ep_file(0..7 or 0xFF), stm(0/1)
    };

    static inline CompactState build_compact_state(const chess::Board &b) {
        CompactState s{};
        int idx = 0;
        for (int color = 0; color < 2; ++color) {
            auto c = color == 0 ? chess::Color::WHITE : chess::Color::BLACK;
            s.bb[idx++] = b.pieces(chess::PieceType::PAWN, c).getBits();
            s.bb[idx++] = b.pieces(chess::PieceType::KNIGHT, c).getBits();
            s.bb[idx++] = b.pieces(chess::PieceType::BISHOP, c).getBits();
            s.bb[idx++] = b.pieces(chess::PieceType::ROOK, c).getBits();
            s.bb[idx++] = b.pieces(chess::PieceType::QUEEN, c).getBits();
            s.bb[idx++] = b.pieces(chess::PieceType::KING, c).getBits();
        }
        std::uint64_t castling = static_cast<std::uint64_t>(b.castlingRights().hashIndex() & 0xF);
        std::uint64_t ep_file = 0xFFu;
        auto ep = b.enpassantSq();
        if (ep != chess::Square::NO_SQ) ep_file = static_cast<std::uint64_t>(ep.file());
        std::uint64_t stm = (b.sideToMove() == chess::Color::WHITE) ? 0ULL : 1ULL;
        s.castling_ep_stm = (castling & 0xF) | ((ep_file & 0xFF) << 8) | ((stm & 0x1) << 16);
        return s;
    }

    static inline Key128 make_key128(const chess::Board &board) {
        std::uint64_t lo = static_cast<std::uint64_t>(board.hash());
        CompactState st = build_compact_state(board);
        std::uint64_t hi = XXH3_64bits(&st, sizeof(st));
        return Key128{hi, lo};
    }

    std::optional<LKSNode> try_load_node(const Key128 &key) const {
        NodeMap::const_accessor acc;
        if (!node_map_.find(acc, key)) return std::nullopt;
        // Return a copy so callers can mutate safely
        return acc->second;
    }

    cppcoro::task<std::optional<LKSNode>> load_or_create_node_copy(const chess::Board &board) {
        const Key128 key = make_key128(board);
        if (auto n = try_load_node(key)) co_return n;
        auto created = co_await create_node(board);
        co_return created; // may be std::nullopt if evaluation was canceled
    }

    LKSNode build_from_eval_and_insert(const chess::Board &board,
                                       const engine_parallel::EvalResult &eval) {
        // Convert scalar value to [-1, 1]
        float value = 2.0f * eval.value - 1.0f;

        // Prepare policy over legal moves using logits from eval.policy (hardest_policy)
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        const bool flip = (board.sideToMove() == chess::Color::BLACK);

        struct Gathered { chess::Move mv; float logit; float U; float Q; int idx; int s1; int s2; };
        std::pmr::vector<Gathered> gathered{std::pmr::polymorphic_allocator<Gathered>(&search_arena_)};
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
        std::pmr::vector<LKSPolicyEntry> entries{std::pmr::polymorphic_allocator<LKSPolicyEntry>(&search_arena_)};
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
            entries.push_back(std::move(e));
        }
        std::sort(entries.begin(), entries.end(), [](const LKSPolicyEntry &a, const LKSPolicyEntry &b){ return a.policy > b.policy; });

        // Compute node-level U from hl logits as wdl_variance and build CDF over bins
        float node_U = 0.0f;
        std::pmr::vector<float> cdf{std::pmr::polymorphic_allocator<float>(&search_arena_)};
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

            // Build suffix-sum CDF: [p1+...+pn, p2+...+pn, ..., pn]
            cdf.resize(static_cast<std::size_t>(bins));
            double running = 0.0;
            for (int i = bins - 1; i >= 0; --i) {
                running += probs[static_cast<std::size_t>(i)];
                cdf[static_cast<std::size_t>(i)] = static_cast<float>(running);
            }

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

        // Insert into unified node map
        {
            NodeMap::accessor acc;
            const Key128 key = make_key128(board);
            node_map_.insert(acc, key);
            acc->second = LKSNode{value, std::move(entries), node_U, std::move(cdf), false};
        }
        // Return a copy of the stored node
        auto loaded = try_load_node(make_key128(board));
        return loaded.value();
    }

    void persist_node_copy(const chess::Board &board, const LKSNode &node, float at_depth) {
        NodeMap::accessor acc;
        const Key128 key = make_key128(board);
        node_map_.insert(acc, key);
        // Only overwrite if this write is at a deeper search depth than stored
        if (at_depth > acc->second.depth_record) {
            LKSNode updated = node;
            updated.depth_record = at_depth;
            acc->second = std::move(updated);
        }
    }

    // --- Transposition Table (embedded in node) ---
public:
    // Query the node-embedded TT for this node. Returns a score if usable.
    std::optional<float> query_tt(const LKSNode &n, float alpha, float beta, float depth, int alpha_bin) {
        const std::uint64_t cur_gen = tt_generation_.load(std::memory_order_relaxed);
        // Prefer exact value if at sufficient depth
        if (n.tt_exact.has && n.tt_exact.gen == cur_gen && n.tt_exact.depth >= depth && alpha_bin >= n.tt_exact.min_bin) return n.tt_exact.score;
        // Lower bound can trigger beta cutoff
        if (n.tt_lower.has && n.tt_lower.gen == cur_gen && n.tt_lower.depth >= depth && alpha_bin >= n.tt_lower.min_bin && n.tt_lower.score >= beta) return n.tt_lower.score;
        // Upper bound can trigger alpha cutoff
        if (n.tt_upper.has && n.tt_upper.gen == cur_gen && n.tt_upper.depth >= depth && alpha_bin >= n.tt_upper.min_bin && n.tt_upper.score <= alpha) return n.tt_upper.score;
        return std::nullopt;
    }

    // Update the node-embedded TT entry for this node with a result at the given window and depth.
    void update_tt(LKSNode &n, float alpha, float beta, float depth, float score, int alpha_bin) {
        const std::uint64_t cur_gen = tt_generation_.load(std::memory_order_relaxed);
        if (score <= alpha) {
            if (!n.tt_upper.has || n.tt_upper.depth <= depth) {
                n.tt_upper.has = true;
                n.tt_upper.score = score;
                n.tt_upper.depth = depth;
                n.tt_upper.min_bin = alpha_bin;
                n.tt_upper.gen = cur_gen;
            }
        } else if (score >= beta) {
            if (!n.tt_lower.has || n.tt_lower.depth <= depth) {
                n.tt_lower.has = true;
                n.tt_lower.score = score;
                n.tt_lower.depth = depth;
                n.tt_lower.min_bin = alpha_bin;
                n.tt_lower.gen = cur_gen;
            }
        } else {
            if (!n.tt_exact.has || n.tt_exact.depth <= depth) {
                n.tt_exact.has = true;
                n.tt_exact.score = score;
                n.tt_exact.depth = depth;
                n.tt_exact.min_bin = alpha_bin;
                n.tt_exact.gen = cur_gen;
            }
        }
    }

private:
    // Map alpha in [-1,1] to integer bin [0,80]
    static inline int alpha_to_bin(float alpha) {
        int bin = static_cast<int>(std::floor(81.0f * (alpha + 1.0f) / 2.0f));
        if (bin < 0) bin = 0;
        if (bin > 80) bin = 80;
        return bin;
    }
    cppcoro::task<Phase2ChildResult> process_phase2_child(
        LKSNode &node,
        const chess::Board &board,
        std::size_t i,
        float new_depth,
        float alpha,
        float beta,
        NodeType node_type,
        int rec_depth,
        int pv_depth
    ) {
        auto &pe = node.policy[i];

        // Ensure child exists in map and backpropagate policy updates
        chess::Board child_board = board;
        child_board.makeMove(pe.move);
        // Load fresh copy for recursive search
        auto child_opt = co_await load_or_create_node_copy(child_board);
        if (!child_opt) {
            co_return Phase2ChildResult{true, false, alpha, 0.0f, chess::Move::NO_MOVE, false};
        }
        LKSNode child_node = std::move(*child_opt);
        // backpropagate_policy_updates(node, child_node, pe);

        float search_alpha = (i == 0) ? -beta : -alpha - NULL_EPS;
        float search_beta = -alpha;
        NodeType next_type;
        if (i == 0 && node_type == NodeType::PV) next_type = NodeType::PV;
        else if (node_type == NodeType::CUT) next_type = NodeType::ALL;
        else next_type = NodeType::CUT;

        // child_board already set
        int child_pv_depth = pv_depth + ((next_type != NodeType::PV) ? 1 : 0);

        auto child_out = co_await lks_search(child_node, child_board, new_depth, search_alpha, search_beta, next_type, rec_depth + 1, child_pv_depth);
        if (child_out.aborted) {
            co_return Phase2ChildResult{true, false, alpha, 0.0f, chess::Move::NO_MOVE, false};
        }
        float score = -child_out.score;

        if (i == 0) {
            float alpha_out = alpha;
            if (score > alpha_out) alpha_out = score;
            bool cutoff = score >= beta;
            co_return Phase2ChildResult{false, cutoff, alpha_out, score, pe.move, false};
        } else {
            bool is_improver = score > alpha;
            co_return Phase2ChildResult{false, false, alpha, score, pe.move, is_improver};
        }
    }

    chess::Board board_;
    std::atomic<bool> stop_requested_{false};
    engine_parallel::NNEvaluator evaluator_;
    std::unique_ptr<cppcoro::static_thread_pool> pool_;
    std::size_t pool_threads_{8};
    int force_all_children_on_pv_depth_{0};

    std::size_t desired_thread_count_from_options() const {
        const std::string val = options_.get("threads", "");
        unsigned int hc = std::thread::hardware_concurrency();
        if (hc == 0u) hc = 8u;
        unsigned int t = hc;
        if (!val.empty()) {
            try {
                unsigned long parsed = std::stoul(val);
                if (parsed > 0ul) t = static_cast<unsigned int>(parsed);
            } catch (...) {
                // ignore parse errors, keep default
            }
        }
        if (t < 1u) t = 1u;
        if (t > 512u) t = 512u;
        return static_cast<std::size_t>(t);
    }

    void ensure_pool_built() {
        const std::size_t desired = desired_thread_count_from_options();
        if (!pool_ || pool_threads_ != desired) {
            pool_threads_ = desired;
            pool_ = std::make_unique<cppcoro::static_thread_pool>(desired);
        }
    }
    
    std::optional<Key128> root_key_;
    std::atomic<std::uint64_t> stat_gpu_evaluations_{0};
    std::atomic<std::uint64_t> stat_nodes_created_{0};
    std::atomic<std::uint64_t> stat_parent_nodes_{0};
    std::atomic<int> stat_seldepth_{0};
    std::atomic<std::uint64_t> stat_tbhits_{0};
    std::atomic<std::uint64_t> stat_tthits_{0};
    std::atomic<std::int64_t> stat_search_start_ns_{0};
    std::atomic<std::uint64_t> tt_generation_{1};

    // --- Helpers for UCI info output ---
    void print_info_line(float depth, float bestScore) {
        std::ostringstream oss;
        {
            const std::string showfrac = options_.get("fractionaldepth", "");
            if (!showfrac.empty() && showfrac != "0") {
                // depth with fractional display
                oss << "info depth " << std::fixed << std::setprecision(1) << depth;
            } else {
                float classic_depth = std::lround((depth + 0.01f) / IT_DEPTH_STEP);
                oss << "info depth " << classic_depth;
            }
        }
        // seldepth
        oss << " seldepth " << getSelDepth();
        // multipv always 1
        oss << " multipv 1";
        // score in centipawns using expected reward -> cp mapping
        int score_cp = static_cast<int>(std::lround(std::tan(static_cast<double>(bestScore) * 1.563754) * 90.0));
        oss << " score cp " << score_cp;
        // nodes and nps
        std::uint64_t nodes = getGpuEvaluationsCount();
        oss << " nodes " << nodes;
        std::uint64_t ms = getElapsedTimeMs();
        std::uint64_t nps = (ms == 0) ? nodes : (nodes * 1000ULL) / ms;
        oss << " nps " << nps;
        // tbhits and time
        oss << " tbhits " << getTBHitsCount();
        oss << " time " << ms;
        // optional branching factor (gpu evaluations to parent nodes)
        {
            const std::string showbf = options_.get("showbf", "");
            if (!showbf.empty() && showbf != "0") {
                const std::uint64_t evals = getGpuEvaluationsCount();
                const std::uint64_t parents = getParentNodesCount();
                double bf = (parents == 0ULL) ? 0.0 : static_cast<double>(evals) / static_cast<double>(parents);
                oss << " bf " << std::fixed << std::setprecision(2) << bf;
            }
        }
        // pv line
        std::string pv_line = build_pv_line(board_);
        if (!pv_line.empty()) {
            oss << " pv " << pv_line;
        }
        std::cout << oss.str() << '\n';
    }

    std::string build_pv_line(const chess::Board &start_board) const {
        std::ostringstream pv;
        bool first = true;
        chess::Board b = start_board;
        for (int depth = 0; depth < 128; ++depth) {
            auto rec = try_load_node(make_key128(b));
            if (!rec) break;
            if (rec->bestMove == chess::Move::NO_MOVE) break;
            if (!first) pv << ' ';
            pv << chess::uci::moveToUci(rec->bestMove);
            first = false;
            b.makeMove(rec->bestMove);
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

    // Wrapper that ensures continuation resumes on our thread pool and tracks stats
    cppcoro::task<engine_parallel::EvalResult> evaluate_on_pool(const chess::Board &b) {
        // Count this evaluation request
        stat_gpu_evaluations_.fetch_add(1, std::memory_order_relaxed);
        // Await GPU evaluation via callback + event
        auto tokens = engine_tokenizer::tokenizeBoard(b);
        struct Shared { cppcoro::async_manual_reset_event done; engine_parallel::EvalResult res; };
        auto shared = std::shared_ptr<Shared>(new Shared());
        evaluator_.enqueue(tokens, [shared](engine_parallel::EvalResult r){ shared->res = std::move(r); shared->done.set(); });
        co_await shared->done;
        // Bounce back to the search thread pool to continue the coroutine on the desired executor
        co_await pool_->schedule();
        co_return std::move(shared->res);
    }

    void sort_and_normalize(LKSNode &node) {
        std::sort(node.policy.begin(), node.policy.end(), [](const LKSPolicyEntry &a, const LKSPolicyEntry &b){ return a.policy > b.policy; });
        float sum = 0.0f;
        for (const auto &e : node.policy) sum += e.policy;
        if (sum > 0.0f) {
            for (auto &e : node.policy) e.policy = e.policy / sum;
        }
    }

    void backpropagate_policy_updates(LKSNode &parent, const LKSNode &child, LKSPolicyEntry &entry) {
        const float parent_to_node_policy = entry.policy;
        const float parent_Q_for_child = entry.Q;
        const float child_from_parent_perspective = -child.value;
        float backup = (child_from_parent_perspective - parent_Q_for_child) / (parent.value + 1.01f);
        backup = std::clamp(backup, -2.0f, 4.0f);
        const float new_policy_prob = parent_to_node_policy * std::exp(backup);
        entry.policy = new_policy_prob;
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
    cppcoro::task<std::optional<LKSNode>> create_node(const chess::Board &board) {
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
            std::pmr::vector<LKSPolicyEntry> empty_pol{std::pmr::polymorphic_allocator<LKSPolicyEntry>(&search_arena_)};
            // Store terminal node in map
            const Key128 key = make_key128(board);
            {
                NodeMap::accessor acc;
                node_map_.insert(acc, key);
                std::pmr::vector<LKSPolicyEntry> empty_entries{std::pmr::polymorphic_allocator<LKSPolicyEntry>(&search_arena_)};
                acc->second = LKSNode{terminal_value, std::move(empty_entries), 0.0f, std::pmr::vector<float>{std::pmr::polymorphic_allocator<float>(&search_arena_)}, true};
            }
            co_return std::optional<LKSNode>(std::in_place, terminal_value, std::move(empty_pol), 0.0f, true);
        }

        // Syzygy: Use TB WDL to set terminal value (cursed/blessed treated as draw)
        {
            if (auto wdl_v = engine::syzygy::probe_wdl_value(board)) {
                std::pmr::vector<LKSPolicyEntry> empty_pol{std::pmr::polymorphic_allocator<LKSPolicyEntry>(&search_arena_)};
                const Key128 key = make_key128(board);
                {
                    NodeMap::accessor acc;
                    node_map_.insert(acc, key);
                    std::pmr::vector<LKSPolicyEntry> empty_entries{std::pmr::polymorphic_allocator<LKSPolicyEntry>(&search_arena_)};
                    acc->second = LKSNode{*wdl_v, std::move(empty_entries), 0.0f, std::pmr::vector<float>{std::pmr::polymorphic_allocator<float>(&search_arena_)}, true};
                }
                co_return std::optional<LKSNode>(std::in_place, *wdl_v, std::move(empty_pol), 0.0f, true);
            }
        }

        // Node map lookup by 128-bit key
        const Key128 key = make_key128(board);
        if (auto loaded = try_load_node(key)) {
            co_return std::optional<LKSNode>(*loaded);
        }

        // Evaluate network fully (suspend until ready)
        engine_parallel::EvalResult eval = co_await evaluate_on_pool(board);
        if (eval.canceled || stop_requested_.load(std::memory_order_acquire)) {
            co_return std::nullopt;
        }

        // Build node data from eval and store in cache
        auto built = build_from_eval_and_insert(board, eval);
        co_return std::optional<LKSNode>(std::in_place, built.value, std::move(built.policy), built.U, std::move(built.cdf), built.terminal);
    }
};

} // namespace engine




