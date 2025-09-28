#pragma once

#include "chess.hpp"

#include <vector>
#include <memory_resource>
#include <cstdint>
#include <limits>

namespace engine {

struct LKSNode; // forward declaration

// Lightweight per-node TT bound record
struct TTBoundRec {
    bool has{false};
    float score{0.0f};
    float depth{0.0f};
    int min_bin{0}; // minimum alpha-bin (0..80) required to use this record
    std::uint64_t gen{0}; // generation to validate freshness
};

// Represents one policy entry: (move, policy, U, Q)
struct LKSPolicyEntry {
    chess::Move move{chess::Move::NO_MOVE};
    float policy{0.0f};
    float U{0.0f};
    float Q{0.0f};
};

// Light-weight search node for LKS
struct LKSNode {
    float value{0.0f};                    // scalar evaluation in [-1, 1]
    std::pmr::vector<LKSPolicyEntry> policy;   // ordered list of policy moves and heads
    float U{0.0f};                        // node-level uncertainty (e.g., stddev)
    std::pmr::vector<float> cdf;               // CDF over hl bins (suffix sums of softmaxed hl)
    bool terminal{false};                 // terminal position flag
    chess::Move bestMove{chess::Move::NO_MOVE}; // cached best move from search
    // Embedded TT info for this node
    TTBoundRec tt_exact{};
    TTBoundRec tt_lower{};
    TTBoundRec tt_upper{};
    // Depth at which this node snapshot was last persisted
    float depth_record{ -std::numeric_limits<float>::infinity() };

    LKSNode() = default;

    LKSNode(float v,
            std::pmr::vector<LKSPolicyEntry> pol,
            float u,
            bool t)
        : value(v), policy(std::move(pol)), U(u), terminal(t) {}

    LKSNode(float v,
            std::pmr::vector<LKSPolicyEntry> pol,
            float u,
            std::pmr::vector<float> c,
            bool t)
        : value(v), policy(std::move(pol)), U(u), cdf(std::move(c)), terminal(t) {}
};

} // namespace engine


