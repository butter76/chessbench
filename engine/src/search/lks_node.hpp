#pragma once

#include "chess.hpp"

#include <vector>
#include <memory_resource>
#include <memory>

namespace engine {

struct LKSNode; // forward declaration for child pointer

// No-op deleter for arena-allocated nodes
struct NoopDeleter {
    void operator()(LKSNode*) const noexcept {}
};
using NodePtr = std::unique_ptr<LKSNode, NoopDeleter>;

// Represents one policy entry: (move, policy, U, Q)
struct LKSPolicyEntry {
    chess::Move move{chess::Move::NO_MOVE};
    float policy{0.0f};
    float U{0.0f};
    float Q{0.0f};
    NodePtr child; // optional expanded child node
};

// Light-weight search node for LKS
struct LKSNode {
    float value{0.0f};                    // scalar evaluation in [-1, 1]
    std::pmr::vector<LKSPolicyEntry> policy;   // ordered list of policy moves and heads
    float U{0.0f};                        // node-level uncertainty (e.g., stddev)
    std::pmr::vector<float> cdf;               // CDF over hl bins (suffix sums of softmaxed hl)
    bool terminal{false};                 // terminal position flag
    chess::Move bestMove{chess::Move::NO_MOVE}; // cached best move from search

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


