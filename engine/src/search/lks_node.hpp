#pragma once

#include "chess.hpp"

#include <vector>
#include <memory>

namespace engine {

struct LKSNode; // forward declaration for child pointer

// Represents one policy entry: (move, policy, U, Q)
struct LKSPolicyEntry {
    chess::Move move{chess::Move::NO_MOVE};
    float policy{0.0f};
    float U{0.0f};
    float Q{0.0f};
    std::unique_ptr<LKSNode> child; // optional expanded child node
};

// Light-weight search node for LKS
struct LKSNode {
    chess::Board board{};                 // position state
    float value{0.0f};                    // scalar evaluation in [-1, 1]
    std::vector<LKSPolicyEntry> policy;   // ordered list of policy moves and heads
    float U{0.0f};                        // node-level uncertainty (e.g., stddev)
    bool terminal{false};                 // terminal position flag
    chess::Move bestMove{chess::Move::NO_MOVE}; // cached best move from search

    LKSNode() = default;

    LKSNode(const chess::Board &b,
            float v,
            std::vector<LKSPolicyEntry> pol,
            float u,
            bool t)
        : board(b), value(v), policy(std::move(pol)), U(u), terminal(t) {}
};

} // namespace engine


