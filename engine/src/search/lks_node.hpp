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
// Compact representation: all fields occupy 2 bytes each.
// - move is stored as the 16-bit encoded move code
// - policy, U, Q are stored as IEEE-754 half-precision (requires compiler support for _Float16)
struct CompactMove16 {
    std::uint16_t code{chess::Move::NO_MOVE};

    CompactMove16() = default;
    explicit CompactMove16(std::uint16_t c) : code(c) {}
    CompactMove16(const chess::Move &m) : code(m.move()) {}

    CompactMove16 &operator=(const chess::Move &m) {
        code = m.move();
        return *this;
    }

    operator chess::Move() const { return chess::Move(code); }

    bool operator==(const chess::Move &m) const { return code == m.move(); }
    bool operator!=(const chess::Move &m) const { return code != m.move(); }
};

using f16 = _Float16;

struct LKSPolicyEntry {
    CompactMove16 move{};
    f16 policy{static_cast<f16>(0.0f)};
    f16 U{static_cast<f16>(0.0f)};
    f16 Q{static_cast<f16>(0.0f)};
};

// Light-weight search node for LKS
struct LKSNode {
    float value{0.0f};                    // scalar evaluation in [-1, 1]
    std::pmr::vector<LKSPolicyEntry> policy;   // ordered list of policy moves and heads
    float U{0.0f};                        // node-level uncertainty (e.g., stddev)
    std::pmr::vector<f16> cdf;                 // CDF over hl bins (suffix sums of softmaxed hl)
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
            std::pmr::vector<f16> c,
            bool t)
        : value(v), policy(std::move(pol)), U(u), cdf(std::move(c)), terminal(t) {}
};

} // namespace engine


