#pragma once

#include "chess.hpp"

#include <optional>
#include <string>

namespace engine {

// SyzygyHandler provides WDL (+2..-2) and DTZ probing helpers.
// By default this is a no-op stub unless compiled with HAVE_FATHOM and linked
// against a Syzygy probing library (e.g. Fathom). See syzygy_handler.cpp.
class SyzygyHandler {
public:
    // tbPath: directory containing .rtbw (WDL) and .rtbz (DTZ) files.
    explicit SyzygyHandler(const std::string &tbPath = "");

    // Returns true if tablebases are initialized and probing is available.
    bool isAvailable() const noexcept;

    // Win/Draw/Loss on a 5-valued scale (+2 win, +1 cursed win, 0 draw,
    // -1 blessed loss, -2 loss). Returns std::nullopt if unavailable or position
    // not in tablebases.
    std::optional<int> probeWDL(const chess::Board &board) const;

    // DTZ50'' (distance to the next zeroing move or checkmate in plies), if available.
    // Returns std::nullopt if unavailable or position not in tablebases.
    std::optional<int> probeDTZ(const chess::Board &board) const;

private:
    std::string tbPath_;
    bool available_ = false;

    // Internal helpers. Implemented only when HAVE_FATHOM is defined.
    std::optional<int> probeWDLImpl(const chess::Board &board) const;
    std::optional<int> probeDTZImpl(const chess::Board &board) const;
};

} // namespace engine


