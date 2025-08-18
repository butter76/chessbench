#include "syzygy_handler.hpp"

// Optional Fathom includes
#if defined(HAVE_FATHOM)
extern "C" {
    #include <tbprobe.h>
}
#endif

namespace engine {

namespace {

// Convert chess::Board to components suitable for TB probing.
// This function is implemented in-place for both stub and Fathom builds.
struct TbPositionParts {
    uint64_t whitePawns = 0, whiteKnights = 0, whiteBishops = 0, whiteRooks = 0, whiteQueens = 0, whiteKings = 0;
    uint64_t blackPawns = 0, blackKnights = 0, blackBishops = 0, blackRooks = 0, blackQueens = 0, blackKings = 0;
    bool whiteToMove = true;
    int castlingRights = 0; // bitmask as FEN (KQkq)
    int epSquare = -1;      // 0..63 or -1 if none
    int halfmoveClock = 0;  // for DTZ context decisions
};

TbPositionParts extractParts(const chess::Board &b) {
    TbPositionParts p;

    p.whiteToMove = (b.sideToMove() == chess::Color::WHITE);
    p.halfmoveClock = static_cast<int>(b.halfMoveClock());

    const auto cr = b.castlingRights();
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)) p.castlingRights |= 1;   // K
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) p.castlingRights |= 2;  // Q
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)) p.castlingRights |= 4;   // k
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) p.castlingRights |= 8;  // q

    const auto ep = b.enpassantSq();
    if (ep != chess::Square::NO_SQ) p.epSquare = ep.index();

    auto addPiece = [&](chess::Bitboard bb, uint64_t &mask) {
        while (!bb.empty()) {
            const int idx = bb.pop();
            mask |= (1ULL << idx);
        }
    };

    addPiece(b.pieces(chess::Color::WHITE, chess::PieceType::PAWN), p.whitePawns);
    addPiece(b.pieces(chess::Color::WHITE, chess::PieceType::KNIGHT), p.whiteKnights);
    addPiece(b.pieces(chess::Color::WHITE, chess::PieceType::BISHOP), p.whiteBishops);
    addPiece(b.pieces(chess::Color::WHITE, chess::PieceType::ROOK), p.whiteRooks);
    addPiece(b.pieces(chess::Color::WHITE, chess::PieceType::QUEEN), p.whiteQueens);
    addPiece(b.pieces(chess::Color::WHITE, chess::PieceType::KING), p.whiteKings);

    addPiece(b.pieces(chess::Color::BLACK, chess::PieceType::PAWN), p.blackPawns);
    addPiece(b.pieces(chess::Color::BLACK, chess::PieceType::KNIGHT), p.blackKnights);
    addPiece(b.pieces(chess::Color::BLACK, chess::PieceType::BISHOP), p.blackBishops);
    addPiece(b.pieces(chess::Color::BLACK, chess::PieceType::ROOK), p.blackRooks);
    addPiece(b.pieces(chess::Color::BLACK, chess::PieceType::QUEEN), p.blackQueens);
    addPiece(b.pieces(chess::Color::BLACK, chess::PieceType::KING), p.blackKings);

    return p;
}

} // namespace

SyzygyHandler::SyzygyHandler(const std::string &tbPath) : tbPath_(tbPath) {
#if defined(HAVE_FATHOM)
    if (!tbPath_.empty()) {
        available_ = tb_init(tbPath_.c_str()) != 0;
    } else {
        available_ = tb_init(nullptr) != 0;
    }
#else
    (void)tbPath_;
    available_ = false;
#endif
}

bool SyzygyHandler::isAvailable() const noexcept { return available_; }

std::optional<int> SyzygyHandler::probeWDL(const chess::Board &board) const {
    if (!available_) return std::nullopt;
    return probeWDLImpl(board);
}

std::optional<int> SyzygyHandler::probeDTZ(const chess::Board &board) const {
    if (!available_) return std::nullopt;
    return probeDTZImpl(board);
}

std::optional<int> SyzygyHandler::probeWDLImpl(const chess::Board &board) const {
#if defined(HAVE_FATHOM)
    const TbPositionParts p = extractParts(board);

    // Fathom tb_probe_wdl returns: 2 win, 1 cursed win, 0 draw, -1 blessed loss, -2 loss.
    // It expects bitboards for each piece set, stm, castling, ep.
    int wdl = tb_probe_wdl(p.whiteKings, p.whiteQueens, p.whiteRooks, p.whiteBishops, p.whiteKnights, p.whitePawns,
                           p.blackKings, p.blackQueens, p.blackRooks, p.blackBishops, p.blackKnights, p.blackPawns,
                           p.whiteToMove ? 1 : 0, p.castlingRights, p.epSquare);

    if (wdl == TB_RESULT_FAILED) return std::nullopt;
    return wdl; // already in +2..-2 scale
#else
    (void)board;
    return std::nullopt;
#endif
}

std::optional<int> SyzygyHandler::probeDTZImpl(const chess::Board &board) const {
#if defined(HAVE_FATHOM)
    const TbPositionParts p = extractParts(board);

    // tb_probe_root returns DTZ in plies (DTZ50'') for the best move; we call the variant
    // that only computes DTZ for the side-to-move without generating moves.
    int dtz = tb_probe_root(p.whiteKings, p.whiteQueens, p.whiteRooks, p.whiteBishops, p.whiteKnights, p.whitePawns,
                            p.blackKings, p.blackQueens, p.blackRooks, p.blackBishops, p.blackKnights, p.blackPawns,
                            p.whiteToMove ? 1 : 0, p.castlingRights, p.epSquare, nullptr);

    if (dtz == TB_RESULT_FAILED) return std::nullopt;
    return dtz; // DTZ50'' semantics
#else
    (void)board;
    return std::nullopt;
#endif
}

} // namespace engine


