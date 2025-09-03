#pragma once

#include <optional>
#include <mutex>
#include <string>

#include <tbprobe.h>
#include "../third_party/chess-library/chess.hpp"

namespace engine::syzygy {

// Probe Syzygy DTZ at root for <= 7 men positions. Returns best move if available.
inline std::optional<chess::Move> probe_best_move(const chess::Board &board,
                                                       const char *tb_path = "../syzygy_tables/3-4-5/") {
    const int piece_count = static_cast<int>(board.occ().count());
    if (piece_count < 3 || piece_count > 7) return std::nullopt;

    // Skip Syzygy probing if any castling rights are available; TBs don't account for castling
    if (!board.castlingRights().isEmpty()) return std::nullopt;

    static bool tb_inited = false;
    if (!tb_inited) {
        (void)tb_init(tb_path);
        tb_inited = true;
    }

    const uint64_t bb_white   = board.us(chess::Color::WHITE).getBits();
    const uint64_t bb_black   = board.us(chess::Color::BLACK).getBits();
    const uint64_t bb_kings   = board.pieces(chess::PieceType::KING).getBits();
    const uint64_t bb_queens  = board.pieces(chess::PieceType::QUEEN).getBits();
    const uint64_t bb_rooks   = board.pieces(chess::PieceType::ROOK).getBits();
    const uint64_t bb_bishops = board.pieces(chess::PieceType::BISHOP).getBits();
    const uint64_t bb_knights = board.pieces(chess::PieceType::KNIGHT).getBits();
    const uint64_t bb_pawns   = board.pieces(chess::PieceType::PAWN).getBits();

    const unsigned rule50 = static_cast<unsigned>(board.halfMoveClock());
    const unsigned castling = 0;
    const unsigned ep = (board.enpassantSq() == chess::Square::NO_SQ)
        ? 0u
        : static_cast<unsigned>(board.enpassantSq().index());
    const bool turn_is_white = (board.sideToMove() == chess::Color::WHITE);

    unsigned tb_res = tb_probe_root(bb_white, bb_black, bb_kings, bb_queens, bb_rooks,
                                    bb_bishops, bb_knights, bb_pawns,
                                    rule50, castling, ep, turn_is_white, (unsigned*)nullptr);

    if (tb_res == TB_RESULT_FAILED) return std::nullopt;

    const int from = static_cast<int>(TB_GET_FROM(tb_res));
    const int to   = static_cast<int>(TB_GET_TO(tb_res));
    const int prm  = static_cast<int>(TB_GET_PROMOTES(tb_res));
    const bool is_ep = TB_GET_EP(tb_res) != 0;

    if (is_ep) {
        return chess::Move::make<chess::Move::ENPASSANT>(static_cast<chess::Square>(from), static_cast<chess::Square>(to));
    }

    if (prm != TB_PROMOTES_NONE) {
        chess::PieceType::underlying promo_pt = chess::PieceType::QUEEN;
        if (prm == TB_PROMOTES_QUEEN) promo_pt = chess::PieceType::QUEEN;
        else if (prm == TB_PROMOTES_ROOK) promo_pt = chess::PieceType::ROOK;
        else if (prm == TB_PROMOTES_BISHOP) promo_pt = chess::PieceType::BISHOP;
        else if (prm == TB_PROMOTES_KNIGHT) promo_pt = chess::PieceType::KNIGHT;
        return chess::Move::make<chess::Move::PROMOTION>(static_cast<chess::Square>(from), static_cast<chess::Square>(to), promo_pt);
    }

    return chess::Move::make<chess::Move::NORMAL>(static_cast<chess::Square>(from), static_cast<chess::Square>(to));
}

// Probe WDL for positions with <=7 men and convert to engine value in [-1, 0, 1].
// Blessed losses and cursed wins are treated as draws (0).
inline std::optional<float> probe_wdl_value(const chess::Board &board,
                                                  const char *tb_path = "../syzygy_tables/3-4-5/") {
    const int piece_count = static_cast<int>(board.occ().count());
    if (piece_count > 7) return std::nullopt;

    // Skip Syzygy probing if any castling rights are available; TBs don't account for castling
    if (!board.castlingRights().isEmpty()) return std::nullopt;

    static bool tb_inited = false;
    if (!tb_inited) {
        (void)tb_init(tb_path);
        tb_inited = true;
    }

    const uint64_t bb_white   = board.us(chess::Color::WHITE).getBits();
    const uint64_t bb_black   = board.us(chess::Color::BLACK).getBits();
    const uint64_t bb_kings   = board.pieces(chess::PieceType::KING).getBits();
    const uint64_t bb_queens  = board.pieces(chess::PieceType::QUEEN).getBits();
    const uint64_t bb_rooks   = board.pieces(chess::PieceType::ROOK).getBits();
    const uint64_t bb_bishops = board.pieces(chess::PieceType::BISHOP).getBits();
    const uint64_t bb_knights = board.pieces(chess::PieceType::KNIGHT).getBits();
    const uint64_t bb_pawns   = board.pieces(chess::PieceType::PAWN).getBits();

    const unsigned rule50 = static_cast<unsigned>(board.halfMoveClock());
    const unsigned castling = 0;
    const unsigned ep = (board.enpassantSq() == chess::Square::NO_SQ)
        ? 0u
        : static_cast<unsigned>(board.enpassantSq().index());
    const bool turn_is_white = (board.sideToMove() == chess::Color::WHITE);

    unsigned res = tb_probe_root(bb_white, bb_black, bb_kings, bb_queens, bb_rooks,
                                 bb_bishops, bb_knights, bb_pawns,
                                 rule50, castling, ep, turn_is_white, (unsigned*)nullptr);

    if (res == TB_RESULT_FAILED) return std::nullopt;

    if (res == TB_RESULT_STALEMATE) return 0.0f;
    if (res == TB_RESULT_CHECKMATE) return -1.0f; // side to move is checkmated

    const unsigned wdl = TB_GET_WDL(res);
    // Map to engine value per side-to-move perspective
    switch (wdl) {
        case TB_WIN:
            return 1.0f;
        case TB_LOSS:
            return -1.0f;
        case TB_BLESSED_LOSS:
        case TB_CURSED_WIN:
        case TB_DRAW:
        default:
            return 0.0f;
    }
}

} // namespace engine::syzygy


